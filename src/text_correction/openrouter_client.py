"""Small async OpenRouter client with explicit structured-output requirements."""

import asyncio
import math
import logging
import os
import ssl
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import httpx

from .errors import InvalidLLMResponse, OpenRouterError

API_URL = "https://openrouter.ai/api/v1/"
DEFAULT_MODEL = "nvidia/nemotron-3.5-super-vl:free"
DEFAULT_FALLBACK_MODELS = ("dots-studio/dots-3-note-preview:free", "google/gemma-4-26b-a4b-it:free")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OpenRouterConfig:
    api_key: str = field(default="", repr=False)
    model: str = DEFAULT_MODEL
    max_input_tokens: int = 24000
    max_output_tokens: int = 8192
    timeout_seconds: float = 90
    retries: int = 2
    free_only: bool = True
    reasoning_enabled: bool = False
    fallback_models: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("OpenRouter model must be specified")
        for name in ("max_input_tokens", "max_output_tokens"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("HTTP timeout must be positive and finite")
        if not isinstance(self.retries, int) or isinstance(self.retries, bool) or not 0 <= self.retries <= 3:
            raise ValueError("Retry count must be an integer in [0, 3]")
        if not isinstance(self.reasoning_enabled, bool):
            raise ValueError("reasoning_enabled must be a boolean")
        if not isinstance(self.fallback_models, tuple) or any(not isinstance(m, str) or not m.strip() for m in self.fallback_models):
            raise ValueError("fallback_models must be a tuple of model IDs")

    @classmethod
    def from_env(cls):
        return cls(api_key=os.getenv("OPENROUTER_API_KEY", "").strip(),
            model=os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL).strip(),
            max_input_tokens=int(os.getenv("OPENROUTER_MAX_INPUT_TOKENS", "24000")),
            max_output_tokens=int(os.getenv("OPENROUTER_MAX_OUTPUT_TOKENS", "8192")),
            timeout_seconds=float(os.getenv("OPENROUTER_TIMEOUT_SECONDS", "90")),
            retries=int(os.getenv("OPENROUTER_RETRIES", "2")),
            reasoning_enabled=os.getenv("OPENROUTER_REASONING_ENABLED", "false").lower() in ("true", "1", "yes"),
            fallback_models=tuple(dict.fromkeys(m.strip() for m in os.getenv("OPENROUTER_FALLBACK_MODELS", ",".join(DEFAULT_FALLBACK_MODELS)).split(",") if m.strip())),
            free_only=os.getenv("OPENROUTER_FREE_ONLY", "true").lower() not in ("false", "0", "no"))


@dataclass(frozen=True)
class ModelInfo:
    id: str
    context_length: int
    max_completion_tokens: int
    is_free: bool = True
    response_format: str = "json_schema"
    reasoning_optional: bool = False


class OpenRouterClient:
    def __init__(self, config: OpenRouterConfig | None = None, *, http_client=None):
        self.config = config or OpenRouterConfig.from_env()
        self._http = http_client
        self._owns_http = http_client is None
        self._model_info = None
        self._models = None
        self._fallback_clients = {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.aclose()

    async def aclose(self):
        if self._owns_http and self._http is not None:
            await self._http.aclose()
            self._http = None

    @property
    def http(self):
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=self.config.timeout_seconds, verify=ssl.create_default_context())
        return self._http

    def _safe_message(self, message):
        message = str(message)[:1000]
        return message.replace(self.config.api_key, "[redacted]") if self.config.api_key else message

    @staticmethod
    def _retry_delay(header, attempt):
        if header:
            try:
                delay = float(header)
            except ValueError:
                try:
                    delay = (parsedate_to_datetime(header)-datetime.now(timezone.utc)).total_seconds()
                except (ValueError, TypeError, OverflowError):
                    delay = 2**attempt
            if math.isfinite(delay):
                return max(0, delay)
        return float(2**attempt)

    async def _request(self, method, path, *, payload=None):
        headers = {"Accept": "application/json", "X-Title": "Coursework text correction"}
        if method == "POST":
            if not self.config.api_key:
                raise OpenRouterError("Set OPENROUTER_API_KEY in .env before calling the language model")
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        for attempt in range(self.config.retries+1):
            try:
                async with asyncio.timeout(self.config.timeout_seconds):
                    response = await self.http.request(method, API_URL+path, json=payload, headers=headers)
            except TimeoutError:
                raise OpenRouterError("OpenRouter total request timeout; retry explicitly") from None
            except httpx.RequestError as error:
                # A timed-out generation may already have been billed: do not
                # automatically submit another paid generation after transport failure.
                raise OpenRouterError(f"OpenRouter network error ({type(error).__name__}); retry explicitly") from None
            try:
                body = response.json()
            except ValueError:
                body = {}
            if not isinstance(body, dict):
                body = {}
            error = body.get("error")
            if response.is_success and not error:
                return body
            code = response.status_code
            if isinstance(error, dict):
                try:
                    code = int(error.get("code", code))
                except (ValueError, TypeError):
                    pass
            delay = self._retry_delay(response.headers.get("Retry-After"), attempt)
            if code in (408, 429, 502, 503, 504) and attempt < self.config.retries and delay <= 30:
                await asyncio.sleep(delay)
                continue
            message = error.get("message", "Request failed") if isinstance(error, dict) else "Request failed"
            metadata = error.get("metadata") if isinstance(error, dict) else None
            if isinstance(metadata, dict) and isinstance(metadata.get("provider_name"), str):
                message += f" (provider: {metadata['provider_name']})"
            raise OpenRouterError(f"OpenRouter error {code}: {self._safe_message(message)}", code)

    async def list_models(self):
        if self._models is None:
            body = await self._request("GET", "models")
            models = body.get("data")
            if not isinstance(models, list):
                raise OpenRouterError("Invalid OpenRouter model catalog")
            self._models = models
        return self._models

    async def model_info(self) -> ModelInfo:
        if self._model_info is None:
            models = await self.list_models()
            model = next((m for m in models if isinstance(m, dict) and m.get("id") == self.config.model), None)
            if model is None:
                raise OpenRouterError(f"Model {self.config.model} is not in the OpenRouter catalog")
            supported = model.get("supported_parameters") or []
            output_format = "json_schema" if "structured_outputs" in supported else "json_object"
            if "structured_outputs" not in supported and "response_format" not in supported:
                raise OpenRouterError("Choose a model supporting structured_outputs or response_format / JSON mode")
            prices = model.get("pricing") or {}
            try:
                is_free = all(float(prices.get(k, 1)) == 0 for k in ("prompt", "completion")) and float(prices.get("request", 0)) == 0
            except (TypeError, ValueError):
                is_free = False
            if self.config.free_only and not is_free:
                raise OpenRouterError("This model is not free; OPENROUTER_FREE_ONLY requires zero prompt, completion and request prices")
            provider = model.get("top_provider") or {}
            contexts = [int(n) for n in (model.get("context_length"), provider.get("context_length")) if n]
            context_length = min(contexts) if contexts else 0
            output_limit = int(provider.get("max_completion_tokens") or self.config.max_output_tokens)
            if context_length <= 0 or output_limit <= 0:
                raise OpenRouterError("Model does not advertise valid context/output limits")
            reasoning = model.get("reasoning") or {}
            reasoning_optional = "reasoning" in supported and reasoning.get("mandatory") is False
            self._model_info = ModelInfo(self.config.model, context_length, output_limit, is_free, output_format,
                                        reasoning_optional)
        return self._model_info

    async def _complete_once(self, messages, schema, *, max_tokens):
        info = await self.model_info()
        if max_tokens > min(info.max_completion_tokens, self.config.max_output_tokens):
            raise ValueError("Requested output exceeds the configured or model output limit")
        payload = dict(model=info.id, messages=messages, max_tokens=max_tokens, temperature=.2,
            stream=False, provider=dict(require_parameters=True),
            response_format=dict(type="json_schema", json_schema=dict(name="sentence_corrections", strict=True, schema=schema)))
        if self.config.free_only:
            payload["provider"]["max_price"] = dict(prompt=0, completion=0, request=0)
        if info.response_format == "json_object":
            payload["response_format"] = dict(type="json_object")
        if info.reasoning_optional:
            payload["reasoning"] = dict(enabled=self.config.reasoning_enabled)
        body = await self._request("POST", "chat/completions", payload=payload)
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise InvalidLLMResponse("OpenRouter returned no completion choices")
        choice = choices[0]
        message = choice.get("message") or {}
        if not isinstance(message, dict):
            raise InvalidLLMResponse("OpenRouter returned an invalid completion message")
        if message.get("refusal"):
            raise InvalidLLMResponse("The language model declined to analyze this text")
        if choice.get("finish_reason") != "stop":
            raise InvalidLLMResponse("Incomplete LLM response; reduce sentence batch size or increase the output budget")
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            raise InvalidLLMResponse("OpenRouter returned empty JSON content")
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        return dict(content=message["content"], usage=usage,
                    model=body.get("model", info.id), response_id=body.get("id"))

    async def complete(self, messages, schema, *, max_tokens):
        try:
            return await self._complete_once(messages, schema, max_tokens=max_tokens)
        except (OpenRouterError, InvalidLLMResponse) as error:
            retryable = (isinstance(error, InvalidLLMResponse) or error.status_code in (408, 429, 502, 503, 504)
                         or "timeout" in str(error).lower() or "network" in str(error).lower())
            # A transport failure can leave a billed request running upstream.
            # Automatic fallback is restricted to verified zero-price routing.
            if not self.config.free_only or not retryable:
                raise
            primary_error = error
        for model in self.config.fallback_models:
            if model == self.config.model:
                continue
            try:
                if model not in self._fallback_clients:
                    candidate = OpenRouterClient(replace(self.config, model=model, fallback_models=()), http_client=self.http)
                    candidate._models = self._models
                    self._fallback_clients[model] = candidate
                candidate = self._fallback_clients[model]
                info = await candidate.model_info()  # Enforces zero prices and JSON support before generation.
                from .context import request_token_bound
                if (max_tokens > min(info.max_completion_tokens, candidate.config.max_output_tokens)
                        or request_token_bound(messages, schema)+max_tokens+256 > info.context_length):
                    logger.warning("Skipping fallback model %s: insufficient context/output budget", model)
                    continue
                logger.warning("Trying free fallback model %s after %s", model, type(primary_error).__name__)
                result = await candidate._complete_once(messages, schema, max_tokens=max_tokens)
                return dict(result, fallback_from=self.config.model)
            except (OpenRouterError, InvalidLLMResponse) as error:
                logger.warning("Free fallback model %s failed (%s): %s", model, type(error).__name__, error)
                if isinstance(error, OpenRouterError) and error.status_code in (401, 403):
                    raise
        raise primary_error
