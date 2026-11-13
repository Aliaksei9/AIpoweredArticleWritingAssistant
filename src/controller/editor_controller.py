"""UI-independent coordination of modules 3, 4 and 5."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import logging
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)


class ApplicationError(Exception):
    def __init__(self, message, status_code=503):
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class EditorSettings:
    root: Path = ROOT
    text_model: Path = Path("artifacts/text_models/neural_network.joblib")
    social_model: Path = Path("artifacts/social_metrics/model.joblib")
    max_examples: int = 5
    min_score: float = 5.0
    min_similarity: float = 0.6

    def path(self, value):
        return value if value.is_absolute() else self.root / value


def document_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def correction_cards(text, result):
    """Translate Python character positions to browser UTF-16 positions."""
    cards, previous_end, identifiers = [], 0, set()
    for sentence in result["sentences"]:
        start, end = sentence["start"], sentence["end"]
        identifier = str(sentence["sentence_id"])
        if (not isinstance(start, int) or not isinstance(end, int) or not previous_end <= start < end <= len(text)
                or text[start:end] != sentence["original"] or identifier in identifiers):
            raise ApplicationError("Не удалось сопоставить рекомендации с предложениями. Повторите анализ.", 502)
        cards.append(dict(id=identifier, start=len(text[:start].encode("utf-16-le"))//2,
            end=len(text[:end].encode("utf-16-le"))//2, original=sentence["original"],
            rewrite=sentence["rewrite"], recommendations=sentence["recommendations"]))
        previous_end = end
        identifiers.add(identifier)
    if not cards:
        raise ApplicationError("Модель не вернула рекомендации к предложениям.", 502)
    return cards


class ModelEngine:
    """One shared E5 instance for assessment, topic detection and retrieval."""
    def __init__(self, settings):
        import torch
        from src.assesment_conform_of_PP_text_based import AssesmentConformPPtextBased
        from src.social_metrics_prediction import SocialMetricsPredictor
        from src.database.topic_detector import ArticleTopicDetector
        torch.set_num_threads(4)
        text_path, social_path = settings.path(settings.text_model), settings.path(settings.social_model)
        if not text_path.exists() or not social_path.exists():
            raise ApplicationError("Рабочие модели не найдены. Выполните: python -m src.social_metrics_prediction train")
        self.text = AssesmentConformPPtextBased.load(text_path)
        self.social = SocialMetricsPredictor.load(social_path)
        expected = self.social.metadata.get("text_model_sha256")
        if expected and hashlib.sha256(text_path.read_bytes()).hexdigest() != expected:
            raise ApplicationError("Модели не согласованы. Переобучите пару: python -m src.social_metrics_prediction train")
        self.embedder = self.text.encoder.embedder
        self.detector = ArticleTopicDetector(self.embedder, settings.root / "src/database/topics.txt")

    def evaluate(self, text, topic, resource,published_at=None):
        vector = self.embedder.embed_document(text)
        topic = topic or self.detector.detect_from_embedding(vector)
        features = self.text.encoder.transform([text], [topic], [vector])
        score = float(self.text.model.predict(features)[0])
        metrics = self.social.predict(topic, score, resource, resource.source,
            document_embedding=vector,published_at=published_at)
        return dict(topic=topic, score=score, metrics=metrics.to_dict())


class EditorController:
    def __init__(self, settings=None, *, engine=None, resource_loader=None, correction_runner=None, language_runner=None):
        self.settings = settings or EditorSettings()
        self.engine = engine
        self._resource_loader, self._correction_runner = resource_loader, correction_runner
        self._language_runner = language_runner
        self._initializing, self._computing = asyncio.Lock(), asyncio.Lock()
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="editor-models")
        self._catalog = None
        self._catalog_time = 0
        self._database_available = False
        self._client = None

    async def _sync(self, function, *args):
        return await asyncio.get_running_loop().run_in_executor(self._worker, function, *args)

    def _llm_client(self, missing_key_message):
        from src.text_correction import OpenRouterClient, OpenRouterConfig
        config = OpenRouterConfig.from_env()
        if not config.api_key:
            raise ApplicationError(missing_key_message)
        if self._client is None:
            self._client = OpenRouterClient(config)
        return self._client

    async def ensure_ready(self):
        if self.engine is None:
            async with self._initializing:
                if self.engine is None:
                    try:
                        self.engine = await self._sync(ModelEngine, self.settings)
                    except ApplicationError:
                        raise
                    except Exception:
                        raise ApplicationError("Не удалось загрузить модели. Проверьте совместимость артефактов с настроенной моделью эмбеддингов.") from None
        return self.engine

    def _dsn(self):
        from psycopg.conninfo import make_conninfo
        dsn = os.getenv("DATABASE_URL")
        if not dsn:
            raise ApplicationError("Для рекомендаций укажите DATABASE_URL в .env и запустите PostgreSQL в Docker.")
        return make_conninfo(dsn, connect_timeout=3)

    async def _read_resources(self):
        if self._resource_loader:
            return await self._resource_loader()
        from src.database.connection import connect_database
        from src.database.table_classes import ResourceTable
        from src.social_metrics_prediction import ResourceInfo
        async with await connect_database(self._dsn()) as conn:
            rows = await ResourceTable(conn).list_prediction_resources()
        return [ResourceInfo.from_database(r) for r in rows if r["reader_count"] and r["reader_count"] > 0]

    async def resources(self, refresh=False):
        engine = await self.ensure_ready()
        if refresh or self._catalog is None or time.monotonic()-self._catalog_time > 60:
            try:
                catalog = await self._read_resources()
                self._database_available = True
            except Exception:
                catalog = list(engine.social.resources)
                self._database_available = False
            self._catalog, self._catalog_time = catalog, time.monotonic()
        return self._catalog

    async def workspace(self, refresh=False):
        resources = await self.resources(refresh)
        from src.text_correction import OpenRouterConfig
        from src.social_metrics_prediction.forecast_context import FORECAST_HORIZON_DAYS
        config = OpenRouterConfig.from_env()
        return dict(resources=[asdict(r) for r in resources],
            topics=[t.name for t in self.engine.detector.topics],
            database_available=self._database_available,
            audience_source="database" if self._database_available else "model_snapshot",
            corrections_configured=bool(config.api_key), llm_model=config.model,
            free_only=config.free_only,forecast=dict(horizon_days=FORECAST_HORIZON_DAYS,fixed_horizon=True,
                extended_features=bool(getattr(self.engine.social,'uses_extended_features',False))))

    async def evaluate(self, text, topic, resource_id, source,published_at=None):
        from src.social_metrics_prediction.resources import resolve_resource
        resources = await self.resources()
        try:
            resource = resolve_resource(resources, resource_id, source)
        except ValueError:
            raise ApplicationError("Выберите доступный канал и тип ресурса.", 422) from None
        if topic is not None and topic not in [t.name for t in self.engine.detector.topics]:
            raise ApplicationError("Выберите тему из списка или автоматическое определение.", 422)
        from src.social_metrics_prediction.forecast_context import forecast_context
        try:context=forecast_context(published_at)
        except ValueError as error:raise ApplicationError(str(error),422) from error
        extended=bool(getattr(self.engine.social,'uses_extended_features',False))
        async with self._computing:
            result = (await self._sync(self.engine.evaluate,text,topic,resource,context['published_at'])
                      if extended else await self._sync(self.engine.evaluate,text,topic,resource))
        return dict(result, text_hash=document_hash(text), resource=asdict(resource),
                    audience_source="database" if self._database_available else "model_snapshot",
                    forecast=dict(context,horizon_modelled=extended,
                        horizon_sensitive_metrics=(self.engine.social.horizon_metrics_for(resource,source)
                            if hasattr(self.engine.social,'horizon_metrics_for') else
                            list(getattr(self.engine.social,'horizon_sensitive_metrics',[])))))

    async def correct(self, text, topic, *, on_prepared=None):
        await self.ensure_ready()
        if self._correction_runner:
            raw = await self._correction_runner(text, topic)
        else:
            from src.database.connection import connect_database
            from src.database.table_classes import ArticleVectorTable
            from src.database.model_embedding import IncompatibleEmbeddingError
            from src.text_correction import LLManalysis, SearchSimillar, TextCorrectionModule
            from src.text_correction.errors import TextCorrectionError, explanation_for_user
            client = self._llm_client("Для рекомендаций добавьте OPENROUTER_API_KEY в .env и перезапустите приложение.")
            try:
                await client.model_info()
                try:
                    conn = await connect_database(self._dsn())
                except ApplicationError:
                    raise
                except Exception:
                    raise ApplicationError("Не удалось подключиться к БД. Запустите PostgreSQL в Docker и повторите поиск.") from None
                async with conn:
                    search = SearchSimillar(ArticleVectorTable(conn), self.engine.embedder, self.engine.detector,
                                            min_score=self.settings.min_score, min_similarity=self.settings.min_similarity)
                    analysis = LLManalysis(client, max_examples=self.settings.max_examples)
                    async with self._computing:
                        found, plan = await TextCorrectionModule(search, analysis).prepare(text, topic)
                if on_prepared:
                    on_prepared(dict(retrieval=found.retrieval_metadata(), warnings=plan.context["warnings"]))
                raw = await analysis.execute(plan)
                raw["retrieval"] = found.retrieval_metadata()
            except IncompatibleEmbeddingError:
                raise ApplicationError(
                    "Векторы в БД созданы другой моделью или их происхождение неизвестно. "
                    "Выполните: python -m src.database reembed", 503
                ) from None
            except TextCorrectionError as error:
                logger.warning("Text correction failed (%s): %s", type(error).__name__, error)
                raise ApplicationError(explanation_for_user(error), 502) from None
        return dict(text_hash=document_hash(text), topic=raw["topic"], cards=correction_cards(text, raw),
            offset_unit="utf16", references=raw.get("references", []), context=raw.get("context", {}),
            retrieval=raw.get("retrieval", {}), warnings=list(dict.fromkeys([
                *raw.get("retrieval", {}).get("warnings", []), *raw.get("context", {}).get("warnings", [])])))

    async def check_language(self, text, topic=None, *, on_prepared=None):
        from src.text_correction import LanguageCheck
        from src.text_correction.errors import TextCorrectionError, explanation_for_user
        if self._language_runner:
            raw = await self._language_runner(text)
        else:
            client = self._llm_client("Для проверки добавьте OPENROUTER_API_KEY в .env и перезапустите приложение.")
            try:
                raw = await LanguageCheck(client).check(text)
            except TextCorrectionError as error:
                logger.warning("Language check failed (%s): %s", type(error).__name__, error)
                raise ApplicationError(explanation_for_user(error), 502) from None
        def utf16(position):
            return len(text[:position].encode("utf-16-le"))//2
        issues = []
        for issue in raw["issues"]:
            item = dict(issue, start=utf16(issue["start"]), end=utf16(issue["end"]))
            item["edits"] = [dict(e, start=utf16(e["start"]), end=utf16(e["end"])) for e in issue["edits"]]
            issues.append(item)
        unchecked = [dict(s, start=utf16(s["start"]), end=utf16(s["end"])) for s in raw.get("unchecked_sentences", [])]
        return dict(raw, issues=issues, unchecked_sentences=unchecked, text_hash=document_hash(text), offset_unit="utf16")

    async def aclose(self):
        if self._client:
            await self._client.aclose()
        self._worker.shutdown(wait=False, cancel_futures=True)
