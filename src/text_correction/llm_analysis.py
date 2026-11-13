"""Pack retrieved articles into context and validate every sentence-level result."""

import json
from pathlib import Path

from jsonschema import Draft202012Validator

from .context import AnalysisPlan, PreparedBatch, request_token_bound, sentence_batches
from .errors import ContextBudgetError, InvalidLLMResponse
from .sentences import assemble_rewrite, split_sentences
from .search_simillar import normalized_text
from .factual_anchors import validate_rewrite_anchors
from .json_response import load_json_response


def response_schema(sentence_count):
    recommendation = dict(type="object", additionalProperties=False,
        properties=dict(aspect=dict(type="string", enum=["clarity", "structure", "engagement", "style", "grammar"]),
            recommendation=dict(type="string", minLength=1), reason=dict(type="string", minLength=1),
            reference_article_ids=dict(type="array", items=dict(type="string"))),
        required=["aspect", "recommendation", "reason", "reference_article_ids"])
    sentence = dict(type="object", additionalProperties=False,
        properties=dict(sentence_id=dict(type="integer", minimum=1),
            recommendations=dict(type="array", minItems=1, items=recommendation), rewrite=dict(type="string", minLength=1)),
        required=["sentence_id", "recommendations", "rewrite"])
    return dict(type="object", additionalProperties=False, required=["sentences"],
        properties=dict(sentences=dict(type="array", minItems=sentence_count, maxItems=sentence_count, items=sentence)))


class LLManalysis:
    def __init__(self, client, *, max_examples=8, batch_size=8,
                 prompt_path=None):
        for value in (max_examples, batch_size):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError("Example and batch limits must be positive integers")
        self.client = client
        self.max_examples, self.batch_size = max_examples, batch_size
        path = Path(prompt_path) if prompt_path else Path(__file__).with_name("prompts") / "system.txt"
        self.system_prompt = path.read_text(encoding="utf-8-sig").strip()
        if not self.system_prompt:
            raise ValueError("System prompt cannot be empty")

    def _messages(self, text, topic, batch, references, feedback=None):
        data = dict(topic=topic, full_text=text, sentences_to_analyze=[
            dict(sentence_id=s.sentence_id, original=s.original) for s in batch], examples=references,
            example_selection=(references[0].get("selection_method", "content_similarity") if references else "general"))
        if feedback:
            data["validation_feedback"] = feedback
        schema_instruction = "\nJSON Schema ответа:\n"+json.dumps(response_schema(len(batch)), ensure_ascii=False)
        return [dict(role="system", content=self.system_prompt+schema_instruction),
                dict(role="user", content=json.dumps(data, ensure_ascii=False))]

    async def prepare(self, text, articles, *, topic):
        if not isinstance(topic, str) or not topic.strip():
            raise ValueError("A topic is required for LLM analysis")
        topic = topic.strip()
        sentences = split_sentences(text)
        info = await self.client.model_info()
        output_tokens = min(self.client.config.max_output_tokens, info.max_completion_tokens)
        input_budget = min(self.client.config.max_input_tokens, info.context_length-output_tokens-256)
        if input_budget < 1:
            raise ContextBudgetError("Model context is smaller than the reserved output budget")
        batches = sentence_batches(sentences, output_tokens, self.batch_size)
        schemas = [response_schema(len(b)) for b in batches]

        def largest_prompt(references):
            return max(request_token_bound(self._messages(text, topic, b, references), schema)
                       for b, schema in zip(batches, schemas))

        if largest_prompt([]) > input_budget:
            raise ContextBudgetError("Full text and instructions exceed the input budget; increase OPENROUTER_MAX_INPUT_TOKENS or use a larger-context model")
        references, ids, texts = [], set(), set()
        source_key = normalized_text(text)
        skipped_context = 0
        for article in articles:
            if len(references) >= self.max_examples:
                break
            key = normalized_text(article.text)
            if article.article_id in ids or key in texts or key == source_key:
                continue
            ids.add(article.article_id)
            texts.add(key)
            reference = dict(article.to_dict(), truncated=False)
            # Keep the full text, including whitespace. If it cannot fit in
            # every request, try the next example without shortening this one.
            if largest_prompt([*references, reference]) > input_budget:
                skipped_context += 1
                continue
            references.append(reference)
        warnings = []
        if not references:
            warnings.append("Подходящие популярные примеры отсутствуют или не помещаются в контекст; рекомендации будут общими редакторскими.")
        prepared = []
        for batch, schema in zip(batches, schemas):
            messages = self._messages(text, topic, batch, references)
            prepared.append(PreparedBatch(batch, messages, schema, request_token_bound(messages, schema)))
        context = dict(examples_found=len(articles), examples_used=len(references), model_context_tokens=info.context_length,
            examples_skipped_context=skipped_context,
            input_budget_tokens=input_budget, estimated_input_tokens=max(b.estimated_input_tokens for b in prepared),
            output_tokens_per_batch=output_tokens, batches=len(prepared), sentence_count=len(sentences),
            token_count_method="conservative UTF-8 byte upper bound including schema and formatting reserve",
            response_format=info.response_format,
            warnings=warnings)
        return AnalysisPlan(text, topic, info.id, references, prepared, context, output_tokens)

    @staticmethod
    def _inspect(content, batch, reference_ids):
        decoded = load_json_response(content, "Language model response is not valid JSON")
        issues = list(Draft202012Validator(batch.schema).iter_errors(decoded))
        if issues:
            raise InvalidLLMResponse("Language model JSON does not match the sentence correction schema")
        rows = decoded["sentences"]
        by_id = {row["sentence_id"]: row for row in rows}
        if len(by_id) != len(rows) or set(by_id) != {s.sentence_id for s in batch.sentences}:
            raise InvalidLLMResponse("Language model skipped, duplicated or invented sentence IDs")
        results, invalid = [], []
        for sentence in batch.sentences:
            row = by_id[sentence.sentence_id]
            try:
                if not row["rewrite"].strip():
                    raise InvalidLLMResponse("Language model returned an empty rewrite")
                validate_rewrite_anchors(sentence.original, row["rewrite"])
                for recommendation in row["recommendations"]:
                    if not recommendation["recommendation"].strip() or not recommendation["reason"].strip():
                        raise InvalidLLMResponse("Language model returned an empty recommendation")
                    if not set(recommendation["reference_article_ids"]).issubset(reference_ids):
                        raise InvalidLLMResponse("Language model cited an article that was not in its context")
            except InvalidLLMResponse as error:
                invalid.append((sentence, str(error)))
                continue
            # Original text and positions belong to the server. Having the
            # model echo them can change quotes/whitespace and reject valid edits.
            results.append(dict(row, original=sentence.original, start=sentence.start, end=sentence.end))
        return results, invalid

    @classmethod
    def _validate(cls, content, batch, reference_ids):
        results, invalid = cls._inspect(content, batch, reference_ids)
        if invalid:
            raise InvalidLLMResponse(invalid[0][1])
        return results

    def _repair_batch(self, plan, invalid):
        sentences = [sentence for sentence, _ in invalid]
        schema = response_schema(len(sentences))
        feedback = [dict(sentence_id=sentence.sentence_id, problem=problem,
            instruction="Исправьте нарушение проверки. Сохраните все исходные цифры, ссылки и оговорки. Если безопасно переписать не получается, верните исходное предложение в rewrite и рекомендуйте сохранить его.")
            for sentence, problem in invalid]
        references = list(plan.references)
        while True:
            messages = self._messages(plan.text, plan.topic, sentences, references, feedback)
            bound = request_token_bound(messages, schema)
            if bound <= plan.context["input_budget_tokens"]:
                return PreparedBatch(sentences, messages, schema, bound), references
            if not references:
                raise ContextBudgetError("Full text and validation feedback exceed the input budget")
            references.pop()  # Drop an entire example; never shorten the text.

    async def execute(self, plan: AnalysisPlan):
        if plan.model != self.client.config.model:
            raise ValueError("Analysis plan belongs to another model")
        results, calls = [], []
        usage = {key: 0 for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
        costs, warnings, repairs = [], list(plan.context["warnings"]), 0

        async def complete(batch, references, attempt):
            completion = await self.client.complete(batch.messages, batch.schema, max_tokens=plan.output_tokens)
            calls.append(dict(model=completion["model"], response_id=completion["response_id"], attempt=attempt,
                sentence_ids=[s.sentence_id for s in batch.sentences], examples_used=len(references)))
            if completion.get("fallback_from"):
                warning = f"Основная модель временно недоступна; использована резервная бесплатная модель {completion['model']}."
                if warning not in warnings:
                    warnings.append(warning)
            for key in usage:
                value = completion["usage"].get(key, 0)
                if isinstance(value, int) and value >= 0:
                    usage[key] += value
            cost = completion["usage"].get("cost")
            costs.append(cost if isinstance(cost, (int, float)) and cost >= 0 else None)
            return completion

        for batch in plan.batches:
            try:
                completion = await complete(batch, plan.references, "initial")
                valid, invalid = self._inspect(completion["content"], batch, {r["article_id"] for r in plan.references})
            except InvalidLLMResponse as error:
                valid, invalid = [], [(sentence, str(error)) for sentence in batch.sentences]
            if invalid:
                repair, references = self._repair_batch(plan, invalid)
                repairs += 1
                completion = await complete(repair, references, "repair")
                valid.extend(self._validate(completion["content"], repair, {r["article_id"] for r in references}))
            results.extend(valid)
        results.sort(key=lambda row: row["start"])
        usage["cost"] = sum(costs) if all(c is not None for c in costs) else None
        return dict(topic=plan.topic, model=plan.model, references=plan.references,
            context=dict(plan.context, usage=usage, calls=calls, warnings=warnings, validation_repairs=repairs), sentences=results,
            rewritten_text=assemble_rewrite(plan.text, results))

    async def analyze(self, text, articles, *, topic):
        return await self.execute(await self.prepare(text, articles, topic=topic))
