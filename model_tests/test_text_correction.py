"""Retrieval ranking, bounded prompts and complete structured LLM results."""

import json
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx

from src.text_correction import LLManalysis, OpenRouterClient, OpenRouterConfig, SearchSimillar, SimilarArticle, TextCorrectionModule
from src.text_correction.context import request_token_bound
from src.text_correction.errors import ContextBudgetError, InvalidLLMResponse, OpenRouterError, explanation_for_user
from src.text_correction.openrouter_client import DEFAULT_MODEL, ModelInfo
from src.text_correction.sentences import split_sentences
from src.database.table_classes import ArticleVectorTable
from src.text_correction.factual_anchors import validate_rewrite_anchors


def reference(index=1, text="Популярная статья с конкретным началом и понятной структурой."):
    return SimilarArticle(str(index), text, "Игры", 8, .9, 1, .91, "Канал", "TG")


class FakeLLM:
    def __init__(self, **kwargs):
        self.config = OpenRouterConfig(**kwargs)
        self.calls = []

    async def model_info(self):
        return ModelInfo(self.config.model, 65536, 16384)

    async def complete(self, messages, schema, max_tokens):
        data = json.loads(messages[1]["content"])
        self.calls.append(data)
        rows = [dict(sentence_id=s["sentence_id"], rewrite=s["original"], recommendations=[dict(aspect="clarity",
            recommendation="Сохранить конкретную формулировку.", reason="Смысл понятен читателю.",
            reference_article_ids=[])]) for s in data["sentences_to_analyze"]]
        return dict(content=json.dumps(dict(sentences=rows), ensure_ascii=False),
                    usage=dict(prompt_tokens=100, completion_tokens=100, total_tokens=200, cost=0),
                    model=self.config.model, response_id="fake")


class FakeEmbedder:
    def embed_document(self, text):
        return [1]+[0]*383

    def embed_query(self, text):
        return ([1, 0] if text.endswith("Игры") else [0, 1])+[0]*382


class TextCorrectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_text_has_priority_and_duplicates_and_original_are_excluded(self):
        rows = [dict(article_id="a", text="Близкий текст", topic="Технологии", score=6, text_similarity=.99),
                dict(article_id="b", text="Похожая тема", topic="Игры", score=10, text_similarity=.7),
                dict(article_id="c", text="похожая  тема", topic="Игры", score=7, text_similarity=.7),
                dict(article_id="d", text="ИСХОДНЫЙ ТЕКСТ", topic="Игры", score=10, text_similarity=1)]
        table = SimpleNamespace(search_popular=AsyncMock(return_value=rows), best_by_topic=AsyncMock())
        search = SearchSimillar(table, FakeEmbedder(), SimpleNamespace())
        result = await search.search("Исходный текст", "Игры")
        self.assertEqual([a.article_id for a in result.articles], ["a", "b"])
        self.assertEqual(result.candidates_found, 4)
        self.assertEqual(table.search_popular.call_args.kwargs["min_score"], 5)
        table.best_by_topic.assert_not_awaited()
        self.assertEqual(result.warnings, [])
        with self.assertRaises(ValueError):
            SearchSimillar(table, FakeEmbedder(), text_weight=.4, topic_weight=.5, popularity_weight=.1)

    async def test_no_content_matches_uses_best_exact_topic_even_below_threshold(self):
        rows = [dict(article_id="low", text="Пример с хорошей структурой.", topic="Игры", score=1, text_similarity=.2),
                dict(article_id="best", text="Лучший пример в теме.", topic="Игры", score=4, text_similarity=None),
                dict(article_id="duplicate", text="лучший  пример в теме.", topic="Игры", score=2, text_similarity=.1),
                dict(article_id="wrong", text="Другая тема.", topic="Кино", score=10, text_similarity=.3),
                dict(article_id="self", text="ИСХОДНЫЙ ТЕКСТ.", topic="Игры", score=10, text_similarity=1)]
        table = SimpleNamespace(search_popular=AsyncMock(return_value=[]), best_by_topic=AsyncMock(return_value=rows))
        search = SearchSimillar(table, FakeEmbedder(), SimpleNamespace())
        result = await search.search("Исходный текст.", "Игры", limit=2, exclude_article_id="excluded")
        self.assertEqual([a.article_id for a in result.articles], ["best", "low"])
        self.assertEqual(result.selection_method, "topic_popularity")
        self.assertEqual(result.articles[0].selection_method, "topic_popularity")
        self.assertIsNone(result.articles[0].text_similarity)
        self.assertIn("Похожих по содержанию статей не найдено", result.warnings[0])
        self.assertEqual(table.best_by_topic.call_args.kwargs["topic"], "Игры")
        self.assertEqual(table.best_by_topic.call_args.kwargs["exclude_article_id"], "excluded")
        client = FakeLLM()
        analysis = LLManalysis(client)
        answer = await analysis.analyze("Исходный текст.", result.articles, topic=result.topic)
        self.assertEqual(client.calls[0]["example_selection"], "topic_popularity")
        self.assertEqual(len(answer["sentences"]), 1)

    async def test_filtered_original_only_also_triggers_topic_fallback(self):
        table = SimpleNamespace(search_popular=AsyncMock(return_value=[dict(article_id="same", text="Исходник.",
            topic="Игры", score=8, text_similarity=1)]), best_by_topic=AsyncMock(return_value=[dict(
            article_id="other", text="Другой пример.", topic="Игры", score=3, text_similarity=.3)]))
        result = await SearchSimillar(table, FakeEmbedder(), SimpleNamespace()).search("Исходник.", "Игры")
        self.assertEqual(result.selection_method, "topic_popularity")
        self.assertEqual([a.article_id for a in result.articles], ["other"])

    async def test_empty_topic_still_generates_recommendations_with_warning(self):
        table = SimpleNamespace(search_popular=AsyncMock(return_value=[]), best_by_topic=AsyncMock(return_value=[]))
        detector = SimpleNamespace(detect_from_embedding=lambda vector: "Аниме")
        search = SearchSimillar(table, FakeEmbedder(), detector)
        client = FakeLLM()
        result = await TextCorrectionModule(search, LLManalysis(client)).correct("Новый текст.")
        self.assertEqual(result["retrieval"]["selection_method"], "general")
        self.assertIn("Похожих по содержанию статей не найдено", result["context"]["warnings"][0])
        self.assertIn("Аниме", result["context"]["warnings"][0])
        self.assertEqual(client.calls[0]["example_selection"], "general")
        self.assertEqual(result["sentences"][0]["original"], "Новый текст.")

    async def test_topic_fallback_warning_reaches_prepared_plan_and_final_json(self):
        table = SimpleNamespace(search_popular=AsyncMock(return_value=[]), best_by_topic=AsyncMock(return_value=[dict(
            article_id="example", text="Полезный пример.", topic="Игры", score=2, text_similarity=.1)]))
        module = TextCorrectionModule(SearchSimillar(table, FakeEmbedder(), SimpleNamespace()), LLManalysis(FakeLLM()))
        found, plan = await module.prepare("Новая статья.", "Игры")
        self.assertEqual(plan.context["retrieval_method"], "topic_popularity")
        self.assertEqual(plan.context["warnings"], found.warnings)
        result = await module.correct("Новая статья.", "Игры")
        self.assertEqual(result["retrieval"]["warnings"], result["context"]["warnings"])

    async def test_full_sentence_coverage_preserves_whitespace_and_offsets_across_batches(self):
        text = "  В г. Минске вышла игра. Версия 2.5 доступна сегодня!\n\nОна бесплатная.  "
        client = FakeLLM(max_output_tokens=2600)
        analysis = LLManalysis(client, batch_size=1)
        result = await analysis.analyze(text, [reference()], topic="Игры")
        expected = split_sentences(text)
        self.assertEqual(len(result["sentences"]), len(expected))
        self.assertGreater(len(client.calls), 1)
        self.assertEqual(result["rewritten_text"], text)
        for sentence in result["sentences"]:
            self.assertEqual(text[sentence["start"]:sentence["end"]], sentence["original"])
        self.assertTrue(all(call["full_text"] == text for call in client.calls))
        self.assertEqual(result["context"]["usage"]["cost"], 0)

    async def test_original_quotes_and_whitespace_come_from_server_not_llm(self):
        text = '  Второй сезон "Edgerunners".\n\nОн интересный.  '
        client = FakeLLM()
        analysis = LLManalysis(client)
        plan = await analysis.prepare(text, [], topic="Аниме")
        properties = plan.batches[0].schema["properties"]["sentences"]["items"]["properties"]
        self.assertNotIn("original", properties)
        result = await analysis.execute(plan)
        self.assertEqual(result["sentences"][0]["original"], 'Второй сезон "Edgerunners".')
        self.assertEqual(result["rewritten_text"], text)
        for row in result["sentences"]:
            self.assertEqual(text[row["start"]:row["end"]], row["original"])

    async def test_only_invalid_sentence_is_repaired_and_all_calls_are_counted(self):
        class RepairLLM(FakeLLM):
            async def complete(self, messages, schema, max_tokens):
                result = await super().complete(messages, schema, max_tokens)
                if len(self.calls) == 1:
                    body = json.loads(result["content"])
                    body["sentences"][1]["rewrite"] = "Фанфик написан подростком."
                    result["content"] = json.dumps(body, ensure_ascii=False)
                return result
        text = '  Аниме "Edgerunners".\nФанфик написан 12-леткой.  '
        client = RepairLLM()
        result = await LLManalysis(client).analyze(text, [reference()], topic="Аниме")
        self.assertEqual([s["sentence_id"] for s in client.calls[1]["sentences_to_analyze"]], [2])
        self.assertEqual(client.calls[1]["full_text"], text)
        self.assertTrue(client.calls[1]["validation_feedback"])
        self.assertEqual(result["rewritten_text"], text)
        self.assertEqual(result["context"]["validation_repairs"], 1)
        self.assertEqual(result["context"]["usage"]["total_tokens"], 400)

    async def test_repair_cannot_bypass_fact_validation_or_loop_forever(self):
        class InvalidFactsLLM(FakeLLM):
            async def complete(self, messages, schema, max_tokens):
                result = await super().complete(messages, schema, max_tokens)
                body = json.loads(result["content"])
                body["sentences"][0]["rewrite"] = "Фанфик написан подростком."
                result["content"] = json.dumps(body, ensure_ascii=False)
                return result
        client = InvalidFactsLLM()
        with self.assertRaisesRegex(InvalidLLMResponse, "numeric facts"):
            await LLManalysis(client).analyze("Фанфик написан 12-леткой.", [], topic="Аниме")
        self.assertEqual(len(client.calls), 2)

    async def test_repair_keeps_full_examples_and_respects_the_context_budget(self):
        analysis = LLManalysis(FakeLLM())
        text = "До 12 часов работы."
        articles = [reference(i, f"Статья {i}. "+"Подробности. "*100) for i in range(3)]
        plan = await analysis.prepare(text, articles, topic="Игры")
        # Leave room for the original request only; repair feedback must remove
        # one or more whole references instead of cutting any article text.
        plan.context["input_budget_tokens"] = plan.context["estimated_input_tokens"]
        repair, selected = analysis._repair_batch(plan, [(plan.batches[0].sentences[0], "Rewrite changed numeric facts")])
        self.assertLess(len(selected), len(plan.references))
        self.assertLessEqual(repair.estimated_input_tokens, plan.context["input_budget_tokens"])
        by_id = {a.article_id:a.text for a in articles}
        self.assertTrue(all(r["text"] == by_id[r["article_id"]] for r in selected))

    async def test_context_adapts_example_count_without_shortening_texts(self):
        text = "Новая игра вышла сегодня. Она бесплатная."
        articles = [reference(i, (f"  Статья {i}.\n"+"Подробности. 🎮 "*200+"\n  ")) for i in range(12)]
        wide = LLManalysis(FakeLLM(max_input_tokens=50000), max_examples=8, batch_size=1)
        two_examples = await wide.prepare(text, articles[:2], topic="Игры")
        client = FakeLLM(max_input_tokens=two_examples.context["estimated_input_tokens"])
        analysis = LLManalysis(client, max_examples=8, batch_size=1)
        plan = await analysis.prepare(text, articles, topic="Игры")
        self.assertEqual(len(plan.references), 2)
        self.assertGreater(plan.context["examples_skipped_context"], 0)
        self.assertEqual([r["text"] for r in plan.references], [a.text for a in articles[:2]])
        self.assertTrue(all(r["truncated"] is False for r in plan.references))
        for batch in plan.batches:
            self.assertLessEqual(request_token_bound(batch.messages, batch.schema), plan.context["input_budget_tokens"])
            data = json.loads(batch.messages[1]["content"])
            self.assertEqual(data["full_text"], text)
            self.assertEqual([r["text"] for r in data["examples"]], [a.text for a in articles[:2]])
        self.assertFalse(plan.context["warnings"])
        empty = await analysis.prepare("Текст без примеров.", [], topic="Игры")
        self.assertTrue(empty.context["warnings"])
        with self.assertRaises(ContextBudgetError):
            await LLManalysis(FakeLLM(max_input_tokens=100)).prepare("Текст.", articles, topic="Игры")

    async def test_long_example_is_complete_even_above_the_old_per_article_limit(self):
        article = reference(text="  Начало статьи.\n"+"Подробное описание. "*250+"\nКонец статьи.  ")
        self.assertGreater(len(article.text.encode("utf-8")), 3000)
        analysis = LLManalysis(FakeLLM(max_input_tokens=50000))
        plan = await analysis.prepare("Новая статья.", [article], topic="Игры")
        self.assertEqual(len(plan.references), 1)
        self.assertEqual(plan.references[0]["text"], article.text)
        self.assertFalse(plan.references[0]["truncated"])
        self.assertEqual(plan.context["examples_skipped_context"], 0)

    async def test_oversized_example_is_skipped_and_a_later_complete_one_is_used(self):
        articles = [reference(1, "Большая статья. "*10000), reference(2, "  Короткий пример.\nКонец.  ")]
        plan = await LLManalysis(FakeLLM()).prepare("Новая статья.", articles, topic="Игры")
        self.assertEqual([r["article_id"] for r in plan.references], ["2"])
        self.assertEqual(plan.references[0]["text"], articles[1].text)
        self.assertEqual(plan.context["examples_skipped_context"], 1)

    async def test_all_examples_too_large_still_produces_general_recommendations(self):
        client = FakeLLM()
        text = "Новая статья."
        result = await LLManalysis(client).analyze(text, [reference(text="Большая статья. "*10000)], topic="Игры")
        self.assertEqual(result["references"], [])
        self.assertEqual(result["context"]["examples_skipped_context"], 1)
        self.assertTrue(result["context"]["warnings"])
        self.assertEqual(client.calls[0]["full_text"], text)
        self.assertEqual(client.calls[0]["example_selection"], "general")
        self.assertEqual(result["sentences"][0]["original"], text)

    async def test_invalid_json_missing_sentences_and_invented_evidence_are_rejected(self):
        analysis = LLManalysis(FakeLLM())
        plan = await analysis.prepare("Первое предложение. Второе предложение.", [reference()], topic="Игры")
        batch = plan.batches[0]
        valid = await analysis.client.complete(batch.messages, batch.schema, max_tokens=plan.output_tokens)
        decoded = json.loads(valid["content"])
        variants = ["not JSON", '{"sentences":[],"sentences":[]}', '{"sentences":NaN}']
        missing = dict(sentences=decoded["sentences"][:1])
        changed = json.loads(valid["content"])
        changed["sentences"][0]["original"] = "Подменённое предложение."
        invented = json.loads(valid["content"])
        invented["sentences"][0]["recommendations"][0]["reference_article_ids"] = ["not-in-context"]
        variants += [json.dumps(v) for v in [missing, changed, invented]]
        for content in variants:
            with self.subTest(content=content), self.assertRaises(InvalidLLMResponse):
                analysis._validate(content, batch, {"1"})

    async def test_db_query_filters_score_similarity_and_excluded_article_before_limit(self):
        cursor = SimpleNamespace(fetchall=AsyncMock(return_value=[]))
        conn = SimpleNamespace(execute=AsyncMock(return_value=cursor))
        await ArticleVectorTable(conn).search_popular([1]+[0]*383, min_score=7, min_similarity=.8, limit=20)
        statement, params = conn.execute.call_args.args
        self.assertIn("JOIN article_metrics", statement)
        self.assertIn("popularity_pattern_score >=", statement)
        self.assertEqual((params["min_score"], params["min_similarity"], params["limit"]), (7, .8, 20))

    def test_rewriting_must_preserve_numeric_limits_and_reported_uncertainty(self):
        validate_rewrite_anchors("До 10 часов работы.", "Не более 10 часов автономной работы.")
        validate_rewrite_anchors("Обновление: https://example.org.", "Обновление доступно: https://example.org!")
        for original, rewrite in [("До 10 часов работы.", "Не менее 10 часов работы."),
                                  ("Память 16 ГБ.", "Память 32 ГБ."),
                                  ("Производитель обещает до 10 часов.", "Производитель гарантирует до 10 часов.")]:
            with self.subTest(rewrite=rewrite), self.assertRaises(InvalidLLMResponse):
                validate_rewrite_anchors(original, rewrite)

    async def test_topic_query_sorts_by_score_without_content_or_score_cutoff(self):
        cursor = SimpleNamespace(fetchall=AsyncMock(return_value=[]))
        conn = SimpleNamespace(execute=AsyncMock(return_value=cursor))
        await ArticleVectorTable(conn).best_by_topic([1]+[0]*383, topic="Аниме", limit=10)
        statement, params = conn.execute.call_args.args
        self.assertIn("t.topic = %(topic)s", statement)
        self.assertIn("popularity_pattern_score IS NOT NULL", statement)
        self.assertIn("ORDER BY m.popularity_pattern_score DESC", statement)
        self.assertIn("LEFT JOIN article_vectors", statement)
        self.assertNotIn("min_similarity", params)
        self.assertNotIn("min_score", params)
        self.assertEqual((params["topic"], params["limit"]), ("Аниме", 10))


class OpenRouterTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def model(price="0"):
        return dict(id=DEFAULT_MODEL, context_length=65536,
            top_provider=dict(context_length=32768, max_completion_tokens=8192),
            supported_parameters=["structured_outputs", "response_format"], pricing=dict(prompt=price, completion=price))

    async def test_request_uses_strict_schema_free_routing_and_cached_catalog(self):
        requests = []
        def handler(request):
            requests.append(request)
            if request.method == "GET":
                self.assertNotIn("Authorization", request.headers)
                return httpx.Response(200, json=dict(data=[self.model()]))
            payload = json.loads(request.content)
            self.assertEqual(payload["response_format"]["type"], "json_schema")
            self.assertTrue(payload["provider"]["require_parameters"])
            self.assertEqual(payload["provider"]["max_price"]["completion"], 0)
            return httpx.Response(200, json=dict(choices=[dict(finish_reason="stop", message=dict(content='{"sentences":[]}'))]))
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
            client = OpenRouterClient(OpenRouterConfig(api_key="secret"), http_client=transport)
            self.assertEqual((await client.model_info()).context_length, 32768)
            await client.complete([], {}, max_tokens=100)
            await client.model_info()
        self.assertEqual(len(requests), 2)
        self.assertNotIn("secret", repr(client.config))

    async def test_optional_reasoning_can_be_disabled_or_enabled_explicitly(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                def handler(request):
                    if request.method == "GET":
                        model = self.model()
                        model["supported_parameters"].append("reasoning")
                        model["reasoning"] = dict(mandatory=False)
                        return httpx.Response(200, json=dict(data=[model]))
                    payload = json.loads(request.content)
                    self.assertEqual(payload["reasoning"], dict(enabled=enabled))
                    self.assertEqual(payload["provider"]["max_price"]["request"], 0)
                    return httpx.Response(200, json=dict(choices=[dict(finish_reason="stop", message=dict(content='{"sentences":[]}'))]))
                async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
                    client = OpenRouterClient(OpenRouterConfig(api_key="secret", reasoning_enabled=enabled), http_client=transport)
                    await client.complete([], {}, max_tokens=100)

    async def test_mandatory_or_unknown_reasoning_is_not_disabled(self):
        for reasoning in (None, dict(mandatory=True)):
            with self.subTest(reasoning=reasoning):
                def handler(request):
                    if request.method == "GET":
                        model = self.model()
                        model["supported_parameters"].append("reasoning")
                        if reasoning is not None:
                            model["reasoning"] = reasoning
                        return httpx.Response(200, json=dict(data=[model]))
                    self.assertNotIn("reasoning", json.loads(request.content))
                    return httpx.Response(200, json=dict(choices=[dict(finish_reason="stop", message=dict(content='{"sentences":[]}'))]))
                async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
                    client = OpenRouterClient(OpenRouterConfig(api_key="secret"), http_client=transport)
                    await client.complete([], {}, max_tokens=100)

    async def test_transient_failure_uses_a_verified_free_fallback(self):
        posted = []
        def handler(request):
            if request.method == "GET":
                primary, secondary = self.model(), self.model()
                secondary["id"] = "test/fallback:free"
                return httpx.Response(200, json=dict(data=[primary, secondary]))
            payload = json.loads(request.content)
            posted.append(payload["model"])
            self.assertEqual(payload["provider"]["max_price"], dict(prompt=0, completion=0, request=0))
            if payload["model"] == DEFAULT_MODEL:
                return httpx.Response(502, json=dict(error=dict(code=502, message="unavailable")))
            return httpx.Response(200, json=dict(model="test/fallback:free", choices=[dict(finish_reason="stop", message=dict(content='{"sentences":[]}'))]))
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
            client = OpenRouterClient(OpenRouterConfig(api_key="secret", retries=0,
                fallback_models=("test/fallback:free",)), http_client=transport)
            result = await client.complete([], {}, max_tokens=100)
        self.assertEqual(posted, [DEFAULT_MODEL, "test/fallback:free"])
        self.assertEqual(result["fallback_from"], DEFAULT_MODEL)

    async def test_paid_or_too_small_fallback_is_skipped_before_generation(self):
        posted = []
        def handler(request):
            if request.method == "GET":
                models = [self.model()]
                for name, price, context in [("paid", "0.01", 32768), ("small", "0", 500), ("free", "0", 32768)]:
                    model = self.model(price)
                    model.update(id="test/"+name, context_length=context)
                    model["top_provider"]["context_length"] = context
                    models.append(model)
                return httpx.Response(200, json=dict(data=models))
            payload = json.loads(request.content)
            posted.append(payload["model"])
            if payload["model"] == DEFAULT_MODEL:
                return httpx.Response(503, json=dict(error=dict(code=503, message="unavailable")))
            return httpx.Response(200, json=dict(choices=[dict(finish_reason="stop", message=dict(content='{"sentences":[]}'))]))
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
            client = OpenRouterClient(OpenRouterConfig(api_key="secret", retries=0,
                fallback_models=("test/paid", "test/small", "test/free")), http_client=transport)
            await client.complete([], {}, max_tokens=100)
        self.assertEqual(posted, [DEFAULT_MODEL, "test/free"])

    async def test_paid_mode_does_not_retry_a_transport_failure_on_another_model(self):
        posted = []
        def handler(request):
            if request.method == "GET":
                return httpx.Response(200, json=dict(data=[self.model()]))
            posted.append(json.loads(request.content)["model"])
            raise httpx.ReadTimeout("timed out", request=request)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
            client = OpenRouterClient(OpenRouterConfig(api_key="secret", free_only=False,
                fallback_models=("test/fallback:free",)), http_client=transport)
            with self.assertRaises(OpenRouterError):
                await client.complete([], {}, max_tokens=100)
        self.assertEqual(posted, [DEFAULT_MODEL])

    async def test_fallback_use_is_reported_in_analysis_context(self):
        class FallbackLLM(FakeLLM):
            async def complete(self, messages, schema, max_tokens):
                return dict(await super().complete(messages, schema, max_tokens),
                            model="test/fallback:free", fallback_from=DEFAULT_MODEL)
        result = await LLManalysis(FallbackLLM()).analyze("Новый текст.", [], topic="Игры")
        self.assertTrue(any("резервная бесплатная" in w for w in result["context"]["warnings"]))
        self.assertEqual(result["context"]["calls"][0]["model"], "test/fallback:free")

    def test_ui_distinguishes_timeout_rate_limit_and_validation_without_raw_errors(self):
        cases = [(OpenRouterError("OpenRouter total request timeout; secret"), "время"),
                 (OpenRouterError("secret", 429), "частоту"),
                 (OpenRouterError("secret", 401), "ключ"),
                 (OpenRouterError("secret", 502), "недоступен"),
                 (InvalidLLMResponse("Rewrite changed numeric facts secret"), "число"),
                 (ContextBudgetError("secret"), "контекст")]
        for error, expected in cases:
            with self.subTest(error=type(error).__name__, expected=expected):
                message = explanation_for_user(error)
                self.assertIn(expected, message)
                self.assertNotIn("secret", message)

    async def test_paid_model_is_rejected_before_generation(self):
        def handler(request):
            self.assertEqual(request.method, "GET")
            return httpx.Response(200, json=dict(data=[self.model("0.01")]))
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
            client = OpenRouterClient(OpenRouterConfig(api_key="secret"), http_client=transport)
            with self.assertRaisesRegex(OpenRouterError, "not free"):
                await client.complete([], {}, max_tokens=100)

    async def test_json_mode_models_are_locally_validated_without_schema_routing(self):
        def handler(request):
            if request.method == "GET":
                model = self.model()
                model["supported_parameters"] = ["response_format"]
                return httpx.Response(200, json=dict(data=[model]))
            self.assertEqual(json.loads(request.content)["response_format"], dict(type="json_object"))
            return httpx.Response(200, json=dict(choices=[dict(finish_reason="stop", message=dict(content='{"sentences":[]}'))]))
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
            client = OpenRouterClient(OpenRouterConfig(api_key="secret"), http_client=transport)
            self.assertEqual((await client.model_info()).response_format, "json_object")
            await client.complete([], {}, max_tokens=100)

    async def test_rate_limit_retries_and_non_retryable_errors_are_redacted(self):
        attempts = 0
        def handler(request):
            nonlocal attempts
            if request.method == "GET":
                return httpx.Response(200, json=dict(data=[self.model()]))
            attempts += 1
            if attempts == 1:
                return httpx.Response(429, headers={"Retry-After": "0"}, json=dict(error=dict(code=429, message="busy")))
            return httpx.Response(401, json=dict(error=dict(code=401, message="invalid secret")))
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
            client = OpenRouterClient(OpenRouterConfig(api_key="secret"), http_client=transport)
            with self.assertRaises(OpenRouterError) as caught:
                await client.complete([], {}, max_tokens=100)
            self.assertNotIn("secret", str(caught.exception))
            self.assertEqual(caught.exception.status_code, 401)
        self.assertEqual(attempts, 2)

    async def test_truncation_is_not_returned_as_a_complete_result(self):
        def handler(request):
            if request.method == "GET":
                return httpx.Response(200, json=dict(data=[self.model()]))
            return httpx.Response(200, json=dict(choices=[dict(finish_reason="length", message=dict(content="{}"))]))
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
            client = OpenRouterClient(OpenRouterConfig(api_key="secret"), http_client=transport)
            with self.assertRaisesRegex(InvalidLLMResponse, "Incomplete"):
                await client.complete([], {}, max_tokens=100)
