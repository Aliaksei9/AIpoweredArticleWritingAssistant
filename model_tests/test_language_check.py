"""Proofreading contracts, exact spans, fact protection and independent API jobs."""

import asyncio
import json
from types import SimpleNamespace
import unittest

import httpx

from src.controller.editor_controller import EditorController, document_hash
from src.text_correction.language_check import LanguageCheck, inspect_sentence
from src.text_correction.errors import InvalidLLMResponse
from src.text_correction.sentences import split_sentences
from src.ui.app import create_app

TEXT = "🎮 Я незнаю почему он опаздал. Когда мы пришли домой начался дождь."


def row(sentence_id=1):
    return dict(sentence_id=sentence_id, spelling=[
        dict(token_id=2, replacement="не знаю", explanation="Не с глаголом пишется раздельно."),
        dict(token_id=5, replacement="опоздал", explanation="Правильное написание.")],
        rewrite="🎮 Я не знаю, почему он опоздал.", sentence_kind="punctuation", explanation="Нужна запятая.")


class FakeClient:
    config = SimpleNamespace(max_input_tokens=24000, max_output_tokens=8192)
    def __init__(self, rows):
        self.rows = rows; self.requests = []
    async def model_info(self):
        return SimpleNamespace(context_length=64000, max_completion_tokens=8192)
    async def complete(self, messages, schema, *, max_tokens):
        self.requests.append(messages)
        data = self.rows.pop(0) if len(self.rows) > 1 else self.rows[0]
        return dict(content=json.dumps(dict(sentences=data), ensure_ascii=False), model="fake:free", response_id="test",
                    usage=dict(prompt_tokens=10, completion_tokens=20, total_tokens=30, cost=0))


class LanguageCheckTests(unittest.IsolatedAsyncioTestCase):
    def test_overlap_uses_atomic_punctuation_edits_and_true_source_words(self):
        source = split_sentences(TEXT)[0]
        issues = inspect_sentence(source, row())
        self.assertEqual([i["kind"] for i in issues], ["spelling", "spelling", "punctuation"])
        blue = issues[-1]
        self.assertEqual(blue["original"], source.original)
        self.assertEqual(blue["rewrite"], "🎮 Я незнаю, почему он опаздал.")
        self.assertEqual(blue["edits"][0]["replacement"], ",")
        for issue in issues:
            self.assertEqual(TEXT[issue["start"]:issue["end"]], issue["original"])

    def test_repeated_words_are_resolved_by_id_not_find_first(self):
        sentence = split_sentences("Опаздал и опаздал.")[0]
        data = dict(sentence_id=1, spelling=[dict(token_id=3, replacement="опоздал", explanation="Опечатка.")],
                    rewrite="Опаздал и опоздал.", sentence_kind="none", explanation="")
        issues = inspect_sentence(sentence, data)
        self.assertEqual(issues[0]["start"], 10)

    def test_invalid_ids_numeric_changes_and_lexical_punctuation_are_rejected(self):
        source = split_sentences(TEXT)[0]
        bad = row(); bad["spelling"][0]["token_id"] = 900
        with self.assertRaises(InvalidLLMResponse): inspect_sentence(source, bad)
        bad = row(); bad["rewrite"] += " 12"
        with self.assertRaises(InvalidLLMResponse): inspect_sentence(source, bad)
        bad = row(); bad["rewrite"] = "🎮 Я не знаю, почему он сильно опоздал."
        with self.assertRaises(InvalidLLMResponse): inspect_sentence(source, bad)

    async def test_correct_text_returns_no_issues_not_fake_errors(self):
        data = dict(sentence_id=1, spelling=[], rewrite="Всё хорошо.", sentence_kind="none", explanation="")
        result = await LanguageCheck(FakeClient([[data]])).check("Всё хорошо.")
        self.assertEqual(result["issues"], [])

    async def test_malformed_result_is_repaired_once_with_feedback(self):
        client = FakeClient([[dict(row(), sentence_id=99)], [row()]])
        result = await LanguageCheck(client).check(split_sentences(TEXT)[0].original)
        self.assertEqual(len(client.requests), 2)
        self.assertIn("validation_feedback", json.loads(client.requests[1][1]["content"]))
        self.assertEqual(result["context"]["usage"]["total_tokens"], 60)

    def test_grammar_can_change_agreement_without_marking_spelling(self):
        source = split_sentences("Моя друзья пришёл домой.")[0]
        data = dict(sentence_id=1, spelling=[], rewrite="Мои друзья пришли домой.", sentence_kind="grammar", explanation="Согласование.")
        issues = inspect_sentence(source, data)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["kind"], "grammar")

    def test_numeric_compounds_are_whole_tokens_and_no_ops_are_ignored(self):
        source = split_sentences("Пост написан 12-леткой.")[0]
        data = dict(sentence_id=1, spelling=[dict(token_id=3, replacement="12-леткой", explanation="Через дефис.")],
                    rewrite=source.original, sentence_kind="none", explanation="")
        self.assertEqual(inspect_sentence(source, data), [])
        data["spelling"][0]["replacement"] = "13-леткой"
        data["rewrite"] = source.original.replace("12", "13")
        with self.assertRaises(InvalidLLMResponse): inspect_sentence(source, data)

    def test_quote_style_and_unchanged_grammar_labels_do_not_reject_the_batch(self):
        source = split_sentences('Сезон "Edgerunners" имеет логичную история.')[0]
        data = dict(sentence_id=1, spelling=[dict(token_id=5, replacement="«историю»", explanation="Нужна форма винительного падежа.")],
                    rewrite='Сезон «Edgerunners» имеет логичную историю.', sentence_kind="none", explanation="")
        issues = inspect_sentence(source, data)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["rewrite"], "историю")
        data = dict(sentence_id=1, spelling=[], rewrite=source.original, sentence_kind="grammar", explanation="Замечание без правки.")
        self.assertEqual(inspect_sentence(source, data), [])

    async def test_only_invalid_sentences_are_retried_and_partial_result_has_a_visible_warning(self):
        text = "Первый текст. Второй текст. Третий текст."
        good = dict(sentence_id=1, spelling=[], rewrite="Первый текст.", sentence_kind="none", explanation="")
        bad = dict(sentence_id=2, spelling=[dict(token_id=999, replacement="другое", explanation="Правка.")],
                   rewrite="Второй текст.", sentence_kind="none", explanation="")
        third = dict(sentence_id=3, spelling=[], rewrite="Третий текст!", sentence_kind="punctuation", explanation="Восклицание.")
        client = FakeClient([[good, bad, third], [bad]])
        result = await LanguageCheck(client).check(text)
        retry = json.loads(client.requests[1][1]["content"])
        self.assertEqual([r["sentence_id"] for r in retry["sentences_to_check"]], [2])
        self.assertEqual(result["context"]["checked_sentence_count"], 2)
        self.assertEqual(result["unchecked_sentences"][0]["sentence_id"], 2)
        self.assertEqual(result["issues"][0]["sentence_id"], 3)
        self.assertIn("частично", result["warnings"][0])

    async def test_no_valid_sentences_is_reported_as_failure_not_no_errors(self):
        bad = dict(row(), sentence_id=99)
        with self.assertRaises(InvalidLLMResponse):
            await LanguageCheck(FakeClient([[bad], [bad]])).check(split_sentences(TEXT)[0].original)

    async def test_missing_sentence_is_retried_without_discarding_unchanged_valid_sentence(self):
        first = dict(sentence_id=1, spelling=[], rewrite="Первый текст.", sentence_kind="none", explanation="")
        second = dict(sentence_id=2, spelling=[], rewrite="Второй текст.", sentence_kind="none", explanation="")
        client = FakeClient([[first], [second]])
        result = await LanguageCheck(client).check("Первый текст. Второй текст.")
        self.assertEqual(result["context"]["checked_sentence_count"], 2)
        self.assertEqual(result["unchecked_sentences"], [])
        self.assertEqual(len(json.loads(client.requests[1][1]["content"])["sentences_to_check"]), 1)

    async def test_language_api_is_independent_of_models_and_editorial_job(self):
        release = asyncio.Event()
        async def language_runner(text):
            return dict(issues=inspect_sentence(split_sentences(text)[0], row()))
        async def slow_editorial(text, topic, *, on_prepared=None):
            await release.wait()
        controller = EditorController(language_runner=language_runner)
        controller.correct = slow_editorial
        app = create_app(controller)
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://testserver") as client:
                editorial = app.state.jobs.create("Текст.", None)
                text = split_sentences(TEXT)[0].original
                created = await client.post("/api/language-checks", json=dict(text=text))
                self.assertEqual(created.status_code, 202)
                identifier = created.json()["id"]
                await app.state.language_jobs.jobs[identifier].task
                result = (await client.get(f"/api/language-checks/{identifier}")).json()["result"]
                self.assertEqual(result["text_hash"], document_hash(text))
                self.assertEqual(result["offset_unit"], "utf16")
                self.assertEqual(result["issues"][0]["start"], text.index("незнаю")+1)
                self.assertIsNone(controller.engine)
                self.assertNotEqual(app.state.jobs.get(editorial["id"])["status"], "cancelled")
                forbidden = await client.post("/api/language-checks", json=dict(text=text), headers={"Origin": "https://other.invalid"})
                self.assertEqual(forbidden.status_code, 403)
        finally:
            await app.state.jobs.aclose(); await app.state.language_jobs.aclose(); await controller.aclose()
