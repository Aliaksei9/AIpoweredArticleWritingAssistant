"""Controller API contracts, Unicode spans and cancellation of remote work."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx

from src.controller import ApplicationError, EditorController
from src.controller.editor_controller import correction_cards, document_hash
from src.controller.correction_jobs import CorrectionJobs
from src.social_metrics_prediction import ResourceInfo
from src.ui.app import create_app

RESOURCE = ResourceInfo("channel-id", "Тестовый канал", 100000)


class FakeEngine:
    def __init__(self):
        self.social = SimpleNamespace(resources=[RESOURCE])
        self.detector = SimpleNamespace(topics=[SimpleNamespace(name="Игры"), SimpleNamespace(name="Технологии")])
        self.last_text = None

    def evaluate(self, text, topic, resource):
        self.last_text = text
        return dict(topic=topic or "Игры", score=2.5, metrics=dict(views=100, reactions=10, comments=1))


async def resource_loader():
    return [RESOURCE]


async def correction_runner(text, topic):
    return dict(topic=topic or "Игры", sentences=[dict(sentence_id=1, start=0, end=len(text),
        original=text, rewrite="Новая формулировка.", recommendations=[dict(aspect="clarity",
        recommendation="Уточнить формулировку.", reason="Так понятнее.", reference_article_ids=[])])])


class EditorControllerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = FakeEngine()
        self.controller = EditorController(engine=self.engine, resource_loader=resource_loader, correction_runner=correction_runner)
        self.app = create_app(self.controller)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(self.app), base_url="http://testserver")

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.app.state.jobs.aclose()
        await self.controller.aclose()

    async def test_workspace_and_prediction_use_the_selected_resource_and_original_text(self):
        workspace = (await self.client.get("/api/workspace")).json()
        self.assertEqual(workspace["resources"][0]["audience"], 100000)
        text = "  🎮 Вышла новая игра.\n"
        response = await self.client.post("/api/predictions", json=dict(text=text, resource_id=RESOURCE.id))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["text_hash"], document_hash(text))
        self.assertEqual(self.engine.last_text, text)
        self.assertEqual(response.json()["resource"]["id"], RESOURCE.id)

    async def test_db_failure_keeps_prediction_available_with_explicit_audience_snapshot(self):
        async def failed():
            raise RuntimeError("database not running")
        self.controller._resource_loader = failed
        workspace = await self.controller.workspace()
        self.assertFalse(workspace["database_available"])
        self.assertEqual(workspace["audience_source"], "model_snapshot")
        result = await self.controller.evaluate("Текст.", None, RESOURCE.id, "TG")
        self.assertEqual(result["metrics"]["views"], 100)

    async def test_proofreading_reuses_one_owned_llm_client(self):
        from src.text_correction import OpenRouterConfig
        config=OpenRouterConfig(api_key='unit-test-key')
        client=SimpleNamespace(aclose=AsyncMock())
        checker=SimpleNamespace(check=AsyncMock(return_value=dict(issues=[])))
        with patch('src.text_correction.OpenRouterConfig.from_env',return_value=config), \
                patch('src.text_correction.OpenRouterClient',return_value=client) as create, \
                patch('src.text_correction.LanguageCheck',return_value=checker):
            await self.controller.check_language('Первый текст.')
            await self.controller.check_language('Второй текст.')
            create.assert_called_once_with(config)
            self.assertEqual(checker.check.await_count,2)
            await self.controller.aclose()
            client.aclose.assert_awaited_once()

    async def test_missing_llm_key_keeps_operation_specific_errors(self):
        from src.text_correction import OpenRouterConfig
        self.controller._correction_runner=None
        with patch('src.text_correction.OpenRouterConfig.from_env',return_value=OpenRouterConfig(api_key='')):
            with self.assertRaisesRegex(ApplicationError,'Для рекомендаций'):
                await self.controller.correct('Текст.','Игры')
            with self.assertRaisesRegex(ApplicationError,'Для проверки'):
                await self.controller.check_language('Текст.')

    async def test_invalid_input_and_foreign_origin_do_not_start_work(self):
        bad = await self.client.post("/api/predictions", json=dict(text="   ", resource_id=RESOURCE.id))
        self.assertEqual(bad.status_code, 422)
        bad_resource = await self.client.post("/api/predictions", json=dict(text="Текст.", resource_id="missing"))
        self.assertEqual(bad_resource.status_code, 422)
        foreign = await self.client.post("/api/corrections", json=dict(text="Текст."), headers={"Origin": "https://other.invalid"})
        self.assertEqual(foreign.status_code, 403)
        self.assertFalse(self.app.state.jobs.jobs)

    async def test_corrections_are_async_and_return_browser_unicode_offsets(self):
        text = "🎮 Игра вышла."
        created = await self.client.post("/api/corrections", json=dict(text=text))
        self.assertEqual(created.status_code, 202)
        identifier = created.json()["id"]
        for _ in range(50):
            response = (await self.client.get(f"/api/corrections/{identifier}")).json()
            if response["status"] == "succeeded":
                break
            await asyncio.sleep(.01)
        self.assertEqual(response["status"], "succeeded")
        result = response["result"]
        self.assertEqual(result["offset_unit"], "utf16")
        self.assertEqual(result["cards"][0]["end"], len(text)+1)

    async def test_in_flight_job_can_be_cancelled_and_replaced(self):
        waiting = asyncio.Event()
        async def slow(text, topic):
            await waiting.wait()
            return await correction_runner(text, topic)
        self.controller._correction_runner = slow
        job = (await self.client.post("/api/corrections", json=dict(text="Первый текст."))).json()
        second = await self.client.post("/api/corrections", json=dict(text="Второй текст."))
        self.assertEqual(second.status_code, 409)
        cancelled = await self.client.delete(f"/api/corrections/{job['id']}")
        self.assertEqual(cancelled.json()["status"], "cancelled")
        second = await self.client.post("/api/corrections", json=dict(text="Второй текст."))
        self.assertEqual(second.status_code, 202)

    async def test_timeout_is_reported_without_blocking_predictions(self):
        async def slow(text, topic):
            await asyncio.Event().wait()
        self.controller._correction_runner = slow
        jobs = CorrectionJobs(self.controller, timeout_seconds=.01)
        job = jobs.create("Текст.", None)
        await asyncio.wait_for(jobs.jobs[job["id"]].task, 2)
        self.assertEqual(jobs.get(job["id"])["status"], "failed")
        self.assertIn("вовремя", jobs.get(job["id"])["error"])
        result = await self.controller.evaluate("Текст.", "Игры", RESOURCE.id, "TG")
        self.assertEqual(result["score"], 2.5)
        await jobs.aclose()

    async def test_api_keeps_retrieval_warnings_alongside_cards(self):
        warning = "Похожих по содержанию статей не найдено. Для примеров подобраны лучшие тексты по теме «Игры»."
        async def fallback(text, topic):
            result = await correction_runner(text, topic)
            result.update(retrieval=dict(selection_method="topic_popularity", warnings=[warning]),
                          context=dict(warnings=[warning]))
            return result
        self.controller._correction_runner = fallback
        result = await self.controller.correct("🎮 Новый текст.", "Игры")
        self.assertEqual(result["warnings"], [warning])
        self.assertEqual(result["retrieval"]["selection_method"], "topic_popularity")
        self.assertEqual(len(result["cards"]), 1)

    async def test_warning_is_visible_while_llm_runs_and_survives_failure(self):
        started, finish = asyncio.Event(), asyncio.Event()
        warning = "Похожих по содержанию статей не найдено."
        class PreparedController:
            async def correct(self, text, topic, *, on_prepared):
                on_prepared(dict(retrieval=dict(selection_method="topic_popularity"), warnings=[warning]))
                started.set()
                await finish.wait()
                raise ApplicationError("LLM недоступна", 502)
        jobs = CorrectionJobs(PreparedController())
        job = jobs.create("Текст.", "Игры")
        try:
            await asyncio.wait_for(started.wait(), 1)
            running = jobs.get(job["id"])
            self.assertEqual(running["status"], "running")
            self.assertEqual(running["warnings"], [warning])
            finish.set()
            await jobs.jobs[job["id"]].task
            failed = jobs.get(job["id"])
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(failed["warnings"], [warning])
        finally:
            await jobs.aclose()

    def test_corrupt_spans_are_rejected(self):
        with self.assertRaises(ApplicationError):
            correction_cards("Текст.", dict(sentences=[dict(sentence_id=1, start=0, end=2, original="wrong")]))
