"""Source extension, shared lifecycle, CLI routing and failed-run reports."""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import AsyncMock, Mock, patch

from src.database.bd_collection import BDdataCollection, CollectionOptions, CollectionResult
from src.database.cli import parse_args, run_cli
from src.database.collection_service import CollectionService, read_resources
from src.database.resources import SourceAdapter, SourceRegistry, TelegramSource


class FakeCollector(BDdataCollection):
    def __init__(self, options):
        self.options = options
        self.calls = []
        self.result = CollectionResult()

    @property
    def last_result(self):
        return self.result

    async def update(self, resource, limit=1000):
        self.calls.append((resource, limit))
        self.result = CollectionResult(created=99, updated=1,
            deleted_expired=3 if len(self.calls) == 1 else 0, resource=dict(name=resource),
            window=dict(as_of=self.options.as_of), details=dict(scanned_items=101))
        if self.on_progress:
            self.on_progress(100)
        if resource == "fail":
            raise RuntimeError("Source request failed")
        if resource == "cancel":
            raise asyncio.CancelledError()
        return self.result.processed


class FakeSource(SourceAdapter):
    code = "WEB"

    def __init__(self, events):
        self.events = events
        self.contexts = []

    def validate_configuration(self):
        self.events.append("configuration")

    @asynccontextmanager
    async def open(self, context):
        self.contexts.append(context)
        self.events.append("client_open")
        self.collector = FakeCollector(context.options)
        try:
            yield self.collector
        finally:
            self.events.append("client_close")


class FakeTelegramSource(FakeSource):
    code = "TG"

    def normalize_resource(self, resource):
        return TelegramSource().normalize_resource(resource)


class CollectionServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.events, self.logs = [], []
        self.connection, self.embedder = object(), object()
        events = self.events
        connection = self.connection
        class ConnectionContext:
            async def __aenter__(self):
                events.append("database_open")
                return connection
            async def __aexit__(self, *_):
                events.append("database_close")
        self.connect = AsyncMock(return_value=ConnectionContext())
        self.load_embedder = AsyncMock(return_value=self.embedder)
        self.service = CollectionService(connect=self.connect, embedder_loader=self.load_embedder, log=self.logs.append)
        self.source = FakeSource(events)
        self.options = CollectionOptions(as_of=datetime(2026, 10, 23, tzinfo=timezone.utc))

    async def test_new_source_uses_one_client_and_model_without_telegram_specific_state(self):
        result = await self.service.update_sources("test-dsn", [self.source], ["feed-a", "feed-b"], self.options)
        self.assertEqual(self.events, ["configuration", "database_open", "client_open", "client_close", "database_close"])
        self.connect.assert_awaited_once_with("test-dsn")
        self.load_embedder.assert_awaited_once()
        self.assertEqual(self.source.collector.calls, [("feed-a", 1000), ("feed-b", 1000)])
        self.assertIs(self.source.contexts[0].options, self.options)
        self.assertIs(self.source.contexts[0].connection, self.connection)
        self.assertEqual(result["deleted_expired"], 3)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["selected_sources"], ["WEB"])
        self.assertEqual(result["sources"][0]["source"], "WEB")
        self.assertEqual(result["resources"][0]["source"], "WEB")
        self.assertNotIn("channels", result)
        self.assertEqual(result["resources"][0]["details"], dict(scanned_items=101))
        self.assertEqual({r["window"]["as_of"] for r in result["resources"]}, {self.options.as_of.isoformat()})
        self.assertIn("feed-a: processed 100", self.logs)
        self.assertIn("feed-b: processed 100", self.logs)

    async def test_failure_keeps_partial_stats_and_closes_both_contexts(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "nested/report.json"
            with self.assertRaisesRegex(RuntimeError, "Source request failed"):
                await self.service.update_sources("dsn", [self.source], ["first", "fail", "never"], self.options, path)
            report = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["deleted_expired"], 3)
        self.assertEqual([r["address"] for r in report["resources"]], ["first", "fail"])
        self.assertEqual(report["resources"][1]["processed"], 100)
        self.assertEqual(report["resources"][1]["status"], "failed")
        self.assertEqual(report["sources"][0]["status"], "failed")
        self.assertEqual(report["sources"][0]["resources"], report["resources"])
        self.assertIn("finished_at", report)
        self.assertEqual(self.events[-2:], ["client_close", "database_close"])

    async def test_configuration_failure_is_reported_before_database_and_model_access(self):
        source = TelegramSource()
        with TemporaryDirectory() as folder, patch.dict(os.environ, {}, clear=True):
            path = Path(folder) / "report.json"
            with self.assertRaisesRegex(ValueError, "TG_API_ID"):
                await self.service.update_sources("dsn", [source], ["@channel"], self.options, path)
            report = json.loads(path.read_text(encoding="utf-8"))
        self.connect.assert_not_awaited()
        self.load_embedder.assert_not_awaited()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["resources"], [])
        self.assertEqual(report["selected_sources"], ["TG"])
        self.assertEqual(report["sources"][0]["status"], "failed")

    async def test_resource_file_failure_is_saved_before_database_and_model_access(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "report.json"
            with self.assertRaises(FileNotFoundError):
                await self.service.update_sources(
                    "dsn", [self.source], [], self.options, path,
                    resources_file=Path(folder) / "missing.txt"
                )
            report = json.loads(path.read_text(encoding="utf-8"))
        self.connect.assert_not_awaited()
        self.load_embedder.assert_not_awaited()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["resources"], [])
        self.assertEqual(report["sources"][0]["status"], "failed")
        self.assertIn("FileNotFoundError", report["sources"][0]["error"])
        self.assert_no_legacy_report_keys(report)

    async def test_database_and_client_setup_failures_are_reported(self):
        for stage in ("database", "client"):
            with self.subTest(stage=stage), TemporaryDirectory() as folder:
                path = Path(folder) / "report.json"
                if stage == "database":
                    self.connect.side_effect = RuntimeError("database unavailable")
                else:
                    self.connect.side_effect = None
                    @asynccontextmanager
                    async def failed_client(context):
                        raise RuntimeError("client unavailable")
                        yield
                    self.source.open = failed_client
                with self.assertRaises(RuntimeError):
                    await self.service.update_sources("dsn", [self.source], ["feed"], self.options, path)
                report = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(report["status"], "failed")
                self.assertEqual(report["sources"][0]["status"], "failed")
                self.assertEqual(report["resources"], [])
                if stage == "database":
                    self.load_embedder.assert_not_awaited()
        self.assertEqual(self.events[-1], "database_close")

    async def test_cancellation_closes_contexts_and_writes_the_report(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "report.json"
            with self.assertRaises(asyncio.CancelledError):
                await self.service.update_sources("dsn", [self.source], ["cancel"], self.options, path)
            report = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "cancelled")
        self.assertEqual(report["sources"][0]["status"], "cancelled")
        self.assertEqual(report["resources"][0]["status"], "cancelled")
        self.assertEqual(self.events[-2:], ["client_close", "database_close"])

    async def test_cancellation_during_model_loading_saves_source_status_and_stops_the_batch(self):
        first = FakeSource(self.events)
        first.code = "FIRST"
        self.load_embedder.side_effect = asyncio.CancelledError()
        with TemporaryDirectory() as folder:
            path = Path(folder) / "report.json"
            with self.assertRaises(asyncio.CancelledError):
                await self.service.update_sources(
                    "dsn", [first, self.source], ["feed"], self.options, path
                )
            report = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "cancelled")
        self.assertEqual([source["status"] for source in report["sources"]], ["cancelled"])
        self.assertEqual(report["resources"], [])
        self.assertEqual(self.events[-1], "database_close")
        self.assertEqual(first.contexts, [])
        self.assertEqual(self.source.contexts, [])
        self.connect.assert_awaited_once()

    async def test_cli_routes_registered_source_without_telegram_credentials(self):
        registry = SourceRegistry(); registry.register(TelegramSource()); registry.register(self.source)
        with patch.dict(os.environ, {"DATABASE_URL": "test-dsn"}, clear=True), \
                patch("src.database.cli.load_dotenv"), patch("src.database.cli.CollectionService", return_value=self.service):
            await run_cli(["update", "feed-a", "feed-b", "--source", "web", "--limit", "321"], registry=registry)
        self.assertEqual(self.source.collector.calls, [("feed-a", 321), ("feed-b", 321)])

    async def test_telegram_adapter_owns_client_and_passes_common_options(self):
        client = AsyncMock()
        client.__aenter__.return_value = client
        collector = Mock()
        from src.database.resources import CollectionContext
        context = CollectionContext(self.connection, self.embedder, self.options)
        with patch("telethon.TelegramClient", return_value=client) as factory, \
                patch("src.database.resources.telegram.collector.TGdataCollection", return_value=collector) as collection_factory, \
                patch.dict(os.environ, {"TG_API_ID": "123", "TG_API_HASH": "test-hash"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "simulate failure"):
                async with TelegramSource().open(context) as actual:
                    self.assertIs(actual, collector)
                    raise RuntimeError("simulate failure")
        factory.assert_called_once()
        client.__aexit__.assert_awaited_once()
        collection_factory.assert_called_once_with(self.connection, client, self.embedder,
            max_age_days=14, as_of=self.options.as_of)

    def test_resources_are_deduplicated_after_source_normalization(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "channels.txt"
            path.write_text("\ufeff# comment\njeteed # note\n@other\nhttps://t.me/channel\n", encoding="utf-8")
            resources = read_resources(TelegramSource(), ["@jeteed", "other"], path)
        self.assertEqual(resources, ["@jeteed", "@other", "https://t.me/channel"])
        self.assertEqual(read_resources(self.source, ["feed-a"]), ["feed-a"])
        with self.assertRaises(ValueError): read_resources(self.source, [])

    def test_registry_extension_and_cli_arguments(self):
        registry = SourceRegistry(); registry.register(TelegramSource()); registry.register(self.source)
        self.assertIs(registry.get("web"), self.source)
        with self.assertRaises(ValueError): registry.register(self.source)
        with self.assertRaises(ValueError): registry.get("unknown")
        args = parse_args(["update", "jeteed", "--resources-file", "channels.txt"], registry=registry)
        self.assertEqual((args.source, args.limit, args.resources_file), (None, 1000, Path("channels.txt")))
        self.assertEqual(parse_args(["login", "--source", "TG"], registry=registry).source, "TG")
        self.assertEqual(parse_args(["login", "--source", "web"], registry=registry).source, "WEB")
        self.assertIsNone(parse_args(["login"], registry=registry).source)
        self.assertEqual(registry.select(), (registry.get("TG"), self.source))
        self.assertEqual(registry.select("web"), (self.source,))
        with self.assertRaises(ValueError): SourceRegistry().select()

    def test_cli_rejects_removed_command_and_option_names(self):
        registry = SourceRegistry(); registry.register(TelegramSource())
        for argv in (["login-tg"], ["update", "--channels-file", "channels.txt"],
                     ["update", "--require-complete-metrics"]):
            with self.subTest(argv=argv), patch("sys.stderr"), self.assertRaises(SystemExit) as error:
                parse_args(argv, registry=registry)
            self.assertEqual(error.exception.code, 2)

    async def test_cli_without_source_collects_every_adapter_with_one_shared_model_and_combined_report(self):
        first = FakeTelegramSource(self.events)
        registry = SourceRegistry(); registry.register(first); registry.register(self.source)
        with TemporaryDirectory() as folder, patch.dict(os.environ, {"DATABASE_URL": "test-dsn"}, clear=True), \
                patch("src.database.cli.load_dotenv"), patch("src.database.cli.CollectionService", return_value=self.service):
            path = Path(folder) / "report.json"
            await run_cli(["update", "feed-a", "--limit", "321", "--report", str(path)], registry=registry)
            report = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(first.collector.calls, [("@feed-a", 321)])
        self.assertEqual(self.source.collector.calls, [("feed-a", 321)])
        self.load_embedder.assert_awaited_once()
        self.assertEqual(self.connect.await_count, 2)
        self.assertIs(first.contexts[0].embedder, self.source.contexts[0].embedder)
        self.assertIs(first.contexts[0].options, self.source.contexts[0].options)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["selected_sources"], ["TG", "WEB"])
        self.assertEqual([r["source"] for r in report["sources"]], ["TG", "WEB"])
        self.assertEqual([r["source"] for r in report["resources"]], ["TG", "WEB"])
        self.assertEqual(report["deleted_expired"], 6)
        self.assertEqual(report["sources"][0]["resources"][0]["address"], "@feed-a")

    async def test_one_registered_source_uses_the_same_report_schema_as_multiple_sources(self):
        first = FakeTelegramSource(self.events)
        registry = SourceRegistry(); registry.register(first)
        with TemporaryDirectory() as folder, patch.dict(os.environ, {"DATABASE_URL": "test-dsn"}, clear=True), \
                patch("src.database.cli.load_dotenv"), patch("src.database.cli.CollectionService", return_value=self.service):
            path = Path(folder) / "report.json"
            await run_cli(["update", "feed", "--report", str(path)], registry=registry)
            report = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(report["selected_sources"], ["TG"])
        self.assertEqual([source["source"] for source in report["sources"]], ["TG"])
        self.assertEqual(report["resources"][0]["address"], "@feed")
        self.assertEqual(report["resources"][0]["source"], "TG")
        self.assertEqual(report["sources"][0]["resources"], report["resources"])
        self.assertEqual(report["limit_per_resource"], 1000)
        self.assertNotIn("source", report)
        self.assert_no_legacy_report_keys(report)

    async def test_unconfigured_source_does_not_prevent_the_remaining_sources_from_running(self):
        with TemporaryDirectory() as folder, patch.dict(os.environ, {}, clear=True):
            path = Path(folder) / "report.json"
            with self.assertRaisesRegex(ValueError, "Collection failed for sources: TG"):
                await self.service.update_sources("dsn", [TelegramSource(), self.source], ["feed"], self.options, path)
            report = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(self.source.collector.calls, [("feed", 1000)])
        self.assertEqual(report["status"], "failed")
        self.assertEqual([r["status"] for r in report["sources"]], ["failed", "complete"])
        self.assertIn("TG_API_ID", report["sources"][0]["error"])
        self.assertEqual(report["deleted_expired"], 3)
        self.load_embedder.assert_awaited_once()

    async def test_loaded_model_is_reused_when_the_first_source_client_fails(self):
        first = FakeSource(self.events)
        first.code = "FIRST"

        @asynccontextmanager
        async def failed_client(context):
            first.contexts.append(context)
            raise RuntimeError("client unavailable")
            yield

        first.open = failed_client
        with TemporaryDirectory() as folder:
            path = Path(folder) / "report.json"
            with self.assertRaisesRegex(ValueError, "Collection failed for sources: FIRST"):
                await self.service.update_sources(
                    "dsn", [first, self.source], ["feed"], self.options, path
                )
            report = json.loads(path.read_text(encoding="utf-8"))
        self.load_embedder.assert_awaited_once()
        self.assertIs(first.contexts[0].embedder, self.source.contexts[0].embedder)
        self.assertEqual(self.source.collector.calls, [("feed", 1000)])
        self.assertEqual([source["status"] for source in report["sources"]], ["failed", "complete"])
        self.assertEqual(report["resources"][0]["source"], "WEB")

    async def test_resource_normalization_failure_is_isolated_to_its_source(self):
        first = FakeTelegramSource(self.events)
        first.normalize_resource = Mock(side_effect=ValueError("Unsupported resource address"))
        with self.assertRaisesRegex(ValueError, "Collection failed for sources: TG"):
            await self.service.update_sources("dsn", [first, self.source], ["feed"], self.options)
        self.assertEqual(first.contexts, [])
        self.assertEqual(self.source.collector.calls, [("feed", 1000)])

    async def test_cancelling_a_batch_does_not_start_another_source(self):
        first = FakeSource(self.events); first.code = "FIRST"
        with TemporaryDirectory() as folder:
            path = Path(folder) / "report.json"
            with self.assertRaises(asyncio.CancelledError):
                await self.service.update_sources("dsn", [first, self.source], ["cancel"], self.options, path)
            report = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "cancelled")
        self.assertEqual(report["sources"][0]["status"], "cancelled")
        self.assertEqual(self.source.contexts, [])
        self.assertEqual(self.events[-2:], ["client_close", "database_close"])

    async def test_generic_login_selects_every_source_and_explicit_login_selects_only_one(self):
        first = FakeTelegramSource(self.events)
        first.login = AsyncMock(); self.source.login = AsyncMock()
        registry = SourceRegistry(); registry.register(first); registry.register(self.source)
        with patch("src.database.cli.load_dotenv"):
            await run_cli(["login"], registry=registry)
            first.login.assert_awaited_once(); self.source.login.assert_awaited_once()
            first.login.reset_mock(); self.source.login.reset_mock()
            await run_cli(["login", "--source", "web"], registry=registry)
            first.login.assert_not_awaited(); self.source.login.assert_awaited_once()
            first.login.reset_mock(); self.source.login.reset_mock()
            await run_cli(["login", "--source", "TG"], registry=registry)
            first.login.assert_awaited_once(); self.source.login.assert_not_awaited()

    async def test_login_failure_still_allows_other_sources_to_authorize(self):
        first = FakeTelegramSource(self.events)
        first.login = AsyncMock(side_effect=ValueError("missing credentials")); self.source.login = AsyncMock()
        registry = SourceRegistry(); registry.register(first); registry.register(self.source)
        with patch("src.database.cli.load_dotenv"), patch("builtins.print"):
            with self.assertRaisesRegex(SystemExit, "Login failed for sources: TG"):
                await run_cli(["login"], registry=registry)
        self.source.login.assert_awaited_once()

    def assert_no_legacy_report_keys(self, value):
        if isinstance(value, dict):
            self.assertTrue({"channels", "channel", "limit_per_channel"}.isdisjoint(value))
            for child in value.values():
                self.assert_no_legacy_report_keys(child)
        elif isinstance(value, list):
            for child in value:
                self.assert_no_legacy_report_keys(child)

    def test_options_are_validated_without_api_or_model_access(self):
        for values in (dict(limit=0), dict(limit=-1), dict(limit=True), dict(max_age_days=2),
                       dict(max_age_days=float("inf")), dict(max_age_days=float("nan"))):
            with self.subTest(values=values), self.assertRaises(ValueError): CollectionOptions(**values)
