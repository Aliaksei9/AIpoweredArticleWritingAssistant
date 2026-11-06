import importlib.util
import sys
import types
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

# Allow offline unit tests while third-party packages are not installed.
if any(importlib.util.find_spec(name) is None for name in ("psycopg", "pgvector", "telethon")):
    psycopg_stub = types.ModuleType("psycopg")
    psycopg_stub.AsyncConnection = object
    rows_stub = types.ModuleType("psycopg.rows")
    rows_stub.dict_row = object
    pgvector_stub = types.ModuleType("pgvector")
    pgvector_stub.Vector = list
    pgvector_psycopg_stub = types.ModuleType("pgvector.psycopg")

    async def register_vector_stub(conn):
        pass

    pgvector_psycopg_stub.register_vector_async = register_vector_stub
    telethon_stub = types.ModuleType("telethon")
    telethon_stub.TelegramClient = object
    errors_stub = types.ModuleType("telethon.errors")
    errors_stub.RPCError = Exception
    tl_stub = types.ModuleType("telethon.tl")
    functions_stub = types.ModuleType("telethon.tl.functions")
    channels_stub = types.ModuleType("telethon.tl.functions.channels")
    channels_stub.GetFullChannelRequest = lambda channel: channel
    types_stub = types.ModuleType("telethon.tl.types")
    types_stub.Channel = object
    sys.modules.update(
        {
            "psycopg": psycopg_stub,
            "psycopg.rows": rows_stub,
            "pgvector": pgvector_stub,
            "pgvector.psycopg": pgvector_psycopg_stub,
            "telethon": telethon_stub,
            "telethon.errors": errors_stub,
            "telethon.tl": tl_stub,
            "telethon.tl.functions": functions_stub,
            "telethon.tl.functions.channels": channels_stub,
            "telethon.tl.types": types_stub,
        }
    )

from src.database.telegram import TGdataCollection, reaction_count
from src.database.bd_collection import BDdataCollection
from src.database.table_classes import ArticleMetricTable, ArticleTextTable, ArticleVectorTable
from src.database.connection import initialize_database
from src.database.ids_converter import article_id_for, resource_id_for
from src.database.model_embedding import EMBEDDING_DIMENSIONS, EmptyTextError, TextEmbedder
from src.database.vector_utils import normalize_vector
from src.database.topic_detector import ArticleTopicDetector, DEFAULT_TOPICS_PATH


class FakeTransaction:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False


class FakeConnection:
    def __init__(self):
        self.statements = []

    def transaction(self):
        return FakeTransaction()

    async def execute(self, statement, parameters=None):
        self.statements.append((statement, parameters))


class FakeCursor:
    def __init__(self, rows):
        self.rows = rows

    async def fetchone(self):
        return self.rows[0] if self.rows else None

    def __aiter__(self):
        async def iterate():
            for row in self.rows:
                yield row

        return iterate()


class FakeInitConnection(FakeConnection):
    def __init__(self, rows):
        super().__init__()
        self.rows = rows
        self.scores = {}
        self.schema_initialized = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    async def execute(self, statement, parameters=None):
        await super().execute(statement, parameters)
        if "information_schema.columns" in statement:
            return FakeCursor([{"data_type": "uuid"}])
        if statement.startswith("CREATE EXTENSION"):
            self.schema_initialized = True
        if "SELECT m.article_id" in statement:
            return FakeCursor(self.rows)
        if "UPDATE article_metrics SET popularity_pattern_score" in statement:
            score, article_id = parameters
            self.scores[article_id] = score
        return FakeCursor([])


class FakeChannel:
    id = 42
    title = "News"
    participants_count = 123


class FakeClient:
    def __init__(self, messages):
        self.messages = messages

    async def get_entity(self, resource):
        return FakeChannel()

    async def __call__(self, request):
        return SimpleNamespace(full_chat=SimpleNamespace(participants_count=124))

    async def iter_messages(self, channel, limit=None, offset_date=None):
        self.request = {"limit": limit, "offset_date": offset_date}
        for message in self.messages[:limit]:
            yield message


class FakeResources:
    async def upsert(self, resource_id, name, reader_count):
        self.last = (resource_id, name, reader_count)


class FakeTexts:
    def __init__(self):
        self.rows = {}
        self.cleanup_cutoffs = []

    async def delete_expired(self, cutoff):
        self.cleanup_cutoffs.append(cutoff)
        expired = [article_id for article_id, row in self.rows.items()
                   if row.get("published_at") is not None
                   and BDdataCollection.utc(row["published_at"]) <= cutoff]
        for article_id in expired:
            del self.rows[article_id]
        return len(expired)

    async def upsert(self, article_id, topic, text, published_at=None):
        self.rows[article_id] = {"topic": topic, "text": text, "published_at": published_at}

    async def get(self, article_id):
        return self.rows.get(article_id)

    async def update_topic(self, article_id, topic):
        self.rows[article_id]["topic"] = topic

    async def set_publication_time(self, article_id, published_at):
        self.rows[article_id]["published_at"] = published_at


class FakeSources:
    def __init__(self):
        self.rows = {}

    async def link(self, article_id, resource_id, source):
        self.rows[article_id] = {"resource_id": resource_id, "source": source}


class FakeMetrics:
    def __init__(self):
        self.calls = []

    async def upsert(self, article_id, views, reactions, comments, audience):
        self.last = (article_id, views, reactions, comments, audience)
        self.calls.append(self.last)


class FakeVectors:
    def __init__(self):
        self.rows = {}

    async def get(self, article_id):
        return self.rows.get(article_id)

    async def upsert(self, article_id, embedding):
        self.rows[article_id] = {"embedding": embedding}


class FakeEmbedder:
    def __init__(self):
        self.calls = []

    def embed_document(self, text):
        self.calls.append(text)
        return [0.1] * EMBEDDING_DIMENSIONS


class FakeTopicDetector:
    def __init__(self):
        self.topic = "Технологии"

    def detect_from_embedding(self, embedding):
        return self.topic


class CollectionTests(unittest.IsolatedAsyncioTestCase):
    def assert_sync_counts(self, collector, expected):
        result = collector.last_result
        self.assertEqual({key: getattr(result, key) for key in expected}, expected)

    async def test_last_result_snapshot_is_generic_and_does_not_keep_a_previous_failed_resource(self):
        collector = TGdataCollection(FakeConnection(), FakeClient([]), FakeEmbedder(),
            FakeTopicDetector(), as_of=datetime(2026, 10, 23, tzinfo=timezone.utc))
        collector.resources, collector.texts = FakeResources(), FakeTexts()
        with patch("src.database.telegram.Channel", FakeChannel), patch(
                "src.database.telegram.GetFullChannelRequest", lambda channel: channel):
            await collector.update("@first")
        previous = collector.last_result
        self.assertEqual(previous.resource["name"], "News")
        self.assertEqual(previous.details["scanned_messages"], 0)
        detached = collector.last_result
        detached.resource["name"] = "Modified snapshot"
        detached.details["scanned_messages"] = 999
        self.assertEqual(collector.last_result.resource["name"], "News")
        self.assertEqual(collector.last_result.details["scanned_messages"], 0)
        collector.client.get_entity = AsyncMock(side_effect=RuntimeError("unavailable"))
        with self.assertRaises(RuntimeError):
            await collector.update("@second")
        self.assertIsNone(collector.last_result.resource)
        self.assertEqual(collector.last_result.processed, 0)
        self.assertEqual(previous.resource["name"], "News")

    async def test_unembeddable_post_is_skipped_without_stopping_the_update(self):
        as_of = datetime(2026, 10, 23, tzinfo=timezone.utc)
        messages = [SimpleNamespace(id=index, raw_text=body, date=as_of - timedelta(days=3),
                                   action=None, views=10, reactions=SimpleNamespace(results=[]),
                                   replies=SimpleNamespace(replies=0))
                    for index, body in enumerate(("\u200b", "Valid article"), 1)]
        embedder = FakeEmbedder()
        embedder.embed_document = Mock(side_effect=[
            EmptyTextError("Text contains no tokens"), [0.1] * EMBEDDING_DIMENSIONS,
        ])
        collector = TGdataCollection(FakeConnection(), FakeClient(messages), embedder,
                                     FakeTopicDetector(), as_of=as_of)
        collector.resources, collector.texts = FakeResources(), FakeTexts()
        collector.sources, collector.metrics, collector.vectors = FakeSources(), FakeMetrics(), FakeVectors()
        with patch("src.database.telegram.Channel", FakeChannel), patch(
            "src.database.telegram.GetFullChannelRequest", lambda channel: channel
        ):
            self.assertEqual(await collector.update("@news", limit=1), 1)
        result = collector.last_result
        self.assertEqual(result.details["skipped_unembeddable_text"], 1)
        self.assertEqual(result.details["scanned_messages"], 2)
        self.assertEqual(result.created, 1)
        self.assertEqual(set(collector.texts.rows), {article_id_for("TG", 42, 2)})

    async def test_update_deletes_expired_articles_even_with_empty_history(self):
        as_of = datetime(2026, 10, 22, 12, tzinfo=timezone.utc)
        cutoff = as_of - timedelta(days=14)
        collector = TGdataCollection(FakeConnection(), FakeClient([]), FakeEmbedder(),
                                     FakeTopicDetector(), as_of=as_of)
        collector.texts, collector.resources = FakeTexts(), FakeResources()
        dates = [cutoff - timedelta(microseconds=1),
                 cutoff.astimezone(timezone(timedelta(hours=3))),
                 cutoff + timedelta(microseconds=1), None]
        # Include a different channel: retention applies to the whole database.
        identifiers = [article_id_for("TG", 99, index) for index in range(len(dates))]
        collector.texts.rows = {article_id: {"published_at": date}
                                for article_id, date in zip(identifiers, dates)}
        with patch("src.database.telegram.Channel", FakeChannel), patch(
            "src.database.telegram.GetFullChannelRequest", lambda channel: channel
        ):
            self.assertEqual(await collector.update("@news"), 0)
        self.assertEqual(collector.last_result.deleted_expired, 2)
        self.assertEqual(set(collector.texts.rows), set(identifiers[2:]))
        self.assertEqual(collector.embedder.calls, [])

    async def test_update_prunes_once_per_cutoff_and_uses_configured_age(self):
        as_of = datetime(2026, 10, 22, 12, tzinfo=timezone.utc)
        collector = TGdataCollection(FakeConnection(), FakeClient([]), FakeEmbedder(),
                                     FakeTopicDetector(), as_of=as_of, max_age_days=7)
        collector.texts, collector.resources = FakeTexts(), FakeResources()
        article_id = article_id_for("TG", 42, 1)
        collector.texts.rows[article_id] = {"published_at": as_of - timedelta(days=6)}
        with patch("src.database.telegram.Channel", FakeChannel), patch(
            "src.database.telegram.GetFullChannelRequest", lambda channel: channel
        ):
            await collector.update("@first")
            await collector.update("@second")
            self.assertEqual(collector.texts.cleanup_cutoffs, [as_of - timedelta(days=7)])
            self.assertIn(article_id, collector.texts.rows)
            collector.as_of = as_of + timedelta(days=1)
            await collector.update("@first")
            self.assertEqual(collector.last_result.deleted_expired, 1)
            self.assertNotIn(article_id, collector.texts.rows)
            await collector.update("@second")
            self.assertEqual(collector.last_result.deleted_expired, 0)
            self.assertEqual(len(collector.texts.cleanup_cutoffs), 2)

    async def test_failed_cleanup_is_retried_for_the_same_cutoff(self):
        collector = TGdataCollection(FakeConnection(), FakeClient([]), FakeEmbedder(),
                                     FakeTopicDetector(), as_of=datetime(2026, 10, 22, tzinfo=timezone.utc))
        collector.texts, collector.resources = FakeTexts(), FakeResources()
        collector.texts.delete_expired = AsyncMock(side_effect=[RuntimeError("delete failed"), 2])
        with patch("src.database.telegram.Channel", FakeChannel), patch(
            "src.database.telegram.GetFullChannelRequest", lambda channel: channel
        ):
            with self.assertRaisesRegex(RuntimeError, "delete failed"):
                await collector.update("@news")
            await collector.update("@news")
        self.assertEqual(collector.texts.delete_expired.await_count, 2)
        self.assertEqual(collector.last_result.deleted_expired, 2)

    async def test_complete_metrics_require_positive_views_and_keep_zero_engagement(self):
        as_of = datetime(2026, 10, 21, 12, tzinfo=timezone.utc)
        known_zero_reactions = SimpleNamespace(results=[])
        known_zero_comments = SimpleNamespace(replies=0, comments=True)
        cases = [(None, known_zero_comments, 10),
                 (known_zero_reactions, None, 10),
                 (known_zero_reactions, known_zero_comments, None),
                 (known_zero_reactions, known_zero_comments, 0),
                 (known_zero_reactions, known_zero_comments, 1)]
        messages = [SimpleNamespace(id=index, raw_text=f"Post {index}", action=None,
                                   date=as_of - timedelta(days=3), reactions=reactions,
                                   replies=comments, views=views)
                    for index, (reactions, comments, views) in enumerate(cases, start=1)]
        collector = TGdataCollection(FakeConnection(), FakeClient(messages), FakeEmbedder(),
                                     FakeTopicDetector(), as_of=as_of)
        collector.resources, collector.texts = FakeResources(), FakeTexts()
        collector.sources, collector.metrics, collector.vectors = FakeSources(), FakeMetrics(), FakeVectors()
        with patch("src.database.telegram.Channel", FakeChannel), patch(
            "src.database.telegram.GetFullChannelRequest", lambda channel: channel
        ):
            self.assertEqual(await collector.update("@news"), 1)
        self.assertEqual(collector.last_result.details["skipped_incomplete_metrics"], 4)
        self.assertEqual(collector.embedder.calls, ["Post 5"])
        self.assertEqual(collector.metrics.calls, [(article_id_for("TG", 42, 5), 1, 0, 0, 124)])

    async def test_unknown_or_zero_audience_is_rejected_before_embedding(self):
        as_of = datetime(2026, 10, 23, tzinfo=timezone.utc)
        message = SimpleNamespace(id=1, raw_text="Article", date=as_of - timedelta(days=3),
                                  action=None, views=0, reactions=SimpleNamespace(results=[]),
                                  replies=SimpleNamespace(replies=0))
        for audience in (None, 0):
            collector = TGdataCollection(FakeConnection(), FakeClient([message]), FakeEmbedder(),
                                         FakeTopicDetector(), as_of=as_of)
            collector.resources, collector.texts = FakeResources(), FakeTexts()
            collector.sources, collector.metrics, collector.vectors = FakeSources(), FakeMetrics(), FakeVectors()
            with self.subTest(audience=audience), patch("src.database.telegram.Channel", FakeChannel), patch(
                "src.database.telegram.GetFullChannelRequest", lambda channel: channel
            ), patch.object(FakeChannel, "participants_count", audience), patch.object(
                FakeClient, "__call__", AsyncMock(return_value=SimpleNamespace(
                    full_chat=SimpleNamespace(participants_count=audience)))
            ):
                self.assertEqual(await collector.update("@news"), 0)
            self.assertEqual(collector.last_result.details["skipped_incomplete_metrics"], 1)
            self.assertEqual(collector.embedder.calls, [])
            self.assertEqual(collector.metrics.calls, [])

    def test_publication_window_has_strict_two_day_and_fourteen_day_boundaries(self):
        as_of = datetime(2026, 10, 21, 12, tzinfo=timezone.utc)
        for age in (timedelta(days=2), timedelta(days=14), timedelta(days=30), timedelta(days=-1)):
            self.assertFalse(BDdataCollection.is_article_in_window(as_of - age, as_of))
        for age in (timedelta(days=2, microseconds=1), timedelta(days=14) - timedelta(microseconds=1)):
            published = as_of - age
            self.assertTrue(BDdataCollection.is_article_in_window(published, as_of))
            self.assertTrue(BDdataCollection.is_article_in_window(
                published.astimezone(timezone(timedelta(hours=3))), as_of))
        self.assertFalse(BDdataCollection.is_article_in_window(None, as_of))
        self.assertFalse(BDdataCollection.is_article_in_window(as_of - timedelta(days=8), as_of, timedelta(days=7)))
        for maximum in (2, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                TGdataCollection(FakeConnection(), FakeClient([]), FakeEmbedder(),
                                 FakeTopicDetector(), max_age_days=maximum)

    async def test_limit_counts_eligible_texts_and_history_stops_at_old_boundary(self):
        as_of = datetime(2026, 10, 21, 12, tzinfo=timezone.utc)
        messages = [SimpleNamespace(id=index, raw_text=body, date=as_of - timedelta(days=age),
                                   action=None, views=10, reactions=SimpleNamespace(results=[]),
                                   replies=SimpleNamespace(replies=0))
                    for index, (age, body) in enumerate([
                        (1, "Fresh"), (3, ""), (3, "First"), (4, "Second"),
                        (14, "Too old"), (15, "Unvisited"),
                    ], start=1)]
        collector = TGdataCollection(FakeConnection(), FakeClient(messages), FakeEmbedder(),
                                     FakeTopicDetector(), as_of=as_of)
        collector.resources, collector.texts = FakeResources(), FakeTexts()
        collector.sources, collector.metrics, collector.vectors = FakeSources(), FakeMetrics(), FakeVectors()
        with patch("src.database.telegram.Channel", FakeChannel), patch(
            "src.database.telegram.GetFullChannelRequest", lambda channel: channel
        ):
            self.assertEqual(await collector.update("@news", limit=2), 2)
            self.assertEqual(collector.embedder.calls, ["First", "Second"])
            self.assertEqual(collector.last_result.details["scanned_messages"], 4)
            self.assertEqual(collector.client.request, {"limit": None, "offset_date": as_of - timedelta(days=2)})
            self.assertEqual(await collector.update("@news", limit=1000), 2)
            self.assertEqual(collector.last_result.details["scanned_messages"], 5)
            self.assertEqual(collector.last_result.created, 0)
            self.assertEqual(collector.last_result.updated, 2)
            self.assertTrue(all(row["published_at"] is not None for row in collector.texts.rows.values()))

    async def test_update_skips_recent_and_undated_articles(self):
        as_of = datetime(2026, 10, 13, 12, tzinfo=timezone.utc)
        dates = [as_of - timedelta(hours=1), as_of - timedelta(days=2),
                 as_of - timedelta(days=3), None, as_of + timedelta(days=1)]
        messages = [SimpleNamespace(id=index, raw_text=f"Article {index}", date=date,
                                   edit_date=as_of, action=None, views=10,
                                   reactions=SimpleNamespace(results=[]), replies=SimpleNamespace(replies=0))
                    for index, date in enumerate(dates, start=1)]
        embedder = FakeEmbedder()
        collector = TGdataCollection(FakeConnection(), FakeClient(messages),
                                     embedder, FakeTopicDetector())
        collector.resources = FakeResources()
        collector.texts = FakeTexts()
        collector.sources = FakeSources()
        collector.metrics = FakeMetrics()
        collector.vectors = FakeVectors()
        with patch("src.database.telegram.Channel", FakeChannel), patch(
            "src.database.telegram.GetFullChannelRequest", lambda channel: channel
        ), patch("src.database.telegram.datetime") as clock:
            clock.now.return_value = as_of
            self.assertEqual(await collector.update("@news"), 1)
            self.assert_sync_counts(collector, {
                "created": 1, "updated": 0, "skipped_by_age": 3, "skipped_without_date": 1
            })
            self.assertEqual(await collector.update("@news"), 1)
            self.assert_sync_counts(collector, {
                "created": 0, "updated": 1, "skipped_by_age": 3, "skipped_without_date": 1
            })
            self.assertEqual(embedder.calls, ["Article 3"])
            self.assertEqual(set(collector.texts.rows), {article_id_for("TG", 42, 3)})
            self.assertEqual(set(collector.sources.rows), {article_id_for("TG", 42, 3)})
            self.assertEqual(set(collector.vectors.rows), {article_id_for("TG", 42, 3)})

            # Once publications age past the cutoff, add each at its stable ID.
            clock.now.return_value = as_of + timedelta(days=3)
            self.assertEqual(await collector.update("@news"), 3)
            self.assert_sync_counts(collector, {
                "created": 2, "updated": 1, "skipped_by_age": 1, "skipped_without_date": 1
            })
            self.assertEqual(await collector.update("@news"), 3)
            self.assertEqual(len(collector.texts.rows), 3)
            self.assertEqual(len(embedder.calls), 3)

    async def test_pgvector_driver_result_can_be_reused_for_classification(self):
        values = [1.0] + [0.0] * (EMBEDDING_DIMENSIONS - 1)

        class DriverVector:
            def to_list(self):
                return values

        article_id = article_id_for("TG", 42, 7)
        conn = SimpleNamespace(execute=AsyncMock(return_value=FakeCursor([
            {"article_id": article_id, "embedding": DriverVector()}
        ])))
        with patch("src.database.table_classes.Vector", DriverVector):
            row = await ArticleVectorTable(conn).get(article_id)
        self.assertEqual(row["embedding"], values)
        self.assertEqual(normalize_vector(row["embedding"]), values)

    async def test_topic_update_preserves_article_vector(self):
        conn = FakeConnection()
        article_id = article_id_for("TG", 42, 7)
        await ArticleTextTable(conn).update_topic(article_id, "Игры")
        self.assertEqual(len(conn.statements), 1)
        self.assertNotIn("DELETE", conn.statements[0][0])
        self.assertEqual(conn.statements[0][1], ("Игры", article_id))

    async def test_metrics_writer_saves_and_refreshes_score(self):
        conn = FakeConnection()
        metrics = ArticleMetricTable(conn)
        article_id = article_id_for("TG", 42, 7)
        await metrics.upsert(article_id, 10000, 80, 5, 100000)
        self.assertEqual(conn.statements[-1][1], (article_id, 10000, 80, 5, 100000, 0.0))
        await metrics.upsert(article_id, 10000, 160, 10, 100000)
        self.assertAlmostEqual(conn.statements[-1][1][-1], metrics.assessor.assess(10000, 160, 10, 100000))

    async def test_initialization_recalculates_existing_scores(self):
        first_id = article_id_for("TG", 42, 7)
        second_id = article_id_for("TG", 42, 8)
        rows = [
            dict(article_id=first_id, views=10000, reactions=80, comments=5, audience=100000),
            dict(article_id=second_id, views=0, reactions=0, comments=0, audience=1000),
        ]
        conn = FakeInitConnection(rows)

        class FakeDriver:
            @staticmethod
            async def connect(*args, **kwargs):
                return conn

        with patch("src.database.connection.psycopg.AsyncConnection", FakeDriver):
            count = await initialize_database("fake-dsn")
        self.assertTrue(conn.schema_initialized)
        self.assertEqual(count, 2)
        self.assertEqual(conn.scores, {first_id: 0.0, second_id: None})

    async def test_text_update_invalidates_vector(self):
        conn = FakeConnection()
        article_id = article_id_for("TG", 42, 7)
        await ArticleTextTable(conn).upsert(article_id, "New topic", "New text")
        self.assertIn("INSERT INTO article_text", conn.statements[0][0])
        self.assertIn("DELETE FROM article_vectors", conn.statements[1][0])
        self.assertEqual(conn.statements[1][1], (article_id,))

    async def test_repeated_collection_updates_same_article_and_invalidates_vector(self):
        message = SimpleNamespace(
            id=7,
            raw_text="First line\nBody",
            date=datetime.now(timezone.utc) - timedelta(days=3),
            action=None,
            views=10,
            reactions=SimpleNamespace(results=[SimpleNamespace(count=2)]),
            replies=SimpleNamespace(replies=3),
        )
        client = FakeClient([message])
        embedder = FakeEmbedder()
        topic_detector = FakeTopicDetector()
        collector = TGdataCollection(FakeConnection(), client, embedder, topic_detector)
        collector.resources = FakeResources()
        collector.texts = FakeTexts()
        collector.sources = FakeSources()
        collector.metrics = FakeMetrics()
        collector.vectors = FakeVectors()
        article_id = article_id_for("TG", 42, 7)
        resource_id = resource_id_for("TG", 42)

        with patch("src.database.telegram.Channel", FakeChannel), patch(
            "src.database.telegram.GetFullChannelRequest", lambda channel: channel
        ):
            self.assertEqual(await collector.update("@news"), 1)
            self.assert_sync_counts(collector, {
                "created": 1, "updated": 0, "skipped_by_age": 0, "skipped_without_date": 0
            })
            self.assertEqual(collector.resources.last, (resource_id, "News", 124))
            self.assertEqual(collector.metrics.last, (article_id, 10, 2, 3, 124))
            self.assertEqual(collector.sources.rows[article_id]["source"], "TG")
            self.assertEqual(embedder.calls, ["First line\nBody"])
            self.assertEqual(collector.texts.rows[article_id]["topic"], "Технологии")

            self.assertEqual(await collector.update("@news"), 1)
            self.assert_sync_counts(collector, {
                "created": 0, "updated": 1, "skipped_by_age": 0, "skipped_without_date": 0
            })
            self.assertEqual(len(embedder.calls), 1)

            topic_detector.topic = "Игры"
            saved_vector = collector.vectors.rows[article_id]
            self.assertEqual(await collector.update("@news"), 1)
            self.assertEqual(collector.texts.rows[article_id]["topic"], "Игры")
            self.assertEqual(collector.texts.rows[article_id]["text"], "First line\nBody")
            self.assertIs(collector.vectors.rows[article_id], saved_vector)
            self.assertEqual(len(embedder.calls), 1)

            del collector.vectors.rows[article_id]
            self.assertEqual(await collector.update("@news"), 1)
            self.assertEqual(len(embedder.calls), 2)

            message.raw_text = "Edited line\nBody"
            message.views = 20
            self.assertEqual(await collector.update("@news"), 1)

        self.assertEqual(len(collector.texts.rows), 1)
        self.assertEqual(collector.texts.rows[article_id]["topic"], "Игры")
        self.assertEqual(collector.metrics.last, (article_id, 20, 2, 3, 124))
        self.assertEqual(
            embedder.calls,
            ["First line\nBody", "First line\nBody", "Edited line\nBody"],
        )
        self.assertIn(article_id, collector.vectors.rows)

    def test_message_helpers(self):
        self.assertEqual(reaction_count(SimpleNamespace(reactions=None)), None)
        self.assertEqual(article_id_for("TG", 42, 7), article_id_for("TG", 42, 7))
        self.assertNotEqual(article_id_for("TG", 42, 7), article_id_for("TG", 43, 7))
        self.assertNotEqual(article_id_for("TG", 42, 7), article_id_for("DTF", 42, 7))
        with self.assertRaises(ValueError):
            ArticleVectorTable._vector([0.0, 0.0])

    def test_embedding_prefixes_and_long_text_chunks(self):
        class FakeTokenizer:
            def encode(self, text, add_special_tokens=False, **kwargs):
                return text.split()

            def decode(self, tokens):
                return " ".join(tokens)

        class FakeModel:
            tokenizer = FakeTokenizer()
            max_seq_length = 7

            def encode(self, inputs, **kwargs):
                self.inputs = inputs
                return [[1.0] + [0.0] * (EMBEDDING_DIMENSIONS - 1) for _ in inputs]

        embedder = TextEmbedder.__new__(TextEmbedder)
        embedder.model = FakeModel()
        vector = embedder.embed_document("one two three four five six seven eight")
        self.assertGreater(len(embedder.model.inputs), 1)
        self.assertTrue(all(item.startswith("passage: ") for item in embedder.model.inputs))
        self.assertEqual(len(vector), EMBEDDING_DIMENSIONS)
        self.assertAlmostEqual(vector[0], 1.0)
        embedder.embed_query("what happened")
        self.assertTrue(embedder.model.inputs[0].startswith("query: "))


class SemanticTestEmbedder:
    def __init__(self):
        self.query_calls = []

    @staticmethod
    def basis(index, scale=1.0):
        values = [0.0] * EMBEDDING_DIMENSIONS
        values[index] = scale
        return values

    def embed_query(self, text):
        self.query_calls.append(text)
        for index, name in enumerate(("Аниме", "Игры", "Технологии")):
            if f"«{name}»" in text:
                return self.basis(index, 3.0)
        raise AssertionError(f"Unexpected topic query: {text}")

    def embed_document(self, text):
        return self.basis(0, 5.0)


class TopicDetectorTests(unittest.TestCase):
    def test_selects_topic_using_cosine_similarity_and_caches_topic_vectors(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "topics.txt"
            path.write_text("Аниме | Японская анимация\nИгры | Видеоигры\n", encoding="utf-8")
            embedder = SemanticTestEmbedder()
            detector = ArticleTopicDetector(embedder, path)
            self.assertEqual(detector.detect_from_embedding(embedder.embed_document("Some full article text")), "Аниме")
            self.assertEqual(detector.detect_from_embedding(embedder.basis(1, 8.0)), "Игры")
            self.assertEqual(len(embedder.query_calls), 2)

    def test_topics_can_be_appended_without_code_changes(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "topics.txt"
            path.write_text("# Categories\nАниме\nИгры | Видеоигры\n", encoding="utf-8")
            embedder = SemanticTestEmbedder()
            original = ArticleTopicDetector(embedder, path)
            with path.open("a", encoding="utf-8") as file:
                file.write("Технологии\n")
            updated = ArticleTopicDetector(embedder, path)
            self.assertEqual(len(original.topics), 2)
            self.assertEqual(updated.detect_from_embedding(embedder.basis(2)), "Технологии")

    def test_default_file_contains_requested_topics(self):
        detector = ArticleTopicDetector(SemanticTestEmbedder(), DEFAULT_TOPICS_PATH)
        self.assertEqual(
            [topic.name for topic in detector.topics],
            ["Аниме", "Игры", "Технологии", "Кино", "Музыка", "Мода", "Юмор",
             "Знаменитости", "Книги", "Комиксы", "Соцсети", "Еда", "Искусство", "Животные"],
        )

    def test_invalid_configuration_and_vectors_are_rejected(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "topics.txt"
            for configuration in ("# No topics", "Аниме\nаниме", "| Missing name"):
                path.write_text(configuration, encoding="utf-8")
                with self.assertRaises(ValueError):
                    ArticleTopicDetector(SemanticTestEmbedder(), path)
            path.write_text("Аниме", encoding="utf-8")
            detector = ArticleTopicDetector(SemanticTestEmbedder(), path)
            for vector in ([0.0] * EMBEDDING_DIMENSIONS, [1.0], [float("nan")] * EMBEDDING_DIMENSIONS):
                with self.assertRaises(ValueError):
                    detector.detect_from_embedding(vector)
