"""Check retention against real PostgreSQL without Telegram or a neural model.

Run explicitly: python -m model_tests.retention_check
Creates and removes an isolated temporary database; application data is untouched.
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import psycopg
from dotenv import load_dotenv
from psycopg import sql
from psycopg.conninfo import make_conninfo

from src.database.connection import connect_database, initialize_database
from src.database.ids_converter import article_id_for, resource_id_for
from src.database.model_embedding import EMBEDDING_DIMENSIONS
from src.database.resources.telegram.collector import TGdataCollection


ROOT = Path(__file__).resolve().parents[1]
VECTOR = [1.0] + [0.0] * (EMBEDDING_DIMENSIONS - 1)


class FixtureChannel:
    id = 424242
    title = "Retention check channel"
    participants_count = 1000


class EmptyClient:
    async def get_entity(self, resource):
        return FixtureChannel()

    async def __call__(self, request):
        return SimpleNamespace(full_chat=SimpleNamespace(participants_count=1000))

    async def iter_messages(self, channel, limit=None, offset_date=None):
        for message in ():
            yield message


class UnusedModel:
    def embed_document(self, text):
        raise AssertionError("Empty Telegram history must not require embeddings")

    def detect_from_embedding(self, embedding):
        raise AssertionError("Empty Telegram history must not require classification")


async def run_checks(conn) -> list[str]:
    """Use only an isolated database initialized by main()."""
    as_of = datetime(2026, 10, 22, 12, tzinfo=timezone.utc)
    cutoff = as_of - timedelta(days=14)
    unused = UnusedModel()
    collector = TGdataCollection(
        conn, EmptyClient(), unused, topic_detector=unused, as_of=as_of
    )
    checks = []
    second_channel_id = FixtureChannel.id + 1
    for channel_id in (FixtureChannel.id, second_channel_id):
        await collector.resources.upsert(
            resource_id_for("TG", channel_id), f"Fixture {channel_id}", 1000
        )

    async def seed(channel_id, message_id, published_at):
        identifier = article_id_for("TG", channel_id, message_id)
        async with conn.transaction():
            await collector.texts.upsert(identifier, "Fixture", "Fixture text", published_at)
            await collector.sources.link(identifier, resource_id_for("TG", channel_id), "TG")
            await collector.metrics.upsert(identifier, 100, 10, 1, 1000)
            await collector.vectors.upsert(identifier, VECTOR)
        return identifier

    async def assert_present(identifier, expected):
        for table in (collector.texts, collector.sources, collector.metrics, collector.vectors):
            assert ((await table.get(identifier)) is not None) == expected, (
                type(table).__name__, identifier, expected
            )

    expired = await seed(FixtureChannel.id, 1, cutoff - timedelta(days=1))
    exact = await seed(second_channel_id, 2, cutoff)
    fresh = await seed(second_channel_id, 3, cutoff + timedelta(microseconds=1))
    undated = await seed(FixtureChannel.id, 4, None)

    with patch("src.database.resources.telegram.collector.Channel", FixtureChannel), patch(
        "src.database.resources.telegram.collector.GetFullChannelRequest", lambda channel: channel
    ):
        assert await collector.update("fixture") == 0
        assert collector.last_result.deleted_expired == 2
        await assert_present(expired, False)
        await assert_present(exact, False)
        checks.append("empty_history_prunes_expired_and_exact_boundary_globally")
        checks.append("article_delete_cascades_to_sources_metrics_and_vectors")

        await assert_present(fresh, True)
        await assert_present(undated, True)
        checks.append("just_newer_boundary_and_unknown_dates_are_retained")
        for channel_id in (FixtureChannel.id, second_channel_id):
            assert await collector.resources.get(resource_id_for("TG", channel_id)) is not None
        checks.append("channel_records_are_preserved")

        # A second channel update at the same cutoff must not run the global
        # cleanup again. This row is intentionally inserted after the first purge.
        late_insert = await seed(second_channel_id, 5, cutoff - timedelta(days=1))
        assert await collector.update("fixture") == 0
        assert collector.last_result.deleted_expired == 0
        await assert_present(late_insert, True)
        checks.append("successful_cleanup_is_reused_for_the_same_cutoff")

        collector.as_of = as_of + timedelta(seconds=1)
        assert await collector.update("fixture") == 0
        assert collector.last_result.deleted_expired == 2
        await assert_present(late_insert, False)
        await assert_present(fresh, False)
        await assert_present(undated, True)
        checks.append("advancing_time_runs_cleanup_again_and_expires_previous_survivor")

    return checks


async def main():
    load_dotenv(ROOT / ".env")
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise SystemExit("Set DATABASE_URL in .env before running the retention check")
    temporary_name = "coursework_retention_check_" + uuid4().hex[:12]
    temporary_created = False
    async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as admin:
        try:
            await admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(temporary_name)))
            temporary_created = True
            test_dsn = make_conninfo(dsn, dbname=temporary_name)
            await initialize_database(test_dsn, recalculate_scores=False)
            async with await connect_database(test_dsn) as conn:
                checks = await run_checks(conn)
        finally:
            if temporary_created:
                await admin.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(temporary_name)))
    print(f"Retention check passed: {len(checks)} checks; temporary database removed", flush=True)
    for check in checks:
        print(f"  {check}", flush=True)


if __name__ == "__main__":
    loop_factory = asyncio.SelectorEventLoop if sys.platform == "win32" else None
    with asyncio.Runner(loop_factory=loop_factory) as runner:
        runner.run(main())
