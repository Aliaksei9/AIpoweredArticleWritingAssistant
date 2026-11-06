"""Check real PostgreSQL and E5, using deterministic Telegram API fixtures.

Run separately from offline tests: python -m model_tests.integration_check
A temporary database is created and removed; the configured database's
articles are not changed. Only the Telegram network transport is simulated.
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import psycopg
from dotenv import load_dotenv
from psycopg import sql
from psycopg.conninfo import make_conninfo
from psycopg.rows import dict_row

from src.database.connection import clear_database, connect_database, initialize_database
from src.database.ids_converter import article_id_for
from src.database.model_embedding import MODEL_NAME, TextEmbedder
from src.database.telegram import TGdataCollection


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "model_tests" / "results" / "database_integration_report.json"


class TestChannel:
    id = 424242
    title = "Integration test channel"
    participants_count = 1000


class FixtureClient:
    def __init__(self, messages):
        self.messages = messages

    async def get_entity(self, resource):
        return TestChannel()

    async def __call__(self, request):
        return SimpleNamespace(full_chat=SimpleNamespace(participants_count=1000))

    async def iter_messages(self, channel, limit=None, offset_date=None):
        for message in self.messages[:limit]:
            yield message


def message(message_id, text, views=500, reactions=50, comments=10):
    return SimpleNamespace(
        id=message_id,
        raw_text=text,
        date=datetime.now(timezone.utc) - timedelta(days=3),
        action=None,
        views=views,
        reactions=SimpleNamespace(results=[SimpleNamespace(count=reactions)]),
        replies=SimpleNamespace(replies=comments),
    )


async def run_checks(conn, embedder, report):
    samples = [
        message(101, "Индексы в PostgreSQL. Как ускорить SQL-запросы к базе данных? "
                "Разбираем B-tree индексы, планы EXPLAIN ANALYZE, оптимизацию запросов "
                "и работу программиста с системой управления базами данных."),
        message(102, "Рецепт домашних блинов. Смешайте молоко, яйца, муку, сахар и соль. "
                "Добавьте растительное масло, замесите тесто и обжарьте блины на сковороде. "
                "Подавайте готовое блюдо со сметаной и вареньем.", 1000, 100, 20),
        message(103, "Как ухаживать за домашней кошкой. Подбирайте корм для питомца, "
                "чистите лоток и вычёсывайте шерсть. Следите за поведением животного, "
                "делайте прививки и регулярно показывайте кошку ветеринару."),
    ]
    expected_topics = ["Технологии", "Еда", "Животные"]
    as_of = datetime.now(timezone.utc)
    excluded = [message(number, samples[0].raw_text) for number in range(201, 207)]
    for item, date in zip(excluded, (as_of - timedelta(days=1),
                                    as_of - timedelta(days=2), None,
                                    as_of + timedelta(days=1),
                                    as_of - timedelta(days=14), as_of - timedelta(days=15))):
        item.date = date
    client = FixtureClient(samples + excluded)
    collector = TGdataCollection(conn, client, embedder)
    checks = report["checks"]

    with patch("src.database.telegram.Channel", TestChannel), patch(
        "src.database.telegram.GetFullChannelRequest", lambda channel: channel
    ), patch("src.database.telegram.datetime") as clock:
        clock.now.return_value = as_of
        print("Collecting three fixtures with real model and database", flush=True)
        assert await collector.update("fixture") == 3
        result = collector.last_result
        assert result.processed == 3
        assert (result.created, result.updated, result.skipped_by_age,
                result.skipped_without_date) == (3, 0, 4, 1)
        for item in excluded:
            excluded_id = article_id_for("TG", TestChannel.id, item.id)
            for table in (collector.texts, collector.sources, collector.vectors, collector.metrics):
                assert await table.get(excluded_id) is None
        checks.append("age_filter_rejects_recent_exact_48h_exact_14days_expired_future_and_undated")
        cursor = await conn.execute(
            """SELECT t.id, t.topic, m.popularity_pattern_score AS score,
                      vector_dims(v.embedding) AS dimensions,
                      v.embedding::text AS vector_text
               FROM article_text t JOIN article_metrics m ON m.article_id = t.id
               JOIN article_sources s ON s.article_id = t.id
               JOIN resource r ON r.id = s.resource_id
               JOIN article_vectors v ON v.article_id = t.id"""
        )
        rows = {row["id"]: row for row in await cursor.fetchall()}
        assert len(rows) == 3
        report["samples"] = []
        for item, expected_topic in zip(samples, expected_topics):
            article_id = article_id_for("TG", TestChannel.id, item.id)
            row = rows[article_id]
            report["samples"].append(dict(
                article_id=str(article_id), expected_topic=expected_topic,
                topic=row["topic"], dimensions=row["dimensions"], score=row["score"],
            ))
            assert row["topic"] == expected_topic, (expected_topic, row["topic"])
            assert row["dimensions"] == 384
            assert -10 <= row["score"] <= 10
        checks.append("all_five_tables_and_real_topic_classification")

        print("Checking repeated collection and metric updates", flush=True)
        first_id = article_id_for("TG", TestChannel.id, 101)
        old_vector = rows[first_id]["vector_text"]
        assert await collector.update("fixture") == 3
        cursor = await conn.execute("SELECT count(*) AS count FROM article_text")
        assert (await cursor.fetchone())["count"] == 3
        checks.append("repeated_collection_without_duplicates")
        samples[0].views = 1000
        samples[0].reactions.results[0].count = 100
        samples[0].replies.replies = 20
        assert await collector.update("fixture") == 3
        metric = await collector.metrics.get(first_id)
        assert metric["views"] == 1000
        assert abs(metric["popularity_pattern_score"] - 9.807079) < 1e-6
        cursor = await conn.execute(
            "SELECT embedding::text AS vector_text FROM article_vectors WHERE article_id = %s",
            (first_id,),
        )
        assert (await cursor.fetchone())["vector_text"] == old_vector
        checks.append("metric_updates_preserve_text_embedding")

        before_scores = {}
        for item in samples:
            identifier = article_id_for("TG", TestChannel.id, item.id)
            saved = await collector.metrics.get(identifier)
            assert saved["audience"] == 1000 and saved["collected_at"] is not None
            assert (await collector.texts.get(identifier))["published_at"] is not None
            before_scores[identifier] = saved["popularity_pattern_score"]
        from src.database.ids_converter import resource_id_for
        await collector.resources.upsert(resource_id_for("TG", TestChannel.id), "Grown channel", 2000)
        await collector.metrics.assessor.assess_database(collector.metrics)
        for identifier, score in before_scores.items():
            assert (await collector.metrics.get(identifier))["popularity_pattern_score"] == score
        checks.append("scores_use_observation_audience_not_future_resource_audience")

        print("Checking edited text and missing vector recovery", flush=True)
        samples[0].raw_text = (
            "Основы музыкальной гармонии. Учимся строить мажорные и минорные аккорды, "
            "играть мелодию на гитаре и подбирать аккомпанемент к песне. "
            "Разбираем тональности, ритм и музыкальные интервалы."
        )
        assert await collector.update("fixture") == 3
        assert (await collector.texts.get(first_id))["topic"] == "Музыка"
        cursor = await conn.execute(
            "SELECT embedding::text AS vector_text FROM article_vectors WHERE article_id = %s",
            (first_id,),
        )
        assert (await cursor.fetchone())["vector_text"] != old_vector
        checks.append("edited_text_reembeds_and_changes_topic")
        food_id = article_id_for("TG", TestChannel.id, 102)
        await collector.vectors.delete(food_id)
        assert await collector.update("fixture") == 3
        assert await collector.vectors.get(food_id) is not None
        checks.append("missing_embedding_recovery")

        question = await asyncio.to_thread(
            embedder.embed_query, "Как приготовить блины из молока, яиц и муки?"
        )
        neighbors = await collector.vectors.search(question, limit=1)
        assert neighbors[0]["article_id"] == food_id
        checks.append("real_vector_similarity_search")

        print("Checking transaction rollback and score constraint", flush=True)
        failed_id = article_id_for("TG", TestChannel.id, 104)
        client.messages = [message(104, samples[1].raw_text)]
        with patch.object(collector.metrics, "upsert", AsyncMock(side_effect=RuntimeError("test failure"))):
            try:
                await collector.update("fixture")
            except RuntimeError:
                pass
            else:
                raise AssertionError("Injected write failure did not propagate")
        for table in (collector.texts, collector.sources, collector.vectors, collector.metrics):
            assert await table.get(failed_id) is None
        checks.append("failed_article_write_rolls_back_all_article_tables")
        try:
            async with conn.transaction():
                await conn.execute(
                    "UPDATE article_metrics SET popularity_pattern_score = 11 WHERE article_id = %s",
                    (first_id,),
                )
        except psycopg.errors.CheckViolation:
            checks.append("database_enforces_score_range")
        else:
            raise AssertionError("Database accepted a score outside [-10, 10]")

        await conn.execute("UPDATE article_metrics SET popularity_pattern_score = NULL")


async def main():
    load_dotenv(ROOT / ".env")
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise SystemExit("Set DATABASE_URL in .env before running the integration check")
    model_name = os.getenv("EMBEDDING_MODEL", MODEL_NAME)
    report = dict(
        started_at=datetime.now(timezone.utc).isoformat(), status="running",
        model=model_name, telegram_transport="fixtures", checks=[],
        versions={name: version(name) for name in ("psycopg", "pgvector", "telethon", "sentence-transformers")},
    )
    temporary_name = "coursework_check_" + uuid4().hex[:12]
    temporary_created = False
    try:
        async with await psycopg.AsyncConnection.connect(dsn, autocommit=True, row_factory=dict_row) as admin:
            cursor = await admin.execute("SELECT version() AS version")
            report["postgresql"] = (await cursor.fetchone())["version"]
            await admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(temporary_name)))
            temporary_created = True
            try:
                test_dsn = make_conninfo(dsn, dbname=temporary_name)
                print("Initializing isolated database", flush=True)
                assert await initialize_database(test_dsn) == 0
                print("Loading real E5 model (first run downloads the weights)", flush=True)
                embedder = await asyncio.to_thread(TextEmbedder, model_name)
                async with await connect_database(test_dsn) as conn:
                    cursor = await conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
                    report["pgvector_extension"] = (await cursor.fetchone())["extversion"]
                    await run_checks(conn, embedder, report)
                    assert await initialize_database(test_dsn) == 3
                    cursor = await conn.execute(
                        "SELECT count(*) AS count FROM article_metrics WHERE popularity_pattern_score IS NOT NULL"
                    )
                    assert (await cursor.fetchone())["count"] == 3
                    report["checks"].append("initialization_recalculates_existing_scores")
                    deleted = await clear_database(test_dsn)
                    assert deleted == {"resource": 1, "article_text": 3, "article_sources": 3,
                                       "article_metrics": 3, "article_vectors": 3}
                    for table in deleted:
                        cursor = await conn.execute(f"SELECT count(*) AS count FROM {table}")
                        assert (await cursor.fetchone())["count"] == 0
                    report["checks"].append("reset_clears_only_application_tables")
            finally:
                if temporary_created:
                    await admin.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(temporary_name)))
                    report["temporary_database_removed"] = True
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Integration report: {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    loop_factory = asyncio.SelectorEventLoop if sys.platform == "win32" else None
    with asyncio.Runner(loop_factory=loop_factory) as runner:
        runner.run(main())
