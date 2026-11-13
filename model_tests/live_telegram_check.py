"""Populate the configured database from Telegram and verify the saved rows.

Run explicitly: python -m model_tests.live_telegram_check '@channel' --limit 1000
This check saves real posts in the configured database and requires a local
authorized Telegram session. It never requests login codes in the test run.
"""

import argparse
import asyncio
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from telethon import TelegramClient
from telethon.tl.types import Channel

from src.database.connection import connect_database, initialize_database
from src.database.ids_converter import resource_id_for
from src.database.model_embedding import EMBEDDING_DIMENSIONS, MODEL_NAME, TextEmbedder
from src.database.resources.telegram.collector import TGdataCollection


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "model_tests" / "results" / "telegram_database_report.json"


async def read_articles(collector, resource_id, score_ids):
    resource = await collector.resources.get(resource_id)
    assert resource is not None
    topics = {topic.name for topic in collector.topic_detector.topics}
    rows = {}
    for source in await collector.sources.list_by_resource(resource_id):
        article_id = source["article_id"]
        text = await collector.texts.get(article_id)
        metrics = await collector.metrics.get(article_id)
        vector = await collector.vectors.get(article_id)
        assert source["source"] == "TG"
        assert text is not None and text["text"].strip() and text["topic"] in topics
        assert metrics is not None and vector is not None
        embedding = [float(value) for value in vector["embedding"]]
        assert len(embedding) == EMBEDDING_DIMENSIONS
        assert all(math.isfinite(value) for value in embedding)
        assert abs(math.hypot(*embedding) - 1.0) < 1e-5
        if article_id in score_ids:
            expected_score = collector.metrics.assessor.assess(
                metrics["views"], metrics["reactions"], metrics["comments"], metrics["audience"]
            )
            assert metrics["popularity_pattern_score"] == expected_score
        rows[article_id] = {"text": text, "metrics": metrics, "embedding": embedding}
    return resource, rows


async def sync_and_check(collector, resource_id, resource, limit):
    cursor = await collector.conn.execute("SELECT pg_current_xact_id()::text AS transaction_id")
    revision = int((await cursor.fetchone())["transaction_id"])
    count = await collector.update(resource, limit)
    cursor = await collector.conn.execute(
        """SELECT m.article_id FROM article_metrics m
           JOIN article_sources s ON s.article_id = m.article_id
           WHERE s.resource_id = %s AND m.xmin::text::bigint > %s""",
        (resource_id, revision),
    )
    score_ids = {row["article_id"] for row in await cursor.fetchall()}
    resource_row, articles = await read_articles(collector, resource_id, score_ids)
    return count, resource_row, articles, score_ids


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("resource")
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")
    load_dotenv(ROOT / ".env")
    for name in ("DATABASE_URL", "TG_API_ID", "TG_API_HASH"):
        if not os.getenv(name):
            raise SystemExit(f"Set {name} in .env")
    report = {"started_at": datetime.now(timezone.utc).isoformat(),
              "status": "running", "channel": args.resource, "limit": args.limit,
              "telegram_transport": "live_api", "checks": []}
    client = TelegramClient(os.getenv("TG_SESSION", ".telegram_session"),
                            int(os.environ["TG_API_ID"]), os.environ["TG_API_HASH"])
    try:
        await client.connect()
        if not await client.is_user_authorized():
            raise RuntimeError("Run python -m src.database login --source TG in a local terminal first")
        channel = await client.get_entity(args.resource)
        if not isinstance(channel, Channel):
            raise RuntimeError("Resource must be a Telegram channel")
        resource_id = resource_id_for("TG", channel.id)
        await initialize_database(os.environ["DATABASE_URL"])
        print("Loading real E5 model", flush=True)
        embedder = await asyncio.to_thread(TextEmbedder)
        async with await connect_database(os.environ["DATABASE_URL"]) as conn:
            collector = TGdataCollection(conn, client, embedder)
            print(f"Collecting {args.resource}: up to {args.limit} text posts aged 2..14 days", flush=True)
            report["processed_first_run"], resource, first, first_score_ids = await sync_and_check(
                collector, resource_id, args.resource, args.limit
            )
            report["first_run_stats"] = collector.last_result.to_dict()
            report["articles_after_first_run"] = len(first)
            report["checks"].append("stored_posts_have_all_five_table_records")
            print("Repeating collection through update", flush=True)
            report["processed_second_run"], resource, second, second_score_ids = await sync_and_check(
                collector, resource_id, args.resource, args.limit
            )
            report["second_run_stats"] = collector.last_result.to_dict()
            assert first.keys() <= second.keys()
            unchanged = 0
            for article_id, old in first.items():
                if old["text"]["text"] == second[article_id]["text"]["text"]:
                    assert old["embedding"] == second[article_id]["embedding"]
                    unchanged += 1
            report.update(title=resource["name"], audience=resource["reader_count"],
                          articles_after_second_run=len(second),
                          new_articles_on_second_run=len(second.keys() - first.keys()),
                          unchanged_embeddings=unchanged)
            report["checks"].extend(["stable_article_ids_on_update", "saved_vectors_reused",
                                    "normalized_384_dimension_vectors", "configured_topics_saved"])
            report["scores_verified"] = len(first_score_ids | second_score_ids)
            if report["scores_verified"]:
                report["checks"].append("updated_metric_scores_match_formula")
            if report["processed_first_run"] == report["processed_second_run"] == 0:
                report["checks"].append("no_eligible_posts_in_requested_window")
            report["articles"] = [dict(article_id=str(article_id), topic=row["text"]["topic"],
                                       views=row["metrics"]["views"], reactions=row["metrics"]["reactions"],
                                       comments=row["metrics"]["comments"],
                                       score=row["metrics"]["popularity_pattern_score"],
                                       dimensions=len(row["embedding"]))
                                  for article_id, row in second.items()]
        report["status"] = "passed"
        print(
            f"Database contains {len(second)} articles; "
            f"first update processed {report['processed_first_run']}; "
            f"second update processed {report['processed_second_run']}; "
            f"update added {report['new_articles_on_second_run']}", flush=True
        )
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        await client.disconnect()
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Live collection report: {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    loop_factory = asyncio.SelectorEventLoop if sys.platform == "win32" else None
    with asyncio.Runner(loop_factory=loop_factory) as runner:
        runner.run(main())
