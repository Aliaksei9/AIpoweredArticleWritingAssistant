"""Collect verified public channels until a minimum of distinct training texts.

All database writes are performed by TGdataCollection. The client wrapper only
filters messages with closed discussions; it never joins or posts to channels.
"""
import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

from dotenv import load_dotenv
from telethon import TelegramClient

from src.database.bd_collection import BDdataCollection
from src.database.connection import connect_database
from src.database.model_embedding import TextEmbedder
from src.database.resources.telegram.collector import TGdataCollection, reaction_count
from src.assesment_conform_of_PP_text_based.dataset import text_group

from .discover_channels import CachedEntityClient, write_json


class OpenDiscussionClient(CachedEntityClient):
    def __init__(self, client, oldest):
        super().__init__(client)
        self.oldest = oldest
        self.skipped_closed = 0

    async def iter_messages(self, *args, **kwargs):
        async for message in self.client.iter_messages(*args, **kwargs):
            date = getattr(message, "date", None)
            if date is not None and date <= self.oldest:
                yield message
                return
            if not getattr(getattr(message, "replies", None), "comments", False) or reaction_count(message) is None:
                self.skipped_closed += 1
                continue
            yield message


async def training_counts(conn):
    cursor = await conn.execute("""SELECT t.text, t.topic, s.resource_id
        FROM article_text t JOIN article_metrics m ON m.article_id=t.id
        JOIN article_sources s ON s.article_id=t.id JOIN article_vectors v ON v.article_id=t.id
        WHERE m.popularity_pattern_score IS NOT NULL AND m.views>0 AND m.audience>0
          AND m.reactions IS NOT NULL AND m.comments IS NOT NULL""")
    rows = await cursor.fetchall()
    return dict(training_rows=len(rows), distinct_texts=len({text_group(r["text"]) for r in rows}),
                resources=len({str(r["resource_id"]) for r in rows}),
                topics={topic: sum(r["topic"]==topic for r in rows) for topic in sorted({r["topic"] for r in rows})})


async def main():
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screening", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=int, default=10001)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.target < 1 or args.limit < 1:
        parser.error("Target and limit must be positive")
    screening = json.loads(args.screening.read_text(encoding="utf-8"))
    qualified = {r["username"].lower(): r for r in screening["channels"] if r["qualifies"]}
    originals = [line.partition("#")[0].strip().lstrip("@").lower()
                 for line in (root / "src/database/channels.txt").read_text(encoding="utf-8").splitlines()
                 if line.partition("#")[0].strip()]
    names = [n for n in originals if n in qualified]
    names += sorted((n for n in qualified if n not in names), key=lambda n: -qualified[n].get("estimated_window_posts",0))
    if args.output.exists() and not args.resume:
        parser.error("Report exists; use --resume or another file")
    report = (json.loads(args.output.read_text(encoding="utf-8")) if args.resume and args.output.exists()
              else dict(started_at=datetime.now(timezone.utc).isoformat(), status="running", resources=[]))
    as_of = datetime.fromisoformat(report.get("as_of", datetime.now(timezone.utc).isoformat()))
    report.update(as_of=as_of.isoformat(), target_distinct_texts=args.target, limit_per_resource=args.limit,
                  age_days=dict(min_exclusive=2,max_exclusive=14), open_discussions_only=True)
    completed = {r["username"].lower() for r in report["resources"] if r["status"]=="complete"}
    client = TelegramClient(os.getenv("TG_SESSION", ".telegram_session"), int(os.environ["TG_API_ID"]),
                            os.environ["TG_API_HASH"], flood_sleep_threshold=60)
    await client.connect()
    try:
        if not await client.is_user_authorized():
            raise RuntimeError("Authorize Telegram locally first")
        import torch
        torch.set_num_threads(4)
        print("Loading local E5 for collection",flush=True)
        embedder = await asyncio.to_thread(TextEmbedder)
        oldest, _ = BDdataCollection.publication_window(as_of)
        filtered = OpenDiscussionClient(client, oldest)
        async with await connect_database(os.environ["DATABASE_URL"]) as conn:
            collector = TGdataCollection(conn, filtered, embedder, as_of=as_of)
            for index, name in enumerate(names,1):
                if name in completed:
                    continue
                collector.on_progress = lambda count, n=name: print(f"@{n}: processed {count}", flush=True)
                print(f"Collecting @{name} ({index}/{len(names)}), limit={args.limit}",flush=True)
                try:
                    count = await collector.update("@"+name,args.limit)
                    record = dict(username=name,status="complete",**collector.last_result.to_dict())
                except Exception as error:
                    record = dict(username=name,status="failed",error_type=type(error).__name__,
                                  **collector.last_result.to_dict())
                    if hasattr(error,"seconds"):
                        report["resources"].append(record)
                        report.update(status="rate_limited",retry_after_seconds=error.seconds)
                        write_json(args.output,report)
                        raise
                report["resources"].append(record)
                report["dataset"] = await training_counts(conn)
                report["skipped_closed_messages"] = filtered.skipped_closed
                write_json(args.output,report)
                print(json.dumps(dict(username=name,status=record["status"],processed=record.get("processed",0),
                                      dataset=report["dataset"]),ensure_ascii=False),flush=True)
                if report["dataset"]["distinct_texts"]>=args.target:
                    break
            report.update(status="complete" if report.get("dataset",{}).get("distinct_texts",0)>=args.target else "needs_more_channels",
                          finished_at=datetime.now(timezone.utc).isoformat())
            write_json(args.output,report)
    finally:
        await client.disconnect()
    print(json.dumps(dict(status=report["status"],dataset=report.get("dataset")),ensure_ascii=False),flush=True)
    if report["status"]!="complete":
        raise SystemExit(2)


if __name__ == "__main__":
    with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop if sys.platform == "win32" else None) as runner:
        runner.run(main())
