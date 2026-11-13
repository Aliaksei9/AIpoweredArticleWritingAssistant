"""Read-only Telegram screening for channels with usable social metrics."""

import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from telethon import TelegramClient
from telethon.tl.functions.channels import GetFullChannelRequest
from telethon.tl.types import Channel, ChatReactionsNone

from src.database.bd_collection import BDdataCollection
from src.database.resources.telegram.collector import reaction_count


async def inspect_channel(client, candidate, as_of, sample_size):
    username = candidate["username"].lstrip("@")
    channel = await client.get_entity("@" + username)
    if not isinstance(channel, Channel) or not getattr(channel, "broadcast", False):
        raise ValueError("Not a broadcast channel")
    full = (await client(GetFullChannelRequest(channel))).full_chat
    oldest, newest = BDdataCollection.publication_window(as_of)
    counts = Counter()
    samples = []
    async for message in client.iter_messages(channel, limit=None, offset_date=newest):
        published = getattr(message, "date", None)
        if published is not None and published <= oldest:
            break
        counts["scanned_messages"] += 1
        if counts["scanned_messages"] > 2000:
            break
        text = (getattr(message, "raw_text", None) or "").strip()
        if not text or getattr(message, "action", None) is not None:
            continue
        if not BDdataCollection.is_article_in_window(published, as_of):
            continue
        views = getattr(message, "views", None)
        reactions = reaction_count(message)
        replies = getattr(message, "replies", None)
        comments = getattr(replies, "replies", None)
        counts["text_posts"] += 1
        counts["known_views"] += views is not None
        counts["known_reactions"] += reactions is not None
        counts["known_comments"] += comments is not None
        counts["comments_enabled_posts"] += bool(getattr(replies, "comments", False))
        counts["positive_reactions_posts"] += reactions is not None and reactions > 0
        counts["positive_comments_posts"] += comments is not None and comments > 0
        counts["complete_metrics_posts"] += all(value is not None for value in (views, reactions, comments))
        samples.append(dict(id=message.id, date=published.isoformat(), views=views,
                            reactions=reactions, comments=comments))
        if counts["text_posts"] >= sample_size:
            break
    count = counts["text_posts"]
    available_reactions = getattr(full, "available_reactions", None)
    audience = getattr(full, "participants_count", None)
    complete_ratio = counts["complete_metrics_posts"] / count if count else 0
    reasons = []
    if not audience:
        reasons.append("Audience unavailable")
    if not getattr(full, "linked_chat_id", None) or not counts["comments_enabled_posts"]:
        reasons.append("No available channel comments")
    if isinstance(available_reactions, ChatReactionsNone) or not counts["known_reactions"]:
        reasons.append("Reactions unavailable")
    if count < 10:
        reasons.append("Fewer than 10 text posts in the window")
    if complete_ratio < .9:
        reasons.append("Fewer than 90% of sampled text posts have complete metrics")
    if not counts["positive_comments_posts"] or not counts["positive_reactions_posts"]:
        reasons.append("No evidence of actual comment/reaction activity")
    return dict(username=username, canonical_username=getattr(channel, "username", username),
                channel_id=channel.id, title=channel.title, topic=candidate["topic"],
                url="https://t.me/" + (channel.username or username), audience=audience,
                reactions_type=type(available_reactions).__name__,
                linked_discussion_available=bool(getattr(full, "linked_chat_id", None)),
                sample=dict(counts), complete_metrics_ratio=complete_ratio,
                qualifies=not reasons, reasons=reasons, samples=samples)


async def main():
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=root / "model_tests/channel_candidates.json")
    parser.add_argument("--output", type=Path, default=root / "model_tests/results/channel_screening.json")
    parser.add_argument("--sample-size", type=int, default=100)
    args = parser.parse_args()
    if args.sample_size < 10:
        parser.error("--sample-size must be at least 10")
    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))
    as_of = datetime.now(timezone.utc)
    report = dict(status="running", as_of=as_of.isoformat(), min_age_days=2, max_age_days=14,
                  sample_size=args.sample_size, channels=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    client = TelegramClient(os.getenv("TG_SESSION", ".telegram_session"),
                            int(os.environ["TG_API_ID"]), os.environ["TG_API_HASH"])
    await client.connect()
    try:
        if not await client.is_user_authorized():
            raise RuntimeError("Authorize Telegram locally with login-telegram.cmd")
        for index, candidate in enumerate(candidates, start=1):
            try:
                record = await inspect_channel(client, candidate, as_of, args.sample_size)
            except Exception as exc:
                record = dict(username=candidate["username"], topic=candidate["topic"],
                              qualifies=False, reasons=[f"{type(exc).__name__}: {exc}"])
            report["channels"].append(record)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(dict(index=index, total=len(candidates), username=record["username"],
                                  qualifies=record["qualifies"], audience=record.get("audience"),
                                  text_posts=record.get("sample", {}).get("text_posts", 0),
                                  complete_ratio=record.get("complete_metrics_ratio"), reasons=record["reasons"]),
                             ensure_ascii=True), flush=True)
        report.update(status="complete", finished_at=datetime.now(timezone.utc).isoformat())
    finally:
        await client.disconnect()
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop if sys.platform == "win32" else None) as runner:
        runner.run(main())
