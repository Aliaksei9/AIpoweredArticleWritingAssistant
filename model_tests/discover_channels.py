"""Discover public related channels and verify visible reactions/comments via Telegram.

Read-only: never joins a channel or sends messages. Reports contain counts and
public identities, not credentials. Screening can be resumed from its JSON file.
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
from telethon.errors import FloodWaitError
from telethon.tl.functions.channels import GetChannelRecommendationsRequest

from .select_channels import inspect_channel


EXTRA_SEEDS = {
    "Игры": ["dtfbest", "igromania", "vgtimes", "kanobu", "cybersportru", "gamemagru", "gameguru", "playground_ru", "shazoo"],
    "Технологии": ["4pda", "exploitex", "durov_code", "iguides_ru", "trashbox_feed", "habr_com", "tproger", "iphonesru", "androidinsider_ru", "appleinsider_ru", "mobile_review_com", "ferra_ru"],
    "Аниме": ["AniNewsTV", "anim_one", "top_anime_news", "ccoa_anime", "anifoxis", "animixlib"],
    "Кино": ["movierls", "kinoogon_official", "filmpro", "kinotv", "cinemaholics", "kinometro", "redheadredemption"],
    "Юмор": ["memepedia", "leprum", "rhyme"],
    "Знаменитости": ["SuperRu", "peopletalkru", "spletnikru"],
    "Книги": ["books", "bookandbed", "bookmatejournal", "eksmo", "ast_publishers"],
    "Еда": ["foodru", "eda_ru"],
    "Мода": ["grazia_ru", "elle_ru"],
    "Искусство": ["artchive", "arzamaslive"],
}


class CachedEntityClient:
    """Use Telethon's entity cache before resolving a public username again."""
    def __init__(self, client):
        self.client = client

    async def get_entity(self, address):
        return await self.client.get_entity(await self.client.get_input_entity(address))

    async def __call__(self, request):
        return await self.client(request)

    def iter_messages(self, *args, **kwargs):
        return self.client.iter_messages(*args, **kwargs)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


async def main():
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=40)
    parser.add_argument("--max-candidates", type=int, default=240)
    parser.add_argument("--recommend-seeds", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.sample_size < 10 or args.max_candidates < 1:
        parser.error("Need at least 10 sample posts and a positive candidate limit")
    args.output.mkdir(parents=True, exist_ok=True)
    report_file = args.output / "channel_screening.json"
    if report_file.exists() and not args.resume:
        parser.error("Report already exists; use --resume or another output")
    candidates = {}
    for entry in json.loads((root / "model_tests/channel_candidates.json").read_text(encoding="utf-8")):
        candidates[entry["username"].lower()] = dict(entry, discovered_via="existing_candidates")
    for line in (root / "src/database/channels.txt").read_text(encoding="utf-8").splitlines():
        address = line.partition("#")[0].strip().lstrip("@")
        if address:
            candidates.setdefault(address.lower(), dict(username=address, topic="Существующий канал", discovered_via="existing_channels"))
    for topic, names in EXTRA_SEEDS.items():
        for name in names:
            candidates.setdefault(name.lower(), dict(username=name, topic=topic, discovered_via="public_seed"))
    report = (json.loads(report_file.read_text(encoding="utf-8")) if args.resume and report_file.exists()
              else dict(started_at=datetime.now(timezone.utc).isoformat(), status="running", channels=[], recommendations=[]))
    for entry in report.get("candidates", []):
        candidates.setdefault(entry["username"].lower(), entry)
    client = TelegramClient(os.getenv("TG_SESSION", ".telegram_session"), int(os.environ["TG_API_ID"]),
                            os.environ["TG_API_HASH"], flood_sleep_threshold=60)
    await asyncio.wait_for(client.connect(), 30)
    try:
        if not await client.is_user_authorized():
            raise RuntimeError("Authorize Telegram locally using login-telegram.cmd")
        done_seeds = {r["seed"].lower() for r in report["recommendations"]}
        seeds = list(candidates.values())[:args.recommend_seeds]
        for seed in seeds:
            if seed["username"].lower() in done_seeds:
                continue
            try:
                entity = await client.get_input_entity("@" + seed["username"])
                result = await asyncio.wait_for(client(GetChannelRecommendationsRequest(channel=entity)), 30)
                added = 0
                for channel in result.chats:
                    username = getattr(channel, "username", None)
                    if not username or not getattr(channel, "broadcast", False):
                        continue
                    if username.lower() not in candidates:
                        candidates[username.lower()] = dict(username=username, topic=seed["topic"],
                            discovered_via="recommendation:" + seed["username"])
                        added += 1
                entry = dict(seed=seed["username"], added=added, returned=len(result.chats), status="complete")
            except FloodWaitError:
                raise
            except Exception as error:
                entry = dict(seed=seed["username"], status="failed", error_type=type(error).__name__)
            report["recommendations"].append(entry)
            report["candidates"] = list(candidates.values())
            write_json(report_file, report)
            print(json.dumps(dict(stage="recommendations", **entry), ensure_ascii=False), flush=True)
        as_of = datetime.now(timezone.utc)
        report.update(status="screening", as_of=as_of.isoformat(), sample_size=args.sample_size)
        completed = {r["username"].lower() for r in report["channels"]}
        for index, candidate in enumerate(list(candidates.values())[:args.max_candidates], 1):
            if candidate["username"].lower() in completed:
                continue
            try:
                record = await asyncio.wait_for(inspect_channel(CachedEntityClient(client), candidate, as_of, args.sample_size), 90)
                n = record.get("sample", {}).get("text_posts", 0)
                opened = record.get("sample", {}).get("comments_enabled_posts", 0)
                if n and opened / n < .9:
                    record["reasons"].append("Fewer than 90% of sampled posts have open comments")
                valid = sum(s["views"] is not None and s["views"] > 0 and s["reactions"] is not None
                            and s["comments"] is not None for s in record.get("samples", []))
                if n and valid / n < .9:
                    record["reasons"].append("Too few positive-view posts with complete metrics")
                record["qualifies"] = not record["reasons"]
                dates = [datetime.fromisoformat(s["date"]) for s in record.get("samples", [])]
                span = (max(dates) - min(dates)).total_seconds() / 86400 if len(dates) > 1 else 12
                record["estimated_window_posts"] = round(min(1000, n * 12 / max(span, .25)))
                record["discovered_via"] = candidate["discovered_via"]
            except FloodWaitError:
                raise
            except Exception as error:
                record = dict(username=candidate["username"], topic=candidate["topic"], qualifies=False,
                              reasons=[type(error).__name__], discovered_via=candidate["discovered_via"])
            report["channels"].append(record)
            write_json(report_file, report)
            qualified = sorted((r for r in report["channels"] if r["qualifies"]),
                               key=lambda r: -r.get("estimated_window_posts", 0))
            (args.output / "qualified_channels.txt").write_text("\n".join("@"+r["username"] for r in qualified)+"\n", encoding="utf-8")
            print(json.dumps(dict(stage="screening", index=index, total=min(args.max_candidates, len(candidates)),
                username=record["username"], qualifies=record["qualifies"],
                estimated_posts=record.get("estimated_window_posts"), qualified_channels=len(qualified),
                estimated_total=sum(r.get("estimated_window_posts", 0) for r in qualified), reasons=record["reasons"]), ensure_ascii=False), flush=True)
            await asyncio.sleep(.4)
        report.update(status="complete", finished_at=datetime.now(timezone.utc).isoformat())
    except FloodWaitError as error:
        report.update(status="rate_limited", retry_after_seconds=error.seconds)
        print(f"Telegram rate limit: retry after {error.seconds} seconds; checkpoint saved", flush=True)
        raise
    finally:
        await client.disconnect()
        report["candidates"] = list(candidates.values())
        write_json(report_file, report)


if __name__ == "__main__":
    with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop if sys.platform == "win32" else None) as runner:
        runner.run(main())
