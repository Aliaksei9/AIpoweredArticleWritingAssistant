"""Search examples, preview prompts, or correct a text through OpenRouter."""

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
from dataclasses import replace

from dotenv import load_dotenv

from src.database.connection import connect_database
from src.database.model_embedding import TextEmbedder
from src.database.table_classes import ArticleVectorTable

from .correction import TextCorrectionModule
from .errors import TextCorrectionError
from .llm_analysis import LLManalysis
from .openrouter_client import OpenRouterClient, OpenRouterConfig
from .search_simillar import SearchSimillar


def write_json(path, result):
    content = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix+".tmp")
        temporary.write_text(content+"\n", encoding="utf-8")
        temporary.replace(path)
    else:
        print(content)


async def run():
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["search", "correct"])
    parser.add_argument("--text-file", type=Path, required=True)
    parser.add_argument("--topic", help="Autodetected using configured topics if omitted")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", help="OpenRouter model ID")
    parser.add_argument("--max-examples", type=int, default=int(os.getenv("TEXT_CORRECTION_MAX_EXAMPLES", "8")))
    parser.add_argument("--min-score", type=float, default=float(os.getenv("TEXT_CORRECTION_MIN_SCORE", "5")))
    parser.add_argument("--min-similarity", type=float, default=.6)
    parser.add_argument("--exclude-article-id")
    parser.add_argument("--prepare-only", action="store_true", help="Output prompts without making generation requests")
    args = parser.parse_args()
    if args.max_examples < 1:
        parser.error("--max-examples must be positive")
    if args.command == "search" and args.prepare_only:
        parser.error("--prepare-only applies to correct")
    config = OpenRouterConfig.from_env()
    if args.model:
        config = replace(config, model=args.model)
    if args.command == "correct" and not args.prepare_only and not config.api_key:
        raise TextCorrectionError("Set OPENROUTER_API_KEY in .env")
    text = args.text_file.read_text(encoding="utf-8-sig")
    if not text.strip():
        raise ValueError("Text file is empty")
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise TextCorrectionError("Set DATABASE_URL in .env and start PostgreSQL")
    import torch
    from threadpoolctl import threadpool_limits
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(4)
        with threadpool_limits(limits=4):
            async with await connect_database(dsn) as conn, OpenRouterClient(config) as client:
                embedder = await asyncio.to_thread(TextEmbedder)
                search = SearchSimillar(ArticleVectorTable(conn), embedder, min_score=args.min_score,
                                         min_similarity=args.min_similarity)
                if args.command == "search":
                    result = (await search.search(text, args.topic, limit=args.max_examples,
                              exclude_article_id=args.exclude_article_id)).to_dict()
                else:
                    analysis = LLManalysis(client, max_examples=args.max_examples)
                    module = TextCorrectionModule(search, analysis)
                    if args.prepare_only:
                        found, plan = await module.prepare(text, args.topic, exclude_article_id=args.exclude_article_id)
                        result = dict(plan.to_dict(), retrieval=found.to_dict())
                    else:
                        result = await module.correct(text, args.topic, exclude_article_id=args.exclude_article_id)
    finally:
        torch.set_num_threads(previous_threads)
    write_json(args.output, result)
    if args.output:
        print(json.dumps(dict(saved=str(args.output.resolve())), ensure_ascii=False))


if __name__ == "__main__":
    try:
        with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop if sys.platform == "win32" else None) as runner:
            runner.run(run())
    except (TextCorrectionError, ValueError, OSError) as error:
        print(json.dumps(dict(error=str(error)), ensure_ascii=False), file=sys.stderr)
        sys.exit(2)
