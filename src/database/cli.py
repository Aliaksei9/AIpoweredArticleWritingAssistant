"""CLI commands delegate database work and source lifecycle to separate services."""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from .bd_collection import DEFAULT_LIMIT, MAX_ARTICLE_AGE_DAYS, CollectionOptions
from .collection_service import CollectionService
from .connection import clear_database, initialize_database
from .embeddings import reembed_database
from .resources import SourceRegistry, build_registry
from .resources.base import require_env


async def initialize(args, registry):
    assessed = await initialize_database(require_env("DATABASE_URL"))
    print(f"Database schema is ready; assessed {assessed} articles")


async def reset(args, registry):
    dsn = require_env("DATABASE_URL")
    await initialize_database(dsn, recalculate_scores=False)
    counts = await clear_database(dsn)
    print("Cleared application tables: "+json.dumps(counts))


async def reembed(args, registry):
    dsn = require_env("DATABASE_URL")
    await initialize_database(dsn, recalculate_scores=False)
    result = await reembed_database(dsn)
    print("Recomputed database embeddings: " + json.dumps(result))


async def login(args, registry):
    errors = []
    adapters = registry.select(args.source)
    for adapter in adapters:
        try:
            await adapter.login()
        except Exception as error:
            errors.append((adapter.code, error))
            print(f"{adapter.code}: login failed ({type(error).__name__}: {error})", flush=True)
    if errors:
        if len(adapters) == 1:
            raise errors[0][1]
        raise ValueError("Login failed for sources: "+", ".join(code for code, _ in errors))


async def update(args, registry):
    options = CollectionOptions(limit=args.limit, max_age_days=args.max_age_days)
    await CollectionService().update_sources(require_env("DATABASE_URL"), registry.select(args.source),
        args.resources, options, args.report, resources_file=args.resources_file)


def parse_args(argv=None, *, registry: SourceRegistry | None = None) -> argparse.Namespace:
    registry = registry if registry is not None else build_registry()
    parser = argparse.ArgumentParser(description="Initialize or fill the PostgreSQL database")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init-db", help="Create missing tables and pgvector extension").set_defaults(handler=initialize)
    commands.add_parser("reset-db", help="Clear the five application tables; preserve the database").set_defaults(handler=reset)
    commands.add_parser("reembed", help="Rebuild stored vectors with the configured embedding model").set_defaults(handler=reembed)
    command = commands.add_parser("login", help="Authorize the selected source or all registered sources")
    command.set_defaults(handler=login)
    command.add_argument("--source", type=str.upper, choices=registry.codes,
                         help="Article source (default: all registered sources)")
    command = commands.add_parser("update", help="Add new articles and update existing ones")
    command.set_defaults(handler=update)
    command.add_argument("resources", nargs="*", help="Resource addresses, normalized separately by each source")
    command.add_argument("--source", type=str.upper, choices=registry.codes,
                         help="Article source (default: all registered sources)")
    command.add_argument("--resources-file", type=Path,
                         help="UTF-8 resource list; one address per line")
    command.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                         help="Eligible text posts per resource (default: %(default)s)")
    command.add_argument("--max-age-days", type=float, default=MAX_ARTICLE_AGE_DAYS,
                         help="Exclusive maximum age in days; older stored posts are deleted (default: %(default)s)")
    command.add_argument("--report", type=Path, help="Save an update report as JSON")
    return parser.parse_args(argv)


async def run_cli(argv=None, *, registry: SourceRegistry | None = None) -> None:
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    registry = registry if registry is not None else build_registry()
    args = parse_args(argv, registry=registry)
    try:
        await args.handler(args, registry)
    except (ValueError, OSError) as error:
        raise SystemExit(str(error)) from None
