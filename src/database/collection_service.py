"""Source-independent collection orchestration and JSON reporting."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from .bd_collection import BDdataCollection, CollectionOptions, MIN_ARTICLE_AGE_DAYS
from .connection import connect_database
from .sources import CollectionContext, SourceAdapter


def read_resources(
    adapter: SourceAdapter, resources: Iterable[str], resources_file: Path | None = None
) -> list[str]:
    addresses = list(resources)
    if resources_file is not None:
        addresses.extend(line.partition("#")[0].strip()
                         for line in resources_file.read_text(encoding="utf-8-sig").splitlines()
                         if line.partition("#")[0].strip())
    normalized = list(dict.fromkeys(adapter.normalize_resource(address) for address in addresses))
    if not normalized:
        raise ValueError("Specify resources or --resources-file")
    return normalized


async def load_embedder():
    from .model_embedding import MODEL_NAME, TextEmbedder
    import torch
    torch.set_num_threads(4)
    return await asyncio.to_thread(TextEmbedder, os.getenv("EMBEDDING_MODEL", MODEL_NAME))


def print_progress(message: str) -> None:
    print(message, flush=True)


def save_report(report: dict, path: Path | None) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


class CollectionService:
    def __init__(self, *, connect=None, embedder_loader=None, log=print_progress):
        self.connect = connect or connect_database
        self.load_embedder = embedder_loader or load_embedder
        self.log = log

    async def _update_resource(
        self, collector: BDdataCollection, address: str, options: CollectionOptions, report: dict
    ) -> None:
        self.log(f"{address}: collecting up to {options.limit} text posts, "
                 f"age {MIN_ARTICLE_AGE_DAYS:g}..{options.max_age_days:g} days")
        collector.on_progress = lambda count: self.log(f"{address}: processed {count}")
        status, error = "complete", None
        try:
            await collector.update(address, options.limit)
        except BaseException as exc:
            status = "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed"
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            result = collector.last_result
            record = dict(result.to_dict(), source=report["source"], address=address, status=status)
            if error:
                record["error"] = error
            report["resources"].append(record)
            report["deleted_expired"] += result.deleted_expired
            if result.deleted_expired:
                self.log(f"Removed {result.deleted_expired} expired articles from the database")
        self.log(f"{address}: processed {result.processed}; added {result.created}; updated {result.updated}; "
                 f"skipped by age {result.skipped_by_age}; skipped without date {result.skipped_without_date}")

    async def _update_source(
        self, adapter: SourceAdapter, context: CollectionContext, resources: list[str], report: dict
    ) -> None:
        async with adapter.open(context) as collector:
            for address in resources:
                await self._update_resource(collector, address, context.options, report)

    async def update_sources(
        self, dsn: str, adapters: Iterable[SourceAdapter], resources: Iterable[str], options: CollectionOptions,
        report_path: Path | None = None, *, resources_file: Path | None = None
    ) -> dict:
        """Share one model, isolate source failures, and always save one report format."""
        adapters = tuple(adapters)
        resources = list(resources)
        report = dict(
            started_at=datetime.now(timezone.utc).isoformat(), status="running", command="update",
            selected_sources=[adapter.code for adapter in adapters], sources=[], resources=[],
            limit_per_resource=options.limit, min_age_days=MIN_ARTICLE_AGE_DAYS,
            max_age_days=options.max_age_days,
            deleted_expired=0
        )
        errors, embedder = [], None
        try:
            if not adapters:
                raise ValueError("No article sources are selected")
            for adapter in adapters:
                source_report = dict(
                    source=adapter.code, started_at=datetime.now(timezone.utc).isoformat(),
                    status="running", deleted_expired=0, resources=[]
                )
                report["sources"].append(source_report)
                self.log(f"{adapter.code}: starting collection")
                try:
                    addresses = read_resources(adapter, resources, resources_file)
                    adapter.validate_configuration()
                    async with await self.connect(dsn) as conn:
                        if embedder is None:
                            embedder = await self.load_embedder()
                        context = CollectionContext(conn, embedder, options)
                        await self._update_source(adapter, context, addresses, source_report)
                    source_report["status"] = "complete"
                except BaseException as error:
                    source_report.update(
                        status="cancelled" if isinstance(error, asyncio.CancelledError) else "failed",
                        error=f"{type(error).__name__}: {error}"
                    )
                    if not isinstance(error, Exception):
                        raise
                    errors.append((adapter.code, error))
                    self.log(f"{adapter.code}: collection failed ({type(error).__name__}: {error})")
                finally:
                    source_report["finished_at"] = datetime.now(timezone.utc).isoformat()
            if errors:
                if len(adapters) == 1:
                    raise errors[0][1]
                raise ValueError("Collection failed for sources: " + ", ".join(code for code, _ in errors))
            report["status"] = "complete"
        except BaseException as error:
            report.update(
                status="cancelled" if isinstance(error, asyncio.CancelledError) else "failed",
                error=f"{type(error).__name__}: {error}"
            )
            raise
        finally:
            report["finished_at"] = datetime.now(timezone.utc).isoformat()
            report["deleted_expired"] = sum(source["deleted_expired"] for source in report["sources"])
            report["resources"] = [item for source in report["sources"] for item in source["resources"]]
            save_report(report, report_path)
        return report
