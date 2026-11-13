"""Rebuild stored vectors with the application's configured embedding model."""

import asyncio

from .collection_service import load_embedder, print_progress
from .connection import connect_database
from .table_classes import ArticleTextTable, ArticleVectorTable


async def reembed_database(dsn: str, *, log=print_progress) -> dict[str, int]:
    """Preserve articles and metrics; never attach a vector to an edited text."""
    updated = skipped = 0
    async with await connect_database(dsn) as conn:
        texts, vectors = ArticleTextTable(conn), ArticleVectorTable(conn)
        rows = await texts.list_embedding_inputs()
        embedder = await load_embedder() if rows else None
        for row in rows:
            embedding = await asyncio.to_thread(embedder.embed_document, row["text"])
            async with conn.transaction():
                current = await texts.get(row["id"], for_update=True)
                if current is None or current["text"] != row["text"]:
                    skipped += 1
                    continue
                await vectors.upsert(row["id"], embedding)
            updated += 1
            if updated % 100 == 0:
                log(f"Recomputed {updated}/{len(rows)} article embeddings")
        await vectors.ensure_compatible()
    return {"updated": updated, "skipped_changed_or_deleted": skipped}
