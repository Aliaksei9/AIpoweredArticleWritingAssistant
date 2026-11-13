"""Connection setup and idempotent schema initialization."""

from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector_async
from psycopg.rows import dict_row

from .table_classes import ArticleMetricTable


async def initialize_database(dsn: str, *, recalculate_scores: bool = True) -> int:
    """Create/update the schema and assess all existing article metrics."""
    schema = Path(__file__).with_name("schema.sql").read_text(encoding="utf-8")
    async with await psycopg.AsyncConnection.connect(
        dsn, autocommit=True, row_factory=dict_row,connect_timeout=10
    ) as conn:
        cursor = await conn.execute(
            """SELECT data_type FROM information_schema.columns
               WHERE table_schema = current_schema()
                 AND table_name = 'article_text' AND column_name = 'id'"""
        )
        existing = await cursor.fetchone()
        if existing is not None and existing["data_type"] != "uuid":
            raise RuntimeError(
                "Existing database uses the old BIGINT IDs. Back up and migrate it "
                "before using the UUID schema; initialization never drops data."
            )
        await conn.execute(schema)
        if not recalculate_scores:
            return 0
        return await ArticleMetricTable(conn).recalculate_scores()


async def clear_database(dsn: str) -> dict[str, int]:
    """Explicit reset of the five application tables."""
    async with await connect_database(dsn) as conn:
        async with conn.transaction():
            cursor = await conn.execute(
                """SELECT (SELECT count(*) FROM resource) AS resource,
                          (SELECT count(*) FROM article_text) AS article_text,
                          (SELECT count(*) FROM article_sources) AS article_sources,
                          (SELECT count(*) FROM article_metrics) AS article_metrics,
                          (SELECT count(*) FROM article_vectors) AS article_vectors"""
            )
            counts = await cursor.fetchone()
            await conn.execute("""TRUNCATE TABLE article_vectors, article_metrics,
                                  article_sources, article_text, resource""")
        return counts


async def connect_database(dsn: str) -> psycopg.AsyncConnection:
    conn = await psycopg.AsyncConnection.connect(
        dsn, autocommit=True, row_factory=dict_row,connect_timeout=10
    )
    try:
        await register_vector_async(conn)
    except BaseException:
        # Include cancellation before ownership of the connection reaches its caller.
        await conn.close()
        raise
    return conn
