"""Connection setup and idempotent schema initialization."""

from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector_async
from psycopg.rows import dict_row

from src.assesment_conform_of_PP_metric_based import AssesmentConformPPmetricsBased

from .table_classes import ArticleMetricTable


async def initialize_database(dsn: str, *, recalculate_scores: bool = True) -> int:
    """Create/update the schema and assess all existing article metrics."""
    schema = Path(__file__).with_name("schema.sql").read_text(encoding="utf-8")
    async with await psycopg.AsyncConnection.connect(
        dsn, autocommit=True, row_factory=dict_row
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
        assessor = AssesmentConformPPmetricsBased()
        async with conn.transaction():
            return await assessor.assess_database(ArticleMetricTable(conn, assessor))


async def clear_database(dsn: str) -> dict[str, int]:
    """Explicit reset of the five application tables, preserving the database."""
    async with await connect_database(dsn) as conn:
        async with conn.transaction():
            counts = {}
            for table in ("resource", "article_text", "article_sources", "article_metrics", "article_vectors"):
                # Names come only from the fixed application schema above.
                cursor = await conn.execute(f"SELECT count(*) AS count FROM {table}")
                counts[table] = (await cursor.fetchone())["count"]
            await conn.execute("""TRUNCATE TABLE article_vectors, article_metrics,
                                  article_sources, article_text, resource""")
        return counts


async def connect_database(dsn: str) -> psycopg.AsyncConnection:
    conn = await psycopg.AsyncConnection.connect(
        dsn, autocommit=True, row_factory=dict_row
    )
    try:
        await register_vector_async(conn)
    except BaseException:
        # Include cancellation before ownership of the connection reaches its caller.
        await conn.close()
        raise
    return conn
