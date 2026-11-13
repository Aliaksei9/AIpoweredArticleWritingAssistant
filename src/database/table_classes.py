"""Read and write the single current state of each article."""

import math
from datetime import datetime,timezone
from collections.abc import Sequence
from typing import Any
from uuid import UUID

import psycopg
from pgvector import Vector

from src.assesment_conform_of_PP_metric_based import AssesmentConformPPmetricsBased

from .vector_utils import validate_vector
from .model_embedding import IncompatibleEmbeddingError
from .embedding_config import EMBEDDING_SIGNATURE as MODEL_NAME
from .bd_collection import validate_limit


Row = dict[str, Any]


class ResourceTable:
    def __init__(self, conn: psycopg.AsyncConnection):
        self.conn = conn

    async def upsert(self, resource_id: UUID, name: str, reader_count: int | None) -> None:
        await self.conn.execute(
            """INSERT INTO resource (id, name, reader_count)
               VALUES (%s, %s, %s)
               ON CONFLICT (id) DO UPDATE SET
                   name = EXCLUDED.name,
                   reader_count = COALESCE(EXCLUDED.reader_count, resource.reader_count)""",
            (resource_id, name, reader_count),
        )

    async def get(self, resource_id: UUID) -> Row | None:
        cursor = await self.conn.execute(
            "SELECT id, name, reader_count FROM resource WHERE id = %s", (resource_id,)
        )
        return await cursor.fetchone()

    async def list_all(self) -> list[Row]:
        cursor = await self.conn.execute(
            "SELECT id, name, reader_count FROM resource ORDER BY id"
        )
        return await cursor.fetchall()

    async def list_prediction_resources(self) -> list[Row]:
        """Resource identities, current audiences and generic source types."""
        cursor = await self.conn.execute(
            """SELECT DISTINCT r.id, r.name, r.reader_count, s.source
               FROM resource r JOIN article_sources s ON s.resource_id = r.id
               ORDER BY r.id, s.source"""
        )
        return await cursor.fetchall()


class ArticleTextTable:
    def __init__(self, conn: psycopg.AsyncConnection):
        self.conn = conn

    async def upsert(self, article_id: UUID, topic: str, text: str,
                     published_at: datetime | None = None) -> None:
        """Write text and invalidate any embedding of the previous text."""
        async with self.conn.transaction():
            await self.conn.execute(
                """INSERT INTO article_text (id, topic, text, published_at) VALUES (%s, %s, %s, %s)
                   ON CONFLICT (id) DO UPDATE SET
                       topic = EXCLUDED.topic, text = EXCLUDED.text,
                       published_at = EXCLUDED.published_at""",
                (article_id, topic, text, published_at),
            )
            await self.conn.execute(
                "DELETE FROM article_vectors WHERE article_id = %s", (article_id,)
            )

    async def get(self, article_id: UUID, *, for_update: bool = False) -> Row | None:
        lock = " FOR UPDATE" if for_update else ""
        cursor = await self.conn.execute(
            "SELECT id, topic, text, published_at FROM article_text WHERE id = %s" + lock,
            (article_id,),
        )
        return await cursor.fetchone()

    async def list_embedding_inputs(self) -> list[Row]:
        cursor = await self.conn.execute("SELECT id, text FROM article_text ORDER BY id")
        return await cursor.fetchall()

    async def update_topic(self, article_id: UUID, topic: str) -> None:
        """Change the category while preserving the unchanged text's vector."""
        await self.conn.execute(
            "UPDATE article_text SET topic = %s WHERE id = %s", (topic, article_id)
        )

    async def set_publication_time(self, article_id: UUID, published_at: datetime) -> None:
        await self.conn.execute("UPDATE article_text SET published_at = %s WHERE id = %s",
                                (published_at, article_id))

    async def delete_expired(self, cutoff: datetime) -> int:
        """Delete articles at/before the cutoff; dependent rows cascade."""
        cursor = await self.conn.execute(
            "DELETE FROM article_text WHERE published_at <= %s", (cutoff,)
        )
        return cursor.rowcount

    async def list_training_data(self, *, include_resource_metadata: bool = False) -> list[Row]:
        """One consistent read of text, topic, label and optional cached vector.

        Metrics are returned only for checking label freshness, never as model
        features. NULL labels are retained for the caller's exclusion report.
        """
        await ArticleVectorTable(self.conn).ensure_compatible()
        resource_columns = ", s.source, r.name AS resource_name, r.reader_count" if include_resource_metadata else ""
        resource_join = "LEFT JOIN resource r ON r.id = s.resource_id" if include_resource_metadata else ""
        cursor = await self.conn.execute(
            f"""SELECT t.id AS article_id, t.text, t.topic, t.published_at,
                      m.popularity_pattern_score AS score,
                      m.views, m.reactions, m.comments, m.audience, m.collected_at,
                      s.resource_id, v.embedding, v.embedding_model {resource_columns}
               FROM article_text t
               LEFT JOIN article_metrics m ON m.article_id = t.id
               LEFT JOIN article_sources s ON s.article_id = t.id
               LEFT JOIN article_vectors v ON v.article_id = t.id
               {resource_join}
               ORDER BY t.id"""
        )
        rows = await cursor.fetchall()
        for row in rows:
            if isinstance(row["embedding"], Vector):
                row["embedding"] = row["embedding"].to_list()
        return rows


class ArticleTableOfSources:
    def __init__(self, conn: psycopg.AsyncConnection):
        self.conn = conn

    async def link(
        self, article_id: UUID, resource_id: UUID, source: str
    ) -> None:
        await self.conn.execute(
            """INSERT INTO article_sources (article_id, resource_id, source)
               VALUES (%s, %s, %s)
               ON CONFLICT (article_id) DO UPDATE SET
                   resource_id = EXCLUDED.resource_id, source = EXCLUDED.source""",
            (article_id, resource_id, source),
        )

    async def get(self, article_id: UUID) -> Row | None:
        cursor = await self.conn.execute(
            """SELECT article_id, resource_id, source
               FROM article_sources WHERE article_id = %s""",
            (article_id,),
        )
        return await cursor.fetchone()

    async def list_by_resource(self, resource_id: UUID) -> list[Row]:
        cursor = await self.conn.execute(
            """SELECT article_id, resource_id, source
               FROM article_sources WHERE resource_id = %s
               ORDER BY article_id""",
            (resource_id,),
        )
        return await cursor.fetchall()


class ArticleMetricTable:
    def __init__(
        self,
        conn: psycopg.AsyncConnection,
        assessor: AssesmentConformPPmetricsBased | None = None,
    ):
        self.conn = conn
        self.assessor = assessor or AssesmentConformPPmetricsBased()

    async def upsert(
        self,
        article_id: UUID,
        views: int | None,
        reactions: int | None,
        comments: int | None,
        audience: int | None = None,
        *,collected_at: datetime | None = None,
    ) -> None:
        from .time_utils import utc_datetime
        collected_at=utc_datetime(collected_at or datetime.now(timezone.utc),field='collected_at')
        score = self.assessor.assess(views, reactions, comments, audience)
        async with self.conn.transaction():
            await self.conn.execute(
            """INSERT INTO article_metrics
               (article_id, views, reactions, comments, audience, popularity_pattern_score, collected_at)
               VALUES (%s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT (article_id) DO UPDATE SET
                   views = EXCLUDED.views,
                   reactions = EXCLUDED.reactions,
                   comments = EXCLUDED.comments,
                   audience = EXCLUDED.audience,
                   collected_at = EXCLUDED.collected_at,
                   popularity_pattern_score = EXCLUDED.popularity_pattern_score
               WHERE article_metrics.collected_at IS NULL
                  OR EXCLUDED.collected_at>=article_metrics.collected_at""",
            (article_id, views, reactions, comments, audience, score,collected_at),
        )

    async def get(self, article_id: UUID) -> Row | None:
        cursor = await self.conn.execute(
            """SELECT article_id, views, reactions, comments, popularity_pattern_score, audience, collected_at
               FROM article_metrics WHERE article_id = %s""",
            (article_id,),
        )
        return await cursor.fetchone()

    async def recalculate_scores(self) -> int:
        """Reassess current article metrics using their saved audiences."""
        processed = 0
        async with self.conn.transaction():
            cursor = await self.conn.execute(
                """SELECT m.article_id, m.views, m.reactions, m.comments, m.audience
                   FROM article_metrics m ORDER BY m.article_id FOR UPDATE"""
            )
            async for row in cursor:
                score = self.assessor.assess(
                    row["views"], row["reactions"], row["comments"], row["audience"]
                )
                await self.conn.execute(
                    """UPDATE article_metrics SET popularity_pattern_score = %s
                       WHERE article_id = %s""",
                    (score, row["article_id"]),
                )
                processed += 1
        return processed


class ArticleVectorTable:
    def __init__(self, conn: psycopg.AsyncConnection):
        self.conn = conn

    @staticmethod
    def _vector(embedding: Sequence[float]) -> Vector:
        return Vector(validate_vector(embedding))

    async def ensure_compatible(self) -> None:
        """Reject unknown and foreign vectors, even when dimensions coincide."""
        try:
            cursor = await self.conn.execute(
                """SELECT article_id, embedding_model FROM article_vectors
                   WHERE embedding_model IS DISTINCT FROM %s LIMIT 1""",
                (MODEL_NAME,),
            )
        except psycopg.errors.UndefinedColumn as error:
            raise IncompatibleEmbeddingError(
                "Database vectors have no model metadata. "
                "Run python -m src.database reembed to migrate and rebuild them."
            ) from error
        row = await cursor.fetchone()
        if row is not None:
            raise IncompatibleEmbeddingError(
                f"Incompatible database embedding for article {row['article_id']}: "
                f"{row['embedding_model'] or 'unknown model'}; expected {MODEL_NAME}. "
                "Run python -m src.database reembed."
            )

    async def upsert(self, article_id: UUID, embedding: Sequence[float]) -> None:
        vector = self._vector(embedding)
        await self.conn.execute(
            """INSERT INTO article_vectors (article_id, embedding, embedding_model)
               VALUES (%s, %s, %s)
               ON CONFLICT (article_id) DO UPDATE SET
                   embedding = EXCLUDED.embedding, embedding_model = EXCLUDED.embedding_model""",
            (article_id, vector, MODEL_NAME),
        )

    async def get(self, article_id: UUID) -> Row | None:
        cursor = await self.conn.execute(
            "SELECT article_id, embedding, embedding_model FROM article_vectors WHERE article_id = %s",
            (article_id,),
        )
        row = await cursor.fetchone()
        if row is not None and isinstance(row["embedding"], Vector):
            # pgvector 0.5 returns Vector; 0.4 returns an iterable numpy array.
            row["embedding"] = row["embedding"].to_list()
        return row

    async def delete(self, article_id: UUID) -> None:
        await self.conn.execute(
            "DELETE FROM article_vectors WHERE article_id = %s", (article_id,)
        )

    async def search(
        self, embedding: Sequence[float], limit: int = 10
    ) -> list[Row]:
        validate_limit(limit)
        vector = self._vector(embedding)
        await self.ensure_compatible()
        cursor = await self.conn.execute(
            """SELECT article_id, embedding <=> %s AS cosine_distance
               FROM article_vectors
               WHERE embedding_model = %s
               ORDER BY embedding <=> %s
               LIMIT %s""",
            (vector, MODEL_NAME, vector, limit),
        )
        return await cursor.fetchall()

    async def search_popular(
        self, embedding: Sequence[float], *, min_score: float = 5.0,
        min_similarity: float = 0.6, limit: int = 100,
        exclude_article_id: UUID | None = None,
    ) -> list[Row]:
        """Find nearest article texts with a known high observed metric score."""
        if not math.isfinite(min_score) or not -10 <= min_score <= 10:
            raise ValueError("Minimum popularity score must be finite and in [-10, 10]")
        if not math.isfinite(min_similarity) or not -1 <= min_similarity <= 1:
            raise ValueError("Minimum cosine similarity must be in [-1, 1]")
        validate_limit(limit)
        excluded = None if exclude_article_id is None else UUID(str(exclude_article_id))
        vector = self._vector(embedding)
        await self.ensure_compatible()
        cursor = await self.conn.execute(
            """SELECT t.id AS article_id, t.topic, t.text, t.published_at,
                      m.popularity_pattern_score AS score,
                      s.resource_id, s.source, r.name AS resource_name,
                      1 - (v.embedding <=> %(vector)s) AS text_similarity
               FROM article_vectors v
               JOIN article_text t ON t.id = v.article_id
               JOIN article_metrics m ON m.article_id = t.id
               LEFT JOIN article_sources s ON s.article_id = t.id
               LEFT JOIN resource r ON r.id = s.resource_id
               WHERE m.popularity_pattern_score >= %(min_score)s
                 AND v.embedding_model = %(embedding_model)s
                 AND btrim(t.text) <> '' AND btrim(t.topic) <> ''
                 AND (%(excluded)s::uuid IS NULL OR t.id <> %(excluded)s::uuid)
                 AND 1 - (v.embedding <=> %(vector)s) >= %(min_similarity)s
               ORDER BY v.embedding <=> %(vector)s,
                        m.popularity_pattern_score DESC, t.id
               LIMIT %(limit)s""",
            dict(vector=vector, min_score=min_score, min_similarity=min_similarity,
                 excluded=excluded, limit=limit, embedding_model=MODEL_NAME),
        )
        return await cursor.fetchall()

    async def best_by_topic(
        self, embedding: Sequence[float], *, topic: str, limit: int = 100,
        exclude_article_id: UUID | None = None,
    ) -> list[Row]:
        """Highest observed scores in the exact topic, without a similarity cutoff."""
        if not isinstance(topic, str) or not topic.strip():
            raise ValueError("Topic must be nonempty")
        validate_limit(limit)
        excluded = None if exclude_article_id is None else UUID(str(exclude_article_id))
        vector = self._vector(embedding)
        await self.ensure_compatible()
        cursor = await self.conn.execute(
            """SELECT t.id AS article_id, t.topic, t.text, t.published_at,
                      m.popularity_pattern_score AS score,
                      s.resource_id, s.source, r.name AS resource_name,
                      1 - (v.embedding <=> %(vector)s) AS text_similarity
               FROM article_text t
               JOIN article_metrics m ON m.article_id = t.id
               LEFT JOIN article_vectors v ON v.article_id = t.id
               LEFT JOIN article_sources s ON s.article_id = t.id
               LEFT JOIN resource r ON r.id = s.resource_id
               WHERE t.topic = %(topic)s AND btrim(t.text) <> ''
                 AND (v.article_id IS NULL OR v.embedding_model = %(embedding_model)s)
                 AND m.popularity_pattern_score IS NOT NULL
                 AND (%(excluded)s::uuid IS NULL OR t.id <> %(excluded)s::uuid)
               ORDER BY m.popularity_pattern_score DESC, t.id
               LIMIT %(limit)s""",
            dict(vector=vector, topic=topic.strip(), excluded=excluded, limit=limit,
                 embedding_model=MODEL_NAME),
        )
        return await cursor.fetchall()
