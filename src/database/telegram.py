"""Telegram API implementation of the data collection interface."""

import asyncio
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from datetime import datetime as PublicationDateTime

import psycopg
from telethon import TelegramClient
from telethon.errors import RPCError
from telethon.tl.functions.channels import GetFullChannelRequest
from telethon.tl.types import Channel

from .bd_collection import (
    DEFAULT_LIMIT,
    MAX_ARTICLE_AGE_DAYS,
    BDdataCollection,
    CollectionResult,
    validate_limit,
    validate_max_age_days,
)
from .model_embedding import EmptyTextError, TextEmbedder
from .ids_converter import article_id_for, resource_id_for
from .topic_detector import ArticleTopicDetector
from .table_classes import (
    ArticleMetricTable,
    ArticleTableOfSources,
    ArticleTextTable,
    ArticleVectorTable,
    ResourceTable,
)


def reaction_count(message: object) -> int | None:
    reactions = getattr(message, "reactions", None)
    if reactions is None:
        return None
    return sum(item.count for item in reactions.results)


class TGdataCollection(BDdataCollection):
    def __init__(
        self,
        conn: psycopg.AsyncConnection,
        client: TelegramClient,
        embedder: TextEmbedder,
        topic_detector: ArticleTopicDetector | None = None,
        *, max_age_days: float = MAX_ARTICLE_AGE_DAYS, as_of: datetime | None = None, on_progress=None,
    ):
        validate_max_age_days(max_age_days)
        self.max_age = timedelta(days=max_age_days)
        self.as_of = as_of
        self.on_progress = on_progress
        self.conn = conn
        self.client = client
        self.embedder = embedder
        self.topic_detector = topic_detector or ArticleTopicDetector(embedder)
        self.resources = ResourceTable(conn)
        self.texts = ArticleTextTable(conn)
        self.sources = ArticleTableOfSources(conn)
        self.metrics = ArticleMetricTable(conn)
        self.vectors = ArticleVectorTable(conn)
        self._last_cleanup_cutoff: datetime | None = None
        self._result = CollectionResult()

    @property
    def last_result(self) -> CollectionResult:
        return deepcopy(self._result)

    async def update(self, resource: str, limit: int | None = DEFAULT_LIMIT) -> int:
        """Remove expired articles, then add posts and refresh existing data."""
        result = CollectionResult(details={
            "scanned_messages": 0, "skipped_incomplete_metrics": 0, "skipped_unembeddable_text": 0,
        })
        self._result = result
        if limit is not None:
            validate_limit(limit)

        as_of = self.utc(self.as_of or datetime.now(timezone.utc))
        oldest, newest = self.publication_window(as_of, self.max_age)
        result.window = {"as_of": as_of, "oldest": oldest, "newest": newest}
        channel = await self.client.get_entity(resource)
        if not isinstance(channel, Channel):
            raise ValueError("resource must identify a Telegram channel or supergroup")

        # The CLI shares one cutoff across channels, so prune the database once.
        if self._last_cleanup_cutoff != oldest:
            result.deleted_expired = await self.texts.delete_expired(oldest)
            self._last_cleanup_cutoff = oldest

        reader_count = getattr(channel, "participants_count", None)
        try:
            full = await self.client(GetFullChannelRequest(channel))
            full_count = getattr(full.full_chat, "participants_count", None)
            if full_count is not None:
                reader_count = full_count
        except RPCError:
            # Some channels do not expose the count to the current account.
            pass

        resource_id = resource_id_for("TG", channel.id)
        await self.resources.upsert(resource_id, channel.title, reader_count)
        result.resource = {"id": str(resource_id), "name": channel.title, "audience": reader_count}
        # offset_date is exclusive; Telegram returns history newest to oldest.
        # Apply the limit to eligible text posts, not to raw media or fresh posts.
        async for message in self.client.iter_messages(channel, limit=None, offset_date=newest):
            result.details["scanned_messages"] += 1
            published_at = getattr(message, "date", None)
            if isinstance(published_at, PublicationDateTime) and self.utc(published_at) <= oldest:
                result.skipped_by_age += 1
                break
            body = (getattr(message, "raw_text", None) or "").strip()
            if not body or getattr(message, "action", None) is not None:
                continue

            if published_at is None:
                result.skipped_without_date += 1
                continue
            if not self.is_article_in_window(published_at, as_of, self.max_age):
                result.skipped_by_age += 1
                continue

            views = getattr(message, "views", None)
            reactions = reaction_count(message)
            comments = getattr(getattr(message, "replies", None), "replies", None)
            if (
                reader_count is None or reader_count <= 0
                or (views is not None and views <= 0)
                or any(value is None for value in (views, reactions, comments))
            ):
                # Views/audience must be positive; zero reactions/comments are valid.
                result.details["skipped_incomplete_metrics"] += 1
                continue

            article_id = article_id_for("TG", channel.id, message.id)
            previous = await self.texts.get(article_id)
            text_changed = previous is None or previous["text"] != body
            stored_vector = None if text_changed else await self.vectors.get(article_id)
            needs_vector = stored_vector is None
            try:
                embedding = (
                    await asyncio.to_thread(self.embedder.embed_document, body)
                    if needs_vector
                    else stored_vector["embedding"]
                )
            except EmptyTextError:
                result.details["skipped_unembeddable_text"] += 1
                continue
            topic = await asyncio.to_thread(self.topic_detector.detect_from_embedding, embedding)

            async with self.conn.transaction():
                if text_changed:
                    await self.texts.upsert(article_id, topic, body, self.utc(published_at))
                elif previous["topic"] != topic:
                    await self.texts.update_topic(article_id, topic)
                if not text_changed and previous.get("published_at") != self.utc(published_at):
                    await self.texts.set_publication_time(article_id, self.utc(published_at))
                await self.sources.link(article_id, resource_id, "TG")
                if needs_vector:
                    await self.vectors.upsert(article_id, embedding)

                await self.metrics.upsert(
                    article_id, views, reactions, comments, reader_count,
                )
            if previous is None:
                result.created += 1
            else:
                result.updated += 1
            if self.on_progress is not None and result.processed % 100 == 0:
                self.on_progress(result.processed)
            if limit is not None and result.processed >= limit:
                break
        return result.processed
