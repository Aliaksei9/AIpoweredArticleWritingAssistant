"""Interface for data collection procedures."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
import math
from typing import Callable


DEFAULT_LIMIT = 1000
MIN_ARTICLE_AGE_DAYS = 2
MAX_ARTICLE_AGE_DAYS = 14


def validate_limit(limit: int) -> None:
    if type(limit) is not int or limit <= 0:
        raise ValueError("--limit must be a positive integer")


def validate_max_age_days(max_age_days: float) -> None:
    if not math.isfinite(max_age_days) or max_age_days <= MIN_ARTICLE_AGE_DAYS:
        raise ValueError(f"--max-age-days must be finite and greater than {MIN_ARTICLE_AGE_DAYS}")


@dataclass(frozen=True)
class CollectionOptions:
    """Common parameters with one clock value for the whole run."""
    limit: int = DEFAULT_LIMIT
    max_age_days: float = MAX_ARTICLE_AGE_DAYS
    as_of: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        validate_limit(self.limit)
        validate_max_age_days(self.max_age_days)
        object.__setattr__(self, "as_of", BDdataCollection.utc(self.as_of))


@dataclass
class CollectionResult:
    """Update counters; callers receive a detached snapshot from last_result."""
    created: int = 0
    updated: int = 0
    skipped_by_age: int = 0
    skipped_without_date: int = 0
    deleted_expired: int = 0
    resource: dict | None = None
    window: dict[str, datetime] = field(default_factory=dict)
    details: dict = field(default_factory=dict)

    @property
    def processed(self) -> int:
        return self.created + self.updated

    def to_dict(self) -> dict:
        return dict(asdict(self), processed=self.processed,
                    window={key: value.isoformat() for key, value in self.window.items()})


class BDdataCollection(ABC):
    MIN_ARTICLE_AGE = timedelta(days=MIN_ARTICLE_AGE_DAYS)
    MAX_ARTICLE_AGE = timedelta(days=MAX_ARTICLE_AGE_DAYS)
    on_progress: Callable[[int], None] | None = None

    @property
    @abstractmethod
    def last_result(self) -> CollectionResult:
        """Counters must also be available after a partially completed update."""

    @staticmethod
    def utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @classmethod
    def publication_window(cls, as_of: datetime, max_age: timedelta | None = None) -> tuple[datetime, datetime]:
        maximum = cls.MAX_ARTICLE_AGE if max_age is None else max_age
        if maximum <= cls.MIN_ARTICLE_AGE:
            raise ValueError(f"Maximum article age must be greater than {MIN_ARTICLE_AGE_DAYS} days")
        as_of = cls.utc(as_of)
        return as_of - maximum, as_of - cls.MIN_ARTICLE_AGE

    @classmethod
    def is_article_in_window(cls, published_at: datetime | None, as_of: datetime,
                             max_age: timedelta | None = None) -> bool:
        if not isinstance(published_at, datetime):
            return False
        oldest, newest = cls.publication_window(as_of, max_age)
        return oldest < cls.utc(published_at) < newest

    @abstractmethod
    async def update(self, resource: str, limit: int | None = DEFAULT_LIMIT) -> int:
        """Add new articles and refresh existing data; return processed count."""
