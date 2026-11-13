"""Shared lifecycle and addressing contracts for article resources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
import os
from typing import TYPE_CHECKING

from ..bd_collection import BDdataCollection, CollectionOptions

if TYPE_CHECKING:
    from psycopg import AsyncConnection
    from ..model_embedding import TextEmbedder


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing {name}; set it in .env or the environment")
    return value


@dataclass(frozen=True)
class CollectionContext:
    connection: AsyncConnection
    embedder: TextEmbedder
    options: CollectionOptions


class SourceAdapter(ABC):
    code: str

    def normalize_resource(self, resource: str) -> str:
        resource = resource.strip()
        if not resource:
            raise ValueError("Resource address cannot be empty")
        return resource

    def validate_configuration(self) -> None:
        """Fail before opening the DB or loading the embedding model."""

    @abstractmethod
    def open(self, context: CollectionContext) -> AbstractAsyncContextManager[BDdataCollection]:
        """Own the API client lifecycle and yield one reusable collector."""

    async def login(self) -> None:
        raise ValueError(f"Source {self.code} does not support interactive login")


class SourceRegistry:
    def __init__(self):
        self._adapters: dict[str, SourceAdapter] = {}

    def register(self, adapter: SourceAdapter) -> None:
        code = adapter.code.strip().upper()
        if not code or code in self._adapters:
            raise ValueError(f"Empty or duplicate source code: {code!r}")
        self._adapters[code] = adapter

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(self._adapters)

    def get(self, code: str) -> SourceAdapter:
        try:
            return self._adapters[code.strip().upper()]
        except KeyError:
            raise ValueError(f"Unknown source {code!r}; available: {', '.join(self.codes)}") from None

    def select(self, code: str | None = None) -> tuple[SourceAdapter, ...]:
        if code is not None:
            return (self.get(code),)
        if not self._adapters:
            raise ValueError("No article sources are registered")
        return tuple(self._adapters.values())
