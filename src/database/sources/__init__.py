"""Register new adapters here; the CLI and collection service stay generic."""

from .base import CollectionContext, SourceAdapter, SourceRegistry
from .telegram import TelegramSource


def build_registry() -> SourceRegistry:
    registry = SourceRegistry()
    registry.register(TelegramSource())
    return registry


__all__ = ["CollectionContext", "SourceAdapter", "SourceRegistry", "TelegramSource", "build_registry"]
