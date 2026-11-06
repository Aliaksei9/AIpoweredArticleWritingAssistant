"""PostgreSQL tables and extensible article data collection."""

from .bd_collection import BDdataCollection, CollectionOptions, CollectionResult
from .model_embedding import TextEmbedder
from .ids_converter import article_id_for, resource_id_for
from .topic_detector import ArticleTopicDetector
from .table_classes import (
    ArticleMetricTable,
    ArticleTableOfSources,
    ArticleTextTable,
    ArticleVectorTable,
    ResourceTable,
)

__all__ = [
    "BDdataCollection",
    "CollectionOptions",
    "CollectionResult",
    "TGdataCollection",
    "TextEmbedder",
    "ArticleTopicDetector",
    "article_id_for",
    "resource_id_for",
    "ResourceTable",
    "ArticleTextTable",
    "ArticleVectorTable",
    "ArticleMetricTable",
    "ArticleTableOfSources",
]


def __getattr__(name):
    if name == "TGdataCollection":
        from .telegram import TGdataCollection
        globals()[name] = TGdataCollection
        return TGdataCollection
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
