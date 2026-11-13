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
