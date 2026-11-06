"""Assign an article to a configured topic by embedding similarity."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .model_embedding import TextEmbedder
from .vector_utils import normalize_vector


DEFAULT_TOPICS_PATH = Path(__file__).with_name("topics.txt")


@dataclass(frozen=True)
class Topic:
    name: str
    description: str


class ArticleTopicDetector:
    def __init__(
        self, embedder: TextEmbedder, topics_path: str | Path = DEFAULT_TOPICS_PATH
    ):
        self.embedder = embedder
        self.topics = self._load_topics(Path(topics_path))
        self._topic_vectors: list[list[float]] | None = None

    @staticmethod
    def _load_topics(path: Path) -> list[Topic]:
        topics = []
        names = set()
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        for line_number, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, _, description = line.partition("|")
            name = name.strip()
            if not name or name.casefold() in names:
                raise ValueError(f"Empty or duplicate topic at {path}:{line_number}")
            names.add(name.casefold())
            topics.append(Topic(name, description.strip() or name))
        if not topics:
            raise ValueError(f"No topics configured in {path}")
        return topics

    def detect_from_embedding(self, embedding: Sequence[float]) -> str:
        article_vector = normalize_vector(embedding)
        if self._topic_vectors is None:
            self._topic_vectors = [
                normalize_vector(
                    self.embedder.embed_query(f"Статья на тему «{topic.name}»: {topic.description}")
                )
                for topic in self.topics
            ]
        similarities = [
            math.fsum(
                article_value * topic_value
                for article_value, topic_value in zip(article_vector, vector)
            )
            for vector in self._topic_vectors
        ]
        best_index = max(range(len(self.topics)), key=similarities.__getitem__)
        return self.topics[best_index].name
