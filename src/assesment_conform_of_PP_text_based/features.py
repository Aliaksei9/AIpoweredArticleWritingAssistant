"""Frozen SBERT features: document embedding followed by topic embedding."""

from collections.abc import Sequence

import numpy as np

from src.database.model_embedding import EMBEDDING_DIMENSIONS, MODEL_NAME, TextEmbedder


FEATURE_DIMENSIONS = 2 * EMBEDDING_DIMENSIONS
FEATURE_VERSION = "e5-document-mean-topic-query-v1"


class TextTopicEncoder:
    def __init__(self, model_name: str = MODEL_NAME, embedder=None):
        self.model_name = model_name
        self._embedder = embedder
        self.topic_embeddings: dict[str, list[float]] = {}

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = TextEmbedder(self.model_name)
        return self._embedder

    def __getstate__(self):
        # Store the feature contract and cached topics, not PyTorch weights.
        state = self.__dict__.copy()
        state["_embedder"] = None
        return state

    @staticmethod
    def _nonempty(value: str, name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a nonempty string")
        return value.strip()

    @staticmethod
    def _vector(values) -> np.ndarray:
        vector = np.asarray(values, dtype=np.float32)
        if vector.shape != (EMBEDDING_DIMENSIONS,) or not np.isfinite(vector).all():
            raise ValueError(f"Embedding must contain {EMBEDDING_DIMENSIONS} finite values")
        norm = float(np.linalg.norm(vector))
        if norm == 0:
            raise ValueError("Embedding must not be zero")
        return vector / norm

    def topic_embedding(self, topic: str) -> np.ndarray:
        topic = self._nonempty(topic, "Topic")
        if topic not in self.topic_embeddings:
            vector = self._vector(self.embedder.embed_query(f"Тема статьи: {topic}"))
            self.topic_embeddings[topic] = vector.tolist()
        return self._vector(self.topic_embeddings[topic])

    def transform(
        self,
        texts: Sequence[str],
        topics: Sequence[str],
        text_embeddings: Sequence | None = None,
    ) -> np.ndarray:
        if isinstance(texts, str) or isinstance(topics, str) or len(texts) != len(topics):
            raise ValueError("Texts and topics must be equally sized sequences")
        if text_embeddings is not None and len(text_embeddings) != len(texts):
            raise ValueError("Text embedding count does not match text count")
        features = np.empty((len(texts), FEATURE_DIMENSIONS), dtype=np.float32)
        for index, (text, topic) in enumerate(zip(texts, topics)):
            self._nonempty(text, "Text")
            cached = None if text_embeddings is None else text_embeddings[index]
            document = self._vector(
                self.embedder.embed_document(text) if cached is None else cached
            )
            features[index, :EMBEDDING_DIMENSIONS] = document
            features[index, EMBEDDING_DIMENSIONS:] = self.topic_embedding(topic)
        return features
