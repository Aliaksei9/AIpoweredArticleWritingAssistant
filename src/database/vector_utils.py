"""Shared validation and normalization for article and topic embeddings."""

import math
from collections.abc import Sequence


from .embedding_config import EMBEDDING_DIMENSIONS


def validate_vector(embedding: Sequence[float]) -> list[float]:
    """Return finite, nonzero coordinates without changing their magnitude."""
    values = [float(value) for value in embedding]
    if len(values) != EMBEDDING_DIMENSIONS:
        raise ValueError(f"Embedding must have {EMBEDDING_DIMENSIONS} dimensions")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Embedding must contain finite numbers")
    if not any(values):
        raise ValueError("Embedding must not be a zero vector")
    return values


def normalize_vector(embedding: Sequence[float]) -> list[float]:
    """Return a unit vector, avoiding overflow for large finite coordinates."""
    values = validate_vector(embedding)
    scale = max(abs(value) for value in values)
    scaled = [value / scale for value in values]
    norm = math.hypot(*scaled)
    return [value / norm for value in scaled]
