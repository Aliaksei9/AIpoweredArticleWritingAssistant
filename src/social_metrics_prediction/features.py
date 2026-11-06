"""Training-only categorical vocabulary; counts are never input features."""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted

from .resources import normalize_source

FEATURE_VERSION = "topic-resource-source-onehot-score-audience-v1"
METRICS = ("views", "reactions", "comments")


def input_columns(rows):
    if not rows:
        raise ValueError("At least one input row is required")
    categories, numeric = [], []
    for row in rows:
        topic, resource = row["topic"], row["resource_id"]
        if not isinstance(topic, str) or not topic.strip() or not resource:
            raise ValueError("Topic and resource identity must be nonempty")
        score, audience = float(row["score"]), float(row["audience"])
        if not np.isfinite(score) or not -10 <= score <= 10:
            raise ValueError("Popularity score must be finite and in [-10, 10]")
        if not np.isfinite(audience) or audience <= 0 or not audience.is_integer():
            raise ValueError("Audience must be a positive integer")
        categories.append([topic.strip(), str(resource), normalize_source(row["source"])])
        numeric.append([score / 10, np.log1p(audience)])
    return np.asarray(categories, dtype=object), np.asarray(numeric, dtype=np.float64)


class MetricFeatureEncoder(BaseEstimator):
    def fit(self, rows):
        categories, _ = input_columns(rows)
        self.categories_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                         dtype=np.float64).fit(categories)
        self.sources_ = set(categories[:, 2])
        return self

    def transform(self, rows):
        check_is_fitted(self, "categories_")
        categories, numeric = input_columns(rows)
        if not set(categories[:, 2]).issubset(self.sources_):
            raise ValueError(f"Source has no training data; trained sources: {sorted(self.sources_)}")
        return np.column_stack([self.categories_.transform(categories), numeric])


def validate_counts(counts, length):
    values = np.asarray(counts, dtype=np.float64)
    if values.shape != (length, 3) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Counts must have shape (n, 3) with finite nonnegative values")
    if np.any(values != np.rint(values)):
        raise ValueError("Training counts must be integers")
    return values


def transform_targets(counts, rows, mode="rate"):
    audience = np.asarray([r["audience"] for r in rows], dtype=float)
    rates = counts / audience[:, None] * 1000.0
    return np.log1p(rates) if mode == "log_rate" else rates


def inverse_targets(predictions, rows, mode="rate"):
    audience = np.asarray([r["audience"] for r in rows], dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        rates = np.maximum(0.0, predictions)
        if mode == "log_rate":
            rates = np.expm1(rates)
        counts = rates * audience[:, None] / 1000.0
    if counts.shape != (len(rows), 3) or not np.isfinite(counts).all():
        raise RuntimeError("Model produced invalid social metric predictions")
    return np.rint(counts)
