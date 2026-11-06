"""Production predictor; one selected estimator family for three social counts."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from .features import (FEATURE_VERSION, METRICS, MetricFeatureEncoder, inverse_targets,
                       transform_targets, validate_counts)
from .resources import ResourceInfo, normalize_source, resolve_resource
from .selected_model import MODEL_FAMILY


@dataclass(frozen=True)
class SocialMetricsPrediction:
    views: int
    reactions: int
    comments: int

    def to_dict(self):
        return asdict(self)


class SocialMetricsPredictor(BaseEstimator):
    ARTIFACT_VERSION = 1

    def __init__(self, seed=42, threads=4):
        self.seed, self.threads = seed, threads
        self.resources: list[ResourceInfo] = []
        self.metadata = {}

    def _new_estimator(self, metric_index):
        from .selected_model import create_estimator
        return create_estimator(self.seed + metric_index, self.threads)

    def _fit_estimator(self, estimator, features, targets, weights):
        estimator.fit(features, targets, sample_weight=weights)

    def fit(self, rows, counts):
        """rows contain predicted module-3 scores, not scores derived from counts."""
        counts = validate_counts(counts, len(rows))
        self.encoder_ = MetricFeatureEncoder().fit(rows)
        features = self.encoder_.transform(rows)
        self.target_transform_ = "rate"
        targets = transform_targets(counts, rows, self.target_transform_)
        audience = np.asarray([r["audience"] for r in rows], dtype=float)
        weights = (audience / audience.max()) ** 2
        weights /= weights.mean()
        self.estimators_ = [self._new_estimator(i) for i in range(3)]
        for i, estimator in enumerate(self.estimators_):
            self._fit_estimator(estimator, features, targets[:, i], weights)
        self.metadata.update(trained_at=datetime.now(timezone.utc).isoformat(),
                             model_family=MODEL_FAMILY,
                             training_rows=len(rows), metrics=list(METRICS),
                             targets="count / observation_audience * 1000",
                             loss_weights="audience squared, normalized to mean 1; equivalent to count MSE",
                             score_input="out-of-fold predictions of text neural network")
        return self

    def predict_rows(self, rows):
        check_is_fitted(self, "estimators_")
        features = self.encoder_.transform(rows)
        values = np.column_stack([estimator.predict(features) for estimator in self.estimators_])
        return inverse_targets(values, rows, getattr(self, "target_transform_", "log_rate"))

    def set_resources(self, resources):
        self.resources = list(resources)
        if not all(isinstance(r, ResourceInfo) for r in self.resources):
            raise ValueError("Resources must be ResourceInfo instances")
        return self

    def predict(self, topic: str, popularity_score: float, resource, source: str = "TG"):
        check_is_fitted(self, "estimators_")
        source = normalize_source(source)
        if source not in self.encoder_.sources_:
            raise ValueError(f"Source has no training data; trained sources: {sorted(self.encoder_.sources_)}")
        info = resolve_resource(self.resources, resource, source)
        row = dict(topic=topic, score=popularity_score, resource_id=info.id,
                   source=info.source, audience=info.audience)
        values = self.predict_rows([row])[0]
        return SocialMetricsPrediction(*[int(v) for v in values])

    def save(self, path):
        check_is_fitted(self, "estimators_")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        joblib.dump(dict(artifact_version=self.ARTIFACT_VERSION, feature_version=FEATURE_VERSION,
                         predictor=self), temporary, compress=3)
        temporary.replace(path)

    @classmethod
    def load(cls, path):
        payload = joblib.load(path)
        if (payload.get("artifact_version") != cls.ARTIFACT_VERSION
                or payload.get("feature_version") != FEATURE_VERSION
                or type(payload.get("predictor")) is not cls):
            raise ValueError("Incompatible social metrics artifact")
        return payload["predictor"]
