"""Independent social metric models with one complete shared feature schema."""

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

    def __init__(self, seed=42, threads=4,feature_mode='unified'):
        if feature_mode not in ('legacy','unified'):raise ValueError('Unknown social feature mode')
        self.seed, self.threads = seed, threads
        self.feature_mode=feature_mode
        self.resources: list[ResourceInfo] = []
        self.metadata = {}

    def _new_estimator(self, metric_index):
        from .selected_model import create_estimator
        categories = self.encoder_.categories_.categories_
        offset = len(categories[0])
        resource_columns = tuple(range(offset,offset+len(categories[1])))
        return create_estimator(self.seed + metric_index,self.threads,resource_columns)

    def _fit_estimator(self, estimator, features, targets, weights):
        estimator.fit(features, targets, sample_weight=weights)

    @property
    def uses_extended_features(self):return getattr(self,'feature_mode','legacy')=='unified'

    @property
    def shared_input_fields(self):
        from .unified_features import INPUT_FIELDS
        return list(INPUT_FIELDS) if getattr(self,'feature_mode','legacy')=='unified' else []

    @property
    def horizon_sensitive_metrics(self):
        return list(METRICS) if self.uses_extended_features else []

    def horizon_metrics_for(self,resource,source='TG'):
        return self.horizon_sensitive_metrics

    def fit(self, rows, counts, *, document_embeddings=None,groups=None,configuration=None):
        """rows contain predicted module-3 scores, not scores derived from counts."""
        counts = validate_counts(counts, len(rows))
        if self.feature_mode=='unified':
            from .unified_models import fit_heads
            from .unified_features import FEATURE_VERSION as SHARED_VERSION,document_vectors,INPUT_FIELDS
            from src.database.embedding_config import EMBEDDING_SIGNATURE,EMBEDDING_DIMENSIONS
            if configuration is None and groups is None:raise ValueError('Text groups are required for model selection')
            vectors=document_vectors(document_embeddings,len(rows))
            self.shared_encoder_,self.estimators_,selection=fit_heads(rows,counts,vectors,groups,configuration=configuration)
            self.encoder_=self.shared_encoder_.categories
            self.metadata.update(trained_at=datetime.now(timezone.utc).isoformat(),training_rows=len(rows),
                feature_version=SHARED_VERSION,model_family='independent_models_shared_inputs',
                shared_input_fields=list(INPUT_FIELDS),selection=selection,
                embedding_signature=EMBEDDING_SIGNATURE,embedding_dimensions=EMBEDDING_DIMENSIONS,
                target_horizon_range_days=[2.,14.],default_horizon_days=7.,
                one_row_per_article=True,score_input='out-of-fold text/topic-only neural predictions')
            return self
        self.encoder_ = MetricFeatureEncoder().fit(rows)
        features = self.encoder_.transform(rows)
        self.target_transform_ = "rate"
        targets = transform_targets(counts, rows, self.target_transform_)
        audience = np.asarray([r["audience"] for r in rows], dtype=float)
        weights = (audience / audience.max()) ** 2
        weights /= weights.mean()
        self.estimators_ = [self._new_estimator(i) for i in range(3)]
        dates = [row.get('published_at') for row in rows]
        history_model = False
        for i, estimator in enumerate(self.estimators_):
            if hasattr(estimator,'fit_with_dates'):
                estimator.fit_with_dates(features,targets[:,i],published_at=dates)
                history_model = True
            else:
                self._fit_estimator(estimator, features, targets[:, i], weights)
        from .selected_model import MODEL_CONFIGURATION
        self.metadata.update(trained_at=datetime.now(timezone.utc).isoformat(),
                             model_family=MODEL_FAMILY,
                             model_configuration=dict(MODEL_CONFIGURATION),
                             training_rows=len(rows), metrics=list(METRICS),
                             targets="count / observation_audience * 1000",
                             loss_weights=("per-channel exponential recency weights, training-only upper cap, topic shrinkage and local score ridge"
                                if history_model else "audience squared, normalized to mean 1; equivalent to count MSE"),
                             score_input="out-of-fold predictions of text neural network")
        if history_model:
            self.metadata['publication_dates_known'] = sum(value is not None for value in dates)
        return self

    def predict_rows(self, rows, *, document_embeddings=None):
        check_is_fitted(self, "estimators_")
        if getattr(self,'feature_mode','legacy')=='unified':
            features=self.shared_encoder_.transform(rows,document_embeddings)
            cache={}
            values=np.column_stack([head.predict(features,rows,cache=cache) for head in self.estimators_])
            if not np.isfinite(values).all():raise RuntimeError('Invalid count prediction')
            return values
        features = self.encoder_.transform(rows)
        values = np.column_stack([estimator.predict(features) for estimator in self.estimators_])
        return inverse_targets(values, rows, getattr(self, "target_transform_", "log_rate"))

    def set_resources(self, resources):
        self.resources = list(resources)
        if not all(isinstance(r, ResourceInfo) for r in self.resources):
            raise ValueError("Resources must be ResourceInfo instances")
        return self

    def predict(self, topic: str, popularity_score: float, resource, source: str = "TG", *,
                document_embedding=None,published_at=None):
        check_is_fitted(self, "estimators_")
        source = normalize_source(source)
        if source not in self.encoder_.sources_:
            raise ValueError(f"Source has no training data; trained sources: {sorted(self.encoder_.sources_)}")
        info = resolve_resource(self.resources, resource, source)
        row = dict(topic=topic, score=popularity_score, resource_id=info.id,
                   source=info.source, audience=info.audience)
        if self.uses_extended_features:
            from .forecast_context import forecast_context
            row.update(forecast_context(published_at))
        values = self.predict_rows([row],document_embeddings=[document_embedding] if document_embedding is not None else None)[0]
        return SocialMetricsPrediction(*[int(v) for v in values])

    def save(self, path):
        check_is_fitted(self, "estimators_")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        from .unified_features import FEATURE_VERSION as SHARED_VERSION
        version=SHARED_VERSION if self.uses_extended_features else FEATURE_VERSION
        joblib.dump(dict(artifact_version=self.ARTIFACT_VERSION, feature_version=version,
                         predictor=self), temporary, compress=3)
        temporary.replace(path)

    @classmethod
    def load(cls, path):
        payload = joblib.load(path)
        from .unified_features import FEATURE_VERSION as SHARED_VERSION
        if (payload.get("artifact_version") != cls.ARTIFACT_VERSION
                or payload.get("feature_version") not in (FEATURE_VERSION,SHARED_VERSION)
                or type(payload.get("predictor")) is not cls):
            raise ValueError("Incompatible social metrics artifact")
        predictor=payload['predictor']
        if predictor.uses_extended_features:
            from src.database.embedding_config import EMBEDDING_SIGNATURE,EMBEDDING_DIMENSIONS
            if (predictor.metadata.get('embedding_signature')!=EMBEDDING_SIGNATURE or
                predictor.metadata.get('embedding_dimensions')!=EMBEDDING_DIMENSIONS or
                payload.get('feature_version')!=SHARED_VERSION):
                raise ValueError('Social artifact requires a different embedding profile; reembed and retrain')
        return predictor
