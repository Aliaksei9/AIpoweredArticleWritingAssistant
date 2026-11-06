"""Regressors sharing the same numeric features and score contract."""

import numpy as np
from sklearn.utils.validation import check_is_fitted

from .features import FEATURE_DIMENSIONS


class PopularityRegressor:
    name: str

    def _features(self, features) -> np.ndarray:
        array = np.asarray(features, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != FEATURE_DIMENSIONS:
            raise ValueError(f"Features must have shape (n, {FEATURE_DIMENSIONS})")
        if not np.isfinite(array).all():
            raise ValueError("Features must be finite")
        return array

    def fit(self, features, scores):
        features = self._features(features)
        targets = np.asarray(scores, dtype=np.float64)
        if (targets.shape != (len(features),) or not len(targets)
                or not np.isfinite(targets).all() or np.any(np.abs(targets) > 10)):
            raise ValueError("Training scores must be finite values in [-10, 10], one per row")
        self.estimator.fit(features, targets)
        return self

    def predict(self, features) -> np.ndarray:
        check_is_fitted(self.estimator)
        features = self._features(features)
        if not len(features):
            return np.empty(0, dtype=np.float64)
        predictions = np.asarray(self.estimator.predict(features), dtype=np.float64)
        if not np.isfinite(predictions).all():
            raise RuntimeError("Regressor produced nonfinite predictions")
        return np.clip(predictions, -10.0, 10.0)


class NeuralNetworkModel(PopularityRegressor):
    """A frozen-SBERT MLP head, trained with MSE on metric-based labels."""

    name = "neural_network"

    def __init__(self, seed: int = 42, **parameters):
        from .neural_network import TorchMLPRegressor

        self.estimator = TorchMLPRegressor(seed=seed, **parameters)


MODEL_CLASSES = {model.name: model for model in
                 (NeuralNetworkModel,)}


def create_model(name: str, seed: int = 42, **parameters) -> PopularityRegressor:
    if name not in MODEL_CLASSES:
        raise ValueError(f"Unknown model {name!r}; choose from {list(MODEL_CLASSES)}")
    return MODEL_CLASSES[name](seed=seed, **parameters)
