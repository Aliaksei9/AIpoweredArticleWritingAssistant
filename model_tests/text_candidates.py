"""Alternative text regressors used only for experiments and historical artifacts."""

from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.utils.validation import check_is_fitted

from src.assesment_conform_of_PP_text_based.assessment import AssesmentConformPPtextBased
from src.assesment_conform_of_PP_text_based.features import FEATURE_DIMENSIONS, TextTopicEncoder
from src.assesment_conform_of_PP_text_based.model_classes import NeuralNetworkModel


class PopularityRegressor:
    """Common score contract for experimental regressors with unbounded outputs."""

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


class GradientBoostingModel(PopularityRegressor):
    """Histogram gradient boosting, with early stopping disabled for group CV."""

    name = "gradient_boosting"

    def __init__(self, seed: int = 42, **parameters):
        config = dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                      min_samples_leaf=20, l2_regularization=10.0,
                      early_stopping=False, random_state=seed)
        config.update(parameters)
        self.estimator = HistGradientBoostingRegressor(**config)


class RandomForestModel(PopularityRegressor):
    name = "random_forest"

    def __init__(self, seed: int = 42, **parameters):
        config = dict(n_estimators=160, max_depth=12, min_samples_leaf=6,
                      max_features=0.25, n_jobs=4, random_state=seed)
        config.update(parameters)
        self.estimator = RandomForestRegressor(**config)


class CatBoostModel(PopularityRegressor):
    name = "catboost"

    def __init__(self, seed: int = 42, **parameters):
        from catboost import CatBoostRegressor

        config = dict(iterations=400, depth=6, learning_rate=0.04,
                      l2_leaf_reg=5.0, loss_function="RMSE", random_seed=seed,
                      thread_count=4, verbose=False, allow_writing_files=False)
        config.update(parameters)
        self.estimator = CatBoostRegressor(**config)



MODEL_CLASSES = {model.name: model for model in
                 (GradientBoostingModel, RandomForestModel, CatBoostModel, NeuralNetworkModel)}

class ExperimentalTextAssessment(AssesmentConformPPtextBased):
    def __init__(self, model_name="neural_network", encoder=None, seed=42, **parameters):
        if model_name not in MODEL_CLASSES:
            raise ValueError(f"Unknown experimental model: {model_name}")
        self.encoder = encoder or TextTopicEncoder()
        self.model = MODEL_CLASSES[model_name](seed=seed, **parameters)
        self.metadata = {}

    @classmethod
    def load(cls, path):
        # Old reports referenced production class paths before alternatives were moved.
        from joblib.numpy_pickle import NumpyUnpickler, _validate_fileobject_and_memmap
        class LegacyUnpickler(NumpyUnpickler):
            def find_class(self, module, name):
                if module == "src.assesment_conform_of_PP_text_based.model_classes" and name in {
                    "GradientBoostingModel", "RandomForestModel", "CatBoostModel"}:
                    return globals()[name]
                return super().find_class(module, name)
        path = Path(path)
        with path.open("rb") as raw:
            with _validate_fileobject_and_memmap(raw, str(path), None) as (stream, mmap_mode):
                payload = LegacyUnpickler(str(path), stream, ensure_native_byte_order=True,
                                          mmap_mode=mmap_mode).load()
        return payload["assessment"]
