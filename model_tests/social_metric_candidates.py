"""Candidate model families for module 4; never imported by production code."""

import warnings
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

from src.social_metrics_prediction.predictor import SocialMetricsPredictor

MODEL_NAMES = ("gradient_boosting", "random_forest", "catboost", "neural_network")


def create_candidate(name, seed=42, threads=4):
    if name == "gradient_boosting":
        return HistGradientBoostingRegressor(max_iter=200, max_leaf_nodes=15,
            min_samples_leaf=20, learning_rate=.05, l2_regularization=2,
            early_stopping=False, random_state=seed)
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=200, max_depth=12,
            min_samples_leaf=5, max_features=.8, n_jobs=threads, random_state=seed)
    if name == "catboost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(iterations=400, depth=6, learning_rate=.04,
            l2_leaf_reg=5, loss_function="RMSE", thread_count=threads,
            random_seed=seed, verbose=False, allow_writing_files=False)
    if name == "neural_network":
        estimator = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32),
            activation="relu", solver="adam", alpha=.1, batch_size=128,
            learning_rate_init=.001, max_iter=300, early_stopping=False,
            n_iter_no_change=300, random_state=seed))
        return TransformedTargetRegressor(regressor=estimator, transformer=StandardScaler())
    raise ValueError(f"Unknown candidate {name}")


class CandidateSocialMetricsPredictor(SocialMetricsPredictor):
    def __init__(self, name, seed=42, threads=4):
        super().__init__(seed, threads)
        self.name = name

    def _new_estimator(self, metric_index):
        return create_candidate(self.name, self.seed + metric_index, self.threads)

    def _fit_estimator(self, estimator, features, targets, weights):
        if self.name == "neural_network":
            estimator.fit(features, targets, mlpregressor__sample_weight=weights)
        else:
            super()._fit_estimator(estimator, features, targets, weights)

    def fit(self, rows, counts):
        # This candidate intentionally uses a fixed epoch budget; store the
        # iteration count rather than emitting the expected max_iter warning.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            super().fit(rows, counts)
        self.metadata["model_family"] = self.name
        self.metadata.pop("model_configuration", None)
        return self


class ResourceMeanBaseline:
    """Arithmetic mean counts per subscriber, separately for each resource."""
    def fit(self, rows, counts):
        import numpy as np
        rates = counts / np.asarray([r["audience"] for r in rows])[:, None]
        self.global_mean = rates.mean(axis=0)
        keys = [(r["source"], r["resource_id"]) for r in rows]
        self.means = {key: rates[[i for i, k in enumerate(keys) if k == key]].mean(axis=0)
                      for key in set(keys)}
        return self

    def predict_rows(self, rows):
        import numpy as np
        return np.rint([self.means.get((r["source"], r["resource_id"]), self.global_mean)
                        * r["audience"] for r in rows])
