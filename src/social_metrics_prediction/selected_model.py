"""The selected family only: independent MLP regressors for social metrics."""

import warnings

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits

MODEL_FAMILY = "neural_network"


class SocialMetricNeuralNetwork(RegressorMixin, BaseEstimator):
    """Train-only scaling of inputs and target; fixed 300-epoch Adam budget."""

    def __init__(self, seed=42, threads=4):
        self.seed, self.threads = seed, threads

    def fit(self, X, y, sample_weight=None):
        network = MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
            alpha=.1, batch_size=128, learning_rate_init=.001, max_iter=300,
            early_stopping=False, n_iter_no_change=300, random_state=self.seed)
        estimator = TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(), network),
                                                transformer=StandardScaler())
        with threadpool_limits(limits=self.threads), warnings.catch_warnings():
            # Reaching the fixed budget is expected; iteration/loss history is saved.
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            estimator.fit(X, y, mlpregressor__sample_weight=sample_weight)
        self.estimator_ = estimator
        fitted = estimator.regressor_.named_steps["mlpregressor"]
        self.n_iter_, self.loss_curve_ = fitted.n_iter_, fitted.loss_curve_
        self.n_features_in_ = fitted.n_features_in_
        return self

    def predict(self, X):
        check_is_fitted(self, "estimator_")
        with threadpool_limits(limits=self.threads):
            return self.estimator_.predict(X)


def create_estimator(seed=42, threads=4):
    return SocialMetricNeuralNetwork(seed, threads)
