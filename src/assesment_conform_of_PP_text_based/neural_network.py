"""CPU PyTorch MLP with training-only scaling and bounded predictions."""

import math
from numbers import Integral

import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted, validate_data
from torch import nn

from .features import FEATURE_DIMENSIONS


class PopularityMLP(nn.Module):
    def __init__(self, input_dimensions: int, hidden_layers: tuple[int, ...], dropout: float):
        super().__init__()
        layers = []
        previous = input_dimensions
        for width in hidden_layers:
            layers.extend([nn.Linear(previous, width), nn.ReLU(), nn.Dropout(dropout)])
            previous = width
        layers.extend([nn.Linear(previous, 1), nn.Tanh()])
        self.layers = nn.Sequential(*layers)

    def forward(self, features):
        return 10.0 * self.layers(features).squeeze(-1)


class TorchMLPRegressor(RegressorMixin, BaseEstimator):
    """Fixed epochs keep validation/test labels out of the training procedure.

    Parameters and the scaler are serialized together with the model.
    The training history records MSE in score units, with Dropout disabled.
    """

    def __init__(self, hidden_layers=(256, 128), dropout=0.5, epochs=60,
                 batch_size=128, learning_rate=0.0005, weight_decay=0.1,
                 seed=42, num_threads=4):
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.seed = seed
        self.num_threads = num_threads

    def __sklearn_is_fitted__(self):
        return hasattr(self, "network_")

    def _validate_parameters(self):
        for name in ("epochs", "batch_size", "num_threads"):
            value = getattr(self, name)
            if not isinstance(value, Integral) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not self.hidden_layers or any(
            not isinstance(width, Integral) or isinstance(width, bool) or width <= 0
            for width in self.hidden_layers
        ):
            raise ValueError("hidden_layers must contain positive integers")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be nonnegative and finite")

    def fit(self, X, y):
        self._validate_parameters()
        targets = np.asarray(y, dtype=np.float64)
        if targets.ndim != 1 or np.any(np.abs(targets) > 10):
            raise ValueError("Training scores must be a one-dimensional array in [-10, 10]")
        features, targets = validate_data(self, X, targets, dtype=np.float32, y_numeric=True)
        if features.shape[1] != FEATURE_DIMENSIONS:
            raise ValueError(f"Features must have shape (n, {FEATURE_DIMENSIONS})")
        targets = targets.astype(np.float32)
        scaler = StandardScaler()
        inputs = torch.from_numpy(scaler.fit_transform(features).astype(np.float32))
        expected = torch.from_numpy(targets)
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(self.num_threads)
            # Keep caller RNG state intact; initialization, batches and Dropout
            # are reproducible on this CPU with the same seed and inputs.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.seed)
                network = PopularityMLP(features.shape[1], tuple(self.hidden_layers), self.dropout)
                optimizer = torch.optim.AdamW(network.parameters(), lr=self.learning_rate,
                                              weight_decay=self.weight_decay)
                loss_function = nn.MSELoss()
                generator = torch.Generator().manual_seed(self.seed)
                history = []
                for epoch in range(self.epochs):
                    network.train()
                    order = torch.randperm(len(inputs), generator=generator)
                    for start in range(0, len(inputs), self.batch_size):
                        batch = order[start:start + self.batch_size]
                        optimizer.zero_grad(set_to_none=True)
                        loss = loss_function(network(inputs[batch]), expected[batch])
                        if not torch.isfinite(loss):
                            raise RuntimeError("Neural network training produced a nonfinite loss")
                        loss.backward()
                        optimizer.step()
                    network.eval()
                    with torch.inference_mode():
                        mse = float(loss_function(network(inputs), expected).item())
                    history.append(dict(epoch=epoch + 1, train_mse=mse))
                self.network_ = network
                self.scaler_ = scaler
                self.history_ = history
                self.n_iter_ = self.epochs
        finally:
            torch.set_num_threads(previous_threads)
        return self

    def predict(self, X):
        check_is_fitted(self)
        features = validate_data(self, X, dtype=np.float32, reset=False, ensure_min_samples=0)
        if not len(features):
            return np.empty(0, dtype=np.float64)
        inputs = torch.from_numpy(self.scaler_.transform(features).astype(np.float32))
        self.network_.eval()
        with torch.inference_mode():
            predictions = self.network_(inputs).numpy().astype(np.float64)
        if not np.isfinite(predictions).all():
            raise RuntimeError("Neural network produced nonfinite predictions")
        return predictions
