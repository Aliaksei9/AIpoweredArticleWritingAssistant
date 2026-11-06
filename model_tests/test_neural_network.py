import unittest

import numpy as np
import torch
from sklearn.exceptions import NotFittedError

from src.assesment_conform_of_PP_text_based.model_classes import create_model
from src.assesment_conform_of_PP_text_based.neural_network import TorchMLPRegressor


class NeuralNetworkTests(unittest.TestCase):
    def setUp(self):
        generator = np.random.default_rng(42)
        self.features = generator.normal(size=(24, 768)).astype(np.float32)
        self.scores = 3 * np.tanh(self.features[:, 0])

    @staticmethod
    def make_estimator(**parameters):
        config = dict(hidden_layers=(8,), epochs=4, batch_size=8, dropout=.1,
                      num_threads=1, seed=42)
        config.update(parameters)
        return TorchMLPRegressor(**config)

    def test_scaler_uses_training_only_and_inference_is_deterministic(self):
        estimator = self.make_estimator().fit(self.features, self.scores)
        np.testing.assert_allclose(estimator.scaler_.mean_, self.features.mean(axis=0, dtype=np.float64))
        training_mean = estimator.scaler_.mean_.copy()
        shifted = self.features + 100
        first = estimator.predict(shifted)
        second = estimator.predict(shifted)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(estimator.scaler_.mean_, training_mean)
        self.assertTrue(np.all(np.abs(first) <= 10))

    def test_seed_is_reproducible_and_training_preserves_caller_rng_and_threads(self):
        state = torch.random.get_rng_state().clone()
        previous_threads = torch.get_num_threads()
        first = self.make_estimator().fit(self.features, self.scores)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertEqual(torch.get_num_threads(), previous_threads)
        second = self.make_estimator().fit(self.features, self.scores)
        np.testing.assert_array_equal(first.predict(self.features), second.predict(self.features))

    def test_untrained_and_invalid_parameters_are_rejected(self):
        with self.assertRaises(NotFittedError):
            create_model("neural_network").predict(self.features)
        for config in [dict(epochs=0), dict(batch_size=0), dict(dropout=1),
                       dict(hidden_layers=()), dict(num_threads=0), dict(learning_rate=0)]:
            with self.subTest(config=config), self.assertRaises(ValueError):
                self.make_estimator(**config).fit(self.features, self.scores)
