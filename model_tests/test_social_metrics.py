"""Prediction contracts, serialization and separation from target-derived scores."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.exceptions import NotFittedError

from src.social_metrics_prediction import SocialMetricsPredictor, ResourceInfo
from src.social_metrics_prediction.dataset import metric_dataset_from_rows
from src.social_metrics_prediction.features import MetricFeatureEncoder
from src.social_metrics_prediction.training import cross_fitted_scores
from src.social_metrics_prediction.resources import resolve_resource
from src.assesment_conform_of_PP_text_based import AssesmentConformPPtextBased
from src.database.model_embedding import MODEL_NAME


class SocialMetricsTests(unittest.TestCase):
    @staticmethod
    def rows():
        return [dict(topic="Игры" if i % 2 else "Кино", resource_id="resource",
                     source="TG", audience=100000, score=float(i-5)) for i in range(10)]

    def test_prediction_round_trip_and_audience_scaling(self):
        rows = self.rows()
        targets = np.asarray([[1000*(i+1), 10*(i+1), i] for i in range(10)])
        with patch.object(SocialMetricsPredictor, "_new_estimator",
                          lambda self, i: DecisionTreeRegressor(random_state=i)):
            model = SocialMetricsPredictor(feature_mode='legacy').fit(rows, targets)
        model.set_resources([ResourceInfo("resource", "Канал", 100000)])
        np.testing.assert_array_equal(model.predict_rows(rows), targets)
        value = model.predict("Кино", -5, "Канал", "Telegram")
        self.assertEqual(value.to_dict(), dict(views=1000, reactions=10, comments=0))
        doubled = ResourceInfo("resource", "Канал", 200000)
        # Numeric audience can affect a real estimator; this tree splits on score,
        # so the inverse rate transformation is isolated here.
        self.assertEqual(model.predict("Кино", -5, doubled).views, 2000)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            model.save(path)
            restored = SocialMetricsPredictor.load(path)
            self.assertEqual(value, restored.predict("Кино", -5, "resource"))

    def test_training_vocabulary_and_invalid_inputs(self):
        rows = self.rows()
        encoder = MetricFeatureEncoder().fit(rows)
        original = [c.copy() for c in encoder.categories_.categories_]
        encoder.transform([dict(rows[0], topic="Новая тема", resource_id="new")])
        for before, after in zip(original, encoder.categories_.categories_):
            np.testing.assert_array_equal(before, after)
        for changes in [dict(score=11), dict(score=float("nan")), dict(audience=0),
                        dict(source="VK"), dict(topic=""), dict(audience=1.5)]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                encoder.transform([dict(rows[0], **changes)])
        with self.assertRaises(NotFittedError):
            SocialMetricsPredictor(feature_mode='legacy').predict_rows(rows)
        with self.assertRaises(ValueError):
            SocialMetricsPredictor(feature_mode='legacy').fit(rows, np.full((10, 3), -1))
        with self.assertRaises(ValueError):
            resolve_resource([ResourceInfo("a", "same", 1), ResourceInfo("b", "same", 2)], "same", "TG")

    def test_crossfit_never_trains_on_predicted_text_group(self):
        features = np.arange(36).reshape(18, 2)
        labels = np.linspace(-5, 5, 18)
        groups = np.asarray([str(i//2) for i in range(18)])
        seen = []
        class SpyModel:
            def __init__(self):
                self.model = self
            def train_features(self, X, y):
                self.trained = set(X[:, 0])
                seen.append(self.trained)
            def predict(self, X):
                self.assert_disjoint = not self.trained.intersection(X[:, 0])
                if not self.assert_disjoint:
                    raise AssertionError("Validation row used in upstream training")
                return np.zeros(len(X))
        predictions, audit = cross_fitted_scores(features, labels, groups, factory=SpyModel)
        np.testing.assert_array_equal(predictions, np.zeros(18))
        visited = []
        for fold in audit:
            self.assertFalse(set(groups[fold["train_indices"]]) & set(groups[fold["validation_indices"]]))
            visited += fold["validation_indices"]
        self.assertEqual(sorted(visited), list(range(18)))

    def test_metric_labels_are_not_copied_into_prediction_inputs(self):
        rows = [dict(article_id=str(i), text=f"Статья {i}", topic="Игры", views=10000,
            reactions=80, comments=5, audience=100000, score=0.0, source="TG",
            resource_id="resource", embedding=[1]+[0]*383, embedding_model=MODEL_NAME) for i in range(3)]
        dataset = metric_dataset_from_rows(rows, [],feature_mode='legacy')
        self.assertNotIn("score", dataset.rows[0])
        inputs = dataset.inputs([0, 1], [2.5, -3])
        self.assertEqual(inputs[0]["score"], 2.5)
        for row in inputs:
            self.assertFalse({"views", "reactions", "comments", "text"} & row.keys())
        with self.assertRaises(TypeError):
            AssesmentConformPPtextBased("catboost")


if __name__ == "__main__":
    unittest.main()
