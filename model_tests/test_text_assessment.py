import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import joblib
from sklearn.exceptions import NotFittedError
from sklearn.metrics import mean_squared_error
from threadpoolctl import threadpool_limits

from src.assesment_conform_of_PP_text_based import AssesmentConformPPtextBased, TextTopicEncoder
from src.assesment_conform_of_PP_text_based.dataset import dataset_from_rows, text_group
from src.assesment_conform_of_PP_text_based.model_classes import create_model
from model_tests.data_split import split_dataset
from model_tests.text_candidates import ExperimentalTextAssessment


class FakeEmbedder:
    def __init__(self):
        self.document_calls = []
        self.topic_calls = []

    def embed_document(self, text):
        self.document_calls.append(text)
        vector = np.zeros(384)
        vector[:2] = [float(text.split()[-1]), 1]
        return vector

    def embed_query(self, text):
        self.topic_calls.append(text)
        vector = np.zeros(384)
        vector[2 if text.endswith("Игры") else 3] = 1
        return vector


class TextAssessmentTests(unittest.TestCase):
    def test_cached_and_raw_text_features_match_and_topic_is_cached(self):
        fake = FakeEmbedder()
        encoder = TextTopicEncoder(embedder=fake)
        texts, topics = ["article 1", "article 2"], ["Игры", "Игры"]
        cached = [fake.embed_document(text) for text in texts]
        features = encoder.transform(texts, topics, cached)
        self.assertEqual(len(fake.document_calls), 2)
        np.testing.assert_allclose(features, encoder.transform(texts, topics))
        self.assertEqual(features.shape, (2, 768))
        self.assertEqual(len(fake.topic_calls), 1)
        np.testing.assert_allclose(np.linalg.norm(features[:, :384], axis=1), 1)
        np.testing.assert_allclose(np.linalg.norm(features[:, 384:], axis=1), 1)
        encoder.transform(["article 3"], ["Новая тема"])
        self.assertEqual(len(fake.topic_calls), 2)

    def test_all_models_learn_and_round_trip_with_identical_predictions(self):
        values = np.linspace(-3, 3, 64)
        texts = [f"article {value}" for value in values]
        topics = ["Игры"] * len(texts)
        scores = 5 * np.tanh(values)
        options = {
            "gradient_boosting": dict(max_iter=30, min_samples_leaf=2, l2_regularization=0),
            "random_forest": dict(n_estimators=15, min_samples_leaf=1, max_features=1.0),
            "catboost": dict(iterations=40, depth=3, learning_rate=.2),
            "neural_network": dict(hidden_layers=(32, 16), dropout=0, epochs=80,
                                   learning_rate=.003, batch_size=64, num_threads=2),
        }
        with TemporaryDirectory() as directory, threadpool_limits(limits=2):
            for name, parameters in options.items():
                with self.subTest(model=name):
                    encoder = TextTopicEncoder(embedder=FakeEmbedder())
                    assessment = ExperimentalTextAssessment(name, encoder, **parameters)
                    assessment.train(texts, topics, scores)
                    prediction = assessment.predict(texts, topics)
                    self.assertLess(mean_squared_error(scores, prediction), np.var(scores) / 3)
                    path = Path(directory) / f"{name}.joblib"
                    assessment.save(path)
                    loaded = ExperimentalTextAssessment.load(path)
                    self.assertIsNone(loaded.encoder._embedder)
                    loaded.encoder._embedder = FakeEmbedder()
                    np.testing.assert_allclose(prediction, loaded.predict(texts, topics), atol=1e-8)
                    self.assertTrue(-10 <= loaded.assess("article 100", "Игры") <= 10)

    def test_invalid_inputs_and_untrained_model_fail(self):
        encoder = TextTopicEncoder(embedder=FakeEmbedder())
        for texts, topics, embeddings in [([""], ["Игры"], None), (["article 1"], [""], None),
                                          (["article 1"], [], None),
                                          (["article 1"], ["Игры"], [[0] * 384]),
                                          (["article 1"], ["Игры"], [[float("nan")] * 384])]:
            with self.assertRaises(ValueError):
                encoder.transform(texts, topics, embeddings)
        model = create_model("neural_network", epochs=1)
        with self.assertRaises(NotFittedError):
            model.predict(np.zeros((1, 768)))
        for scores in [[None], [float("nan")], [11], [-11], []]:
            with self.assertRaises(ValueError):
                model.fit(np.ones((1, 768)), scores)
        with self.assertRaises(ValueError):
            create_model("missing")

    def test_production_model_rejects_a_previous_target_formula(self):
        model = AssesmentConformPPtextBased(encoder=TextTopicEncoder(embedder=FakeEmbedder()), epochs=1)
        model.train_features(np.ones((4, 768)), [-1, 0, 1, 2])
        with TemporaryDirectory() as folder:
            path = Path(folder) / "model.joblib"
            model.save(path)
            self.assertEqual(AssesmentConformPPtextBased.load(path).metadata["training_rows"], 4)
            payload = joblib.load(path)
            payload["assessment"].metadata["target_formula_version"] = "log-relative-v2"
            joblib.dump(payload, path)
            with self.assertRaisesRegex(ValueError, "Outdated metric formula"):
                AssesmentConformPPtextBased.load(path)


class TrainingDatasetTests(unittest.TestCase):
    @staticmethod
    def row(index, text):
        return dict(article_id=str(index), text=text, topic="Игры", views=10000,
                    reactions=80, comments=5, audience=100000, score=0.0,
                    embedding=[1] + [0] * 383, resource_id="channel")

    def test_duplicate_text_groups_never_cross_the_split(self):
        rows = [self.row(1, "  Повтор  текста\n"), self.row(2, "повтор текста")]
        rows += [self.row(index, f"Статья {index}") for index in range(3, 30)]
        dataset = dataset_from_rows(rows)
        train, test = split_dataset(dataset)
        self.assertFalse(set(dataset.groups[train]) & set(dataset.groups[test]))
        self.assertEqual(dataset.report["duplicate_rows"], 1)
        self.assertEqual(text_group(rows[0]["text"]), text_group(rows[1]["text"]))
        self.assertEqual(len(dataset.fingerprint), 64)

    def test_stale_labels_are_rejected_and_missing_metrics_excluded(self):
        valid = self.row(1, "Статья")
        missing = self.row(2, "Без метрик")
        missing.update(comments=None, score=None)
        dataset = dataset_from_rows([valid, missing])
        self.assertEqual(dataset.report["excluded_missing_labels"], 1)
        stale = self.row(3, "Старая оценка")
        stale["score"] = -9
        with self.assertRaisesRegex(ValueError, "init-db"):
            dataset_from_rows([stale])
        empty = self.row(4, "  ")
        self.assertEqual(dataset_from_rows([valid, empty]).report["excluded_empty_texts"], 1)
