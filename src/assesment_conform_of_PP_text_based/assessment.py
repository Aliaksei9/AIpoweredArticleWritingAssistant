"""Train and persist an assessment model; predict from text and topic only."""

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from src.assesment_conform_of_PP_metric_based import AssesmentConformPPmetricsBased

from .features import FEATURE_DIMENSIONS, FEATURE_VERSION, TextTopicEncoder
from .model_classes import create_model


class AssesmentConformPPtextBased:
    ARTIFACT_VERSION = 1

    def __init__(self, model_name: str = "neural_network", encoder: TextTopicEncoder | None = None,
                 seed: int = 42, **model_parameters):
        self.encoder = encoder or TextTopicEncoder()
        self.model = create_model(model_name, seed, **model_parameters)
        self.metadata: dict = {}

    def train(self, texts, topics, scores, text_embeddings=None):
        """scores are labels from module 2, never prediction features."""
        features = self.encoder.transform(texts, topics, text_embeddings)
        return self.train_features(features, scores)

    def train_features(self, features, scores):
        self.model.fit(features, scores)
        self.metadata.update(
            trained_at=datetime.now(timezone.utc).isoformat(),
            training_rows=len(scores),
            target_formula_version=AssesmentConformPPmetricsBased.FORMULA_VERSION,
            target_formula_parameters=asdict(AssesmentConformPPmetricsBased()),
        )
        return self

    def predict(self, texts, topics) -> list[float]:
        return self.model.predict(self.encoder.transform(texts, topics)).tolist()

    def assess(self, text: str, topic: str) -> float:
        return self.predict([text], [topic])[0]

    def save(self, path: str | Path) -> None:
        # Confirm that this is a fitted artifact before saving it.
        self.model.predict(np.empty((0, FEATURE_DIMENSIONS)))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(artifact_version=self.ARTIFACT_VERSION,
                       feature_version=FEATURE_VERSION,
                       feature_dimensions=FEATURE_DIMENSIONS,
                       embedding_model=self.encoder.model_name,
                       model_name=self.model.name, assessment=self)
        temporary = path.with_suffix(path.suffix + ".tmp")
        joblib.dump(payload, temporary, compress=3)
        temporary.replace(path)

    @classmethod
    def load(cls, path: str | Path):
        payload = joblib.load(path)
        if (payload.get("artifact_version") != cls.ARTIFACT_VERSION
                or payload.get("feature_version") != FEATURE_VERSION
                or payload.get("feature_dimensions") != FEATURE_DIMENSIONS):
            raise ValueError("Incompatible model artifact or feature format")
        assessment = payload.get("assessment")
        if (not isinstance(assessment, cls)
                or assessment.encoder.model_name != payload.get("embedding_model")
                or assessment.model.name != payload.get("model_name")
                or assessment.model.name != "neural_network"):
            raise ValueError("Invalid assessment artifact")
        if assessment.metadata.get("target_formula_version") != AssesmentConformPPmetricsBased.FORMULA_VERSION:
            raise ValueError("Outdated metric formula in text model; recalculate DB scores and retrain modules 3/4")
        return assessment
