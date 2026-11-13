"""Train and persist an assessment model; predict from text and topic only."""

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.utils.validation import check_is_fitted

from src.assesment_conform_of_PP_metric_based import AssesmentConformPPmetricsBased
from src.database.model_embedding import MODEL_NAME
from src.database.embedding_config import EMBEDDING_SIGNATURE

from .features import FEATURE_DIMENSIONS, FEATURE_VERSION, TextTopicEncoder
from .model_classes import NeuralNetworkModel


class AssesmentConformPPtextBased:
    ARTIFACT_VERSION = 1

    def __init__(self, *, encoder: TextTopicEncoder | None = None, seed: int = 42,
                 **model_parameters):
        self.encoder = encoder or TextTopicEncoder()
        self.model = NeuralNetworkModel(seed=seed, **model_parameters)
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
            model_parameters=self.model.estimator.get_params(),
            text_strategy=getattr(self.model,'strategy','standard'),
        )
        return self

    def predict(self, texts, topics) -> list[float]:
        return self.model.predict(self.encoder.transform(texts, topics)).tolist()

    def assess(self, text: str, topic: str) -> float:
        return self.predict([text], [topic])[0]

    def save(self, path: str | Path) -> None:
        if self.encoder.model_name != MODEL_NAME:
            raise ValueError(f"Assessment requires configured embedding model {MODEL_NAME}")
        if getattr(self.encoder,'embedding_signature',self.encoder.model_name)!=EMBEDDING_SIGNATURE:
            raise ValueError('Embedding configuration changed; recompute vectors and retrain models')
        check_is_fitted(self.model.estimator)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(artifact_version=self.ARTIFACT_VERSION,
                       feature_version=FEATURE_VERSION,
                       feature_dimensions=FEATURE_DIMENSIONS,
                       embedding_model=self.encoder.model_name,
                       embedding_signature=EMBEDDING_SIGNATURE,
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
                or payload.get("embedding_model") != MODEL_NAME
                or assessment.encoder.model_name != payload.get("embedding_model")
                or payload.get('embedding_signature',payload.get('embedding_model'))!=EMBEDDING_SIGNATURE
                or getattr(assessment.encoder,'embedding_signature',assessment.encoder.model_name)!=EMBEDDING_SIGNATURE
                or assessment.model.name != payload.get("model_name")
                or assessment.model.name != NeuralNetworkModel.name):
            raise ValueError("Invalid assessment artifact")
        if assessment.metadata.get("target_formula_version") != AssesmentConformPPmetricsBased.FORMULA_VERSION:
            raise ValueError("Outdated metric formula in text model; recalculate DB scores and retrain modules 3/4")
        return assessment
