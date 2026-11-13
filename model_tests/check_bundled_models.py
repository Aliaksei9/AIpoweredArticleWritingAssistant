"""Exercise the saved model pair with a real cached embedding, entirely offline.

The fixture is public test text embedded once with the configured SBERT model.
This checks artifact loading and the text-to-score-to-metrics interface, not
prediction accuracy. No database connection or SBERT initialization is needed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from threadpoolctl import threadpool_limits

from src.assesment_conform_of_PP_text_based import AssesmentConformPPtextBased
from src.database.embedding_config import EMBEDDING_SIGNATURE
from src.database.vector_utils import validate_vector
from src.social_metrics_prediction import SocialMetricsPredictor
from src.social_metrics_prediction.features import METRICS
from src.social_metrics_prediction.forecast_context import forecast_context


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = ROOT / "model_tests/fixtures/bundled_model_input.json"


def file_sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_fixture(path):
    """Explicit maintenance command; regular CI never enters this branch."""
    from src.database.model_embedding import TextEmbedder

    case = json.loads(path.read_text(encoding="utf-8"))
    case["document_embedding"] = TextEmbedder().embed_document(case["text"])
    case["embedding_signature"] = EMBEDDING_SIGNATURE
    case["text_sha256"] = hashlib.sha256(case["text"].encode("utf-8")).hexdigest()
    path.write_text(json.dumps(case, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def check_bundle(root=ROOT, fixture=DEFAULT_FIXTURE):
    root, fixture = Path(root), Path(fixture)
    manifest = json.loads((root / "artifacts/model_bundle.json").read_text(encoding="utf-8"))
    paths = {}
    for key in ("text_model", "social_model"):
        entry = manifest[key]
        path = root / entry["path"]
        if file_sha256(path) != entry["sha256"]:
            raise ValueError(f"Bundled {key} does not match model_bundle.json")
        paths[key] = path

    case = json.loads(fixture.read_text(encoding="utf-8"))
    if case.get("embedding_signature") != EMBEDDING_SIGNATURE:
        raise ValueError("Cached test embedding is incompatible; regenerate it with --prepare-fixture")
    if case.get("text_sha256") != hashlib.sha256(case["text"].encode("utf-8")).hexdigest():
        raise ValueError("Fixture text changed; regenerate its cached embedding with --prepare-fixture")
    vector = validate_vector(case["document_embedding"])
    text = AssesmentConformPPtextBased.load(paths["text_model"])
    social = SocialMetricsPredictor.load(paths["social_model"])
    if social.metadata.get("text_model_sha256") != manifest["text_model"]["sha256"]:
        raise ValueError("Bundled text/social models are not a compatible pair")
    if not social.uses_extended_features:
        raise ValueError("Bundled social model does not use the shared complete inputs")
    if case["topic"] not in text.encoder.topic_embeddings:
        raise ValueError("The fixture topic has no cached embedding in the text model")
    known_resources = set(social.encoder_.categories_.categories_[1])
    resource = next((r for r in social.resources
                     if r.id in known_resources and r.source in social.encoder_.sources_), None)
    if resource is None:
        raise ValueError("Bundled model has no supported resource in its saved catalog")

    features = text.encoder.transform([case["text"]], [case["topic"]], [vector])
    if text.encoder._embedder is not None:
        raise RuntimeError("Offline smoke check unexpectedly initialized SBERT")
    score = float(text.model.predict(features)[0])
    if not math.isfinite(score) or not -10 <= score <= 10:
        raise ValueError("Bundled text model returned an invalid score")
    prediction = social.predict(case["topic"], score, resource, resource.source,
        document_embedding=vector, published_at=case["published_at"]).to_dict()
    if set(prediction) != set(METRICS) or any(type(value) is not int or value < 0 for value in prediction.values()):
        raise ValueError("Bundled social model returned invalid metric counts")
    context = forecast_context(case["published_at"])
    if context["horizon_days"] != manifest["prediction_policy"]["horizon_days"]:
        raise ValueError("Forecast period differs from the published model bundle")
    return dict(status="passed", model_pair_verified=True, cached_embedding=True,
                horizon_days=context["horizon_days"], score=score, metrics=prediction)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--prepare-fixture", action="store_true",
                        help="Explicitly recompute the cached SBERT vector after an embedding-profile change")
    args = parser.parse_args()
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        if args.prepare_fixture:
            prepare_fixture(args.fixture)
            print(json.dumps(dict(status="fixture_updated", fixture=str(args.fixture))))
        else:
            print(json.dumps(check_bundle(args.root, args.fixture), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
