"""Check the DB labels, retrained bundle, cross-fitting and local prediction API."""

import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import sys

from dotenv import load_dotenv
import httpx

from src.assesment_conform_of_PP_metric_based import AssesmentConformPPmetricsBased
from src.assesment_conform_of_PP_text_based import AssesmentConformPPtextBased
from src.social_metrics_prediction import SocialMetricsPredictor
from src.social_metrics_prediction.dataset import load_metric_dataset


async def run():
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root/".env")
    output = root/"model_tests/results/metric_formula_v3_20261023"
    dataset = await load_metric_dataset(os.environ["DATABASE_URL"])
    text_path, social_path = root/"artifacts/text_models/neural_network.joblib", root/"artifacts/social_metrics/model.joblib"
    text = AssesmentConformPPtextBased.load(text_path)
    social = SocialMetricsPredictor.load(social_path)
    manifest = json.loads(social_path.with_suffix(".training.json").read_text(encoding="utf-8"))
    assert text.metadata["target_formula_version"] == AssesmentConformPPmetricsBased.FORMULA_VERSION
    assert text.metadata["dataset_fingerprint"] == dataset.text.fingerprint
    assert social.metadata["dataset_fingerprint"] == dataset.fingerprint == manifest["dataset_fingerprint"]
    assert social.metadata["text_model_sha256"] == hashlib.sha256(text_path.read_bytes()).hexdigest()
    assert manifest["article_ids"] == dataset.text.article_ids
    visited = []
    for fold in manifest["crossfit"]:
        a, b = fold["train_indices"], fold["validation_indices"]
        assert not set(a) & set(b)
        assert not set(dataset.groups[a]) & set(dataset.groups[b])
        assert set(a) | set(b) == set(range(len(dataset.scores)))
        visited.extend(b)
    assert sorted(visited) == list(range(len(dataset.scores)))
    previous = json.loads((output/"previous_scores.json").read_text(encoding="utf-8"))
    assert {r["article_id"] for r in previous} == set(dataset.text.article_ids)
    resource = next(r for r in social.resources if r.name.casefold() == "jeteed")
    async with httpx.AsyncClient(base_url="http://127.0.0.1:8000", timeout=120) as client:
        workspace = await client.get("/api/workspace")
        workspace.raise_for_status()
        response = await client.post("/api/predictions", json=dict(text=(root/"model_tests/example_edgerunners.txt").read_text(encoding="utf-8"),
            topic="Аниме", resource_id=resource.id, source="TG"))
        response.raise_for_status()
        result = response.json()
    assert math.isfinite(result["score"]) and -10 <= result["score"] <= 10
    assert all(type(v) is int and v >= 0 for v in result["metrics"].values())
    report = dict(passed=True, formula=text.metadata["target_formula_version"], rows=len(dataset.scores),
        all_database_labels_match_formula=True, article_ids_preserved=True, crossfit_groups_disjoint=True,
        matching_model_hash=True, text_training_rows=text.metadata["training_rows"], social_training_rows=social.metadata["training_rows"],
        formula_examples={"zero_views": AssesmentConformPPmetricsBased().assess(0, 80, 5, 100000),
            "reference": AssesmentConformPPmetricsBased().assess(10000, 80, 5, 100000),
            "no_comments_strong_engagement": AssesmentConformPPmetricsBased().assess(100000, 800, 0, 100000)},
        prediction=result)
    (output/"production_verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop if sys.platform == "win32" else None) as runner:
        runner.run(run())
