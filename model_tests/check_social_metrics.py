"""Verify the production bundle against the selected experiment and split audit."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from src.social_metrics_prediction import SocialMetricsPredictor
from src.assesment_conform_of_PP_text_based import AssesmentConformPPtextBased
from .benchmark_utils import summarize
from .text_candidates import ExperimentalTextAssessment


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    directory = args.run
    report = read_json(directory / "report.json")
    splits = read_json(directory / "splits.json")
    inputs = read_json(directory / "inputs.json")
    with np.load(directory / "dataset.npz") as snapshot:
        groups, counts = snapshot["groups"], snapshot["counts"]
        ids = snapshot["article_ids"].tolist()
    test = set(splits["test_indices"])
    def audit(train, validation, crossfit):
        train, validation = set(train), set(validation)
        assert not train & validation
        assert not set(groups[list(train)]) & set(groups[list(validation)])
        visited = []
        for fold in crossfit:
            a, b = set(fold["train_indices"]), set(fold["validation_indices"])
            assert a | b == train and not a & b
            assert not set(groups[list(a)]) & set(groups[list(b)])
            assert not (a | b) & validation
            visited += list(b)
        assert set(visited) == train and len(visited) == len(train)
    audit(splits["train_indices"], splits["test_indices"], splits["upstream_crossfit"])
    for fold in splits["cv"]:
        assert not test & (set(fold["train_indices"]) | set(fold["validation_indices"]))
        audit(fold["train_indices"], fold["validation_indices"], fold["upstream_crossfit"])

    train = np.asarray(splits["train_indices"])
    rows = inputs["heldout"]["train"]
    validation = inputs["heldout"]["test"]
    with threadpool_limits(limits=4):
        evaluation = SocialMetricsPredictor(seed=report["seed"], threads=4).fit(rows, counts[train])
        predicted = evaluation.predict_rows(validation)
    with np.load(directory / "test_predictions.npz") as values:
        difference = float(np.max(np.abs(predicted-values[report["selected_model"]])))
    assert difference == 0, f"Production implementation differs from chosen candidate: {difference}"

    model_path = Path("artifacts/social_metrics/model.joblib")
    text_path = Path("artifacts/text_models/neural_network.joblib")
    manifest = read_json(model_path.with_suffix(".training.json"))
    social = SocialMetricsPredictor.load(model_path)
    text = AssesmentConformPPtextBased.load(text_path)
    assert manifest["article_ids"] == ids
    assert social.metadata["training_rows"] == len(ids) == text.metadata["training_rows"]
    assert social.metadata["dataset_fingerprint"] == report["dataset_fingerprint"]
    assert hashlib.sha256(text_path.read_bytes()).hexdigest() == social.metadata["text_model_sha256"]
    audit(range(len(ids)), [], manifest["crossfit"])
    expected = social.predict("Игры", 2.5, "@Jeteed", "Telegram")
    assert all(isinstance(v, int) and v >= 0 for v in expected.to_dict().values())
    try:
        social.predict("Игры", 2.5, "@Jeteed", "VK")
    except ValueError as error:
        assert "Source has no training data" in str(error)
    else:
        raise AssertionError("An untrained platform was accepted")
    durations = []
    for _ in range(21):
        started = time.perf_counter()
        assert social.predict("Игры", 2.5, "@Jeteed", "TG") == expected
        durations.append(time.perf_counter()-started)
    old_directory = Path("model_tests/results/20261021_141831_UTC/evaluation_models")
    legacy = []
    if old_directory.exists():
        for path in old_directory.glob("*.joblib"):
            loaded = ExperimentalTextAssessment.load(path)
            assert loaded.model.name == path.stem
            legacy.append(path.name)
    verification = dict(passed=True, training_rows=len(ids), resources=len(social.resources),
        split_audit_passed=True, production_matches_selected_candidate=True,
        maximum_prediction_difference=difference, legacy_artifacts_loaded=legacy,
        sample=dict(topic="Игры", score=2.5, resource="Jeteed", **expected.to_dict()),
        production_single_prediction_seconds=summarize(durations))
    (directory / "production_verification.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(verification, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
