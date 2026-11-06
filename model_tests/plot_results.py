"""Regenerate figures from a completed comparison without retraining."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .plots import create_plots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    directory = args.run_directory
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    if report.get("status") != "complete":
        raise SystemExit("The comparison has not completed")
    with (directory / "test_predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    actual = np.asarray([float(row["actual"]) for row in rows])
    predictions = {name: np.asarray([float(row[name]) for row in rows]) for name in report["results"]}
    history_path = directory / "neural_network_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else None
    create_plots(directory, report, actual, predictions, history)
    print(f"Graphs saved in {directory.resolve()}")


if __name__ == "__main__":
    main()
