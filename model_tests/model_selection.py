"""Choose accuracy under a response-time constraint; never use fit duration."""
import math


def select_model(results, names, error_key, max_response_seconds=10.0):
    if not math.isfinite(max_response_seconds) or max_response_seconds <= 0:
        raise ValueError("Response-time limit must be positive and finite")
    eligible = []
    for name in names:
        row = results[name]
        error = float(row[error_key]["mean"])
        response = float(row["response_time"].get("max", row["response_time"]["p95"]))
        if not math.isfinite(error) or error < 0 or not math.isfinite(response) or response < 0:
            raise ValueError("Selection needs finite errors and response times")
        if response <= max_response_seconds:
            eligible.append(name)
    if not eligible:
        raise ValueError("No model satisfies the end-to-end response-time limit")
    selected = min(eligible, key=lambda name: (results[name][error_key]["mean"], results[name]["response_time"]["p95"]))
    return selected
