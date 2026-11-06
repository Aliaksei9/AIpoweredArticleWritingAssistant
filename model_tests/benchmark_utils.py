"""Measurements used only by the external model comparison."""

import time

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def metrics(actual, predicted):
    mse = float(mean_squared_error(actual, predicted))
    return dict(mse=mse, rmse=float(np.sqrt(mse)),
                mae=float(mean_absolute_error(actual, predicted)),
                r2=float(r2_score(actual, predicted)))


def summarize(values):
    values = np.asarray(values, dtype=float)
    return dict(mean=float(np.mean(values)), median=float(np.median(values)),
                std=float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                p95=float(np.quantile(values, .95)), samples=values.tolist())


def prediction_timing(model, features, repeats):
    """Warm prediction time, excluding SBERT and model loading."""
    if repeats < 1 or len(features) < 1:
        raise ValueError("Prediction timing requires rows and positive repeats")
    model.predict(features)
    durations = []
    for _ in range(repeats):
        started = time.perf_counter()
        prediction = model.predict(features)
        durations.append(time.perf_counter() - started)
        if len(prediction) != len(features):
            raise ValueError("Prediction count does not match input count")
    result = summarize(durations)
    result.update(rows=len(features), repeats=repeats,
                  median_ms_per_article=result["median"] * 1000 / len(features))
    return result
