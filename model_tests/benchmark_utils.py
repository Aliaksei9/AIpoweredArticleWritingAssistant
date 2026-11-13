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


def representative_cases(texts, indices, maximum=8):
    """Training inputs spanning text lengths, without consulting target labels."""
    ordered = sorted((int(i) for i in indices), key=lambda i: (len(texts[i]), i))
    if not ordered:
        raise ValueError("Response timing needs training inputs")
    positions = np.linspace(0, len(ordered)-1, min(maximum, len(ordered))).round().astype(int)
    return [ordered[int(i)] for i in positions]


def response_timing(predict_case, cases):
    """Warm end-to-end wall time including raw-text vectorization, excluding I/O."""
    if not cases:
        raise ValueError("Response timing needs cases")
    predict_case(cases[0])
    durations = []
    for case in cases:
        started = time.perf_counter()
        predict_case(case)
        durations.append(time.perf_counter()-started)
    result = summarize(durations)
    result.update(max=float(max(durations)), cases=len(cases), scope="raw text to prediction; warmed models; no database/network")
    return result
