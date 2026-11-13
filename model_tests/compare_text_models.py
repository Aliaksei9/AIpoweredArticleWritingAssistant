"""Compare four popularity regressors by MSE and CPU training/prediction time.

Run from the project root: python -m model_tests.compare_text_models
All experiment code, models, data snapshots and graphs stay outside src.
"""

import argparse
import asyncio
import csv
import importlib.metadata
import json
import os
import platform
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from src.assesment_conform_of_PP_metric_based import AssesmentConformPPmetricsBased
from src.assesment_conform_of_PP_text_based import AssesmentConformPPtextBased, TextTopicEncoder
from src.assesment_conform_of_PP_text_based.dataset import load_training_dataset
from .text_candidates import MODEL_CLASSES, ExperimentalTextAssessment
from src.database.connection import initialize_database
from src.database.model_embedding import MODEL_NAME

from .benchmark_utils import metrics, prediction_timing, summarize, representative_cases, response_timing
from .data_split import split_dataset
from .model_selection import select_model


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def make_assessment(name, encoder, seed, threads):
    parameters = {"random_forest": {"n_jobs": threads},
                  "catboost": {"thread_count": threads},
                  "neural_network": {"num_threads": threads}}.get(name, {})
    return ExperimentalTextAssessment(name, encoder, seed, **parameters)


def write_summary(output, report):
    fields = ["model", "cv_mse_mean", "cv_mse_std", "test_mse", "test_rmse", "test_mae", "test_r2",
              "fit_seconds", "cv_fit_mean_seconds", "batch_prediction_median_ms",
              "batch_prediction_median_ms_per_article", "single_prediction_median_ms", "response_median_seconds", "response_p95_seconds", "response_max_seconds"]
    with (output / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        for name, result in report["results"].items():
            writer.writerow([name, result["cv_mse"]["mean"], result["cv_mse"]["std"],
                             *[result["test"][key] for key in ["mse", "rmse", "mae", "r2"]],
                             result["fit_seconds"],
                             None if result["cv_fit_seconds"] is None else result["cv_fit_seconds"]["mean"],
                             result["batch_prediction"]["median"] * 1000,
                             result["batch_prediction"]["median_ms_per_article"],
                             result["single_prediction"]["median"] * 1000,
                             result["response_time"]["median"], result["response_time"]["p95"], result["response_time"]["max"]])
    lines = ["# Результаты сравнения", "",
             f"Статей: {report['dataset']['training_rows']}; обучение: {report['train_rows']}; "
             f"отложенная проверка: {report['test_rows']}. CPU, потоков: {report['threads']}.", "",
             "| Модель | MSE CV | MSE проверки | Ответ с E5, медиана, с | p95, с | Максимум, с |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, result in report["results"].items():
        lines.append(f"| {name} | {result['cv_mse']['mean']:.4f} | {result['test']['mse']:.4f} "
                     f"| {result['response_time']['median']:.4f} | {result['response_time']['p95']:.4f} | {result['response_time']['max']:.4f} |")
    lines += ["", f"Наиболее точный допустимый регрессор по MSE CV: **{report['selected_model']}**.",
              f"Лучший с учётом среднего прогноза: **{report['best_including_baseline']}**.", "",
              f"Предел максимального измеренного полного ответа: {report['max_response_seconds']:g} с. "
              "Ответ включает векторизацию исходного текста E5 и регрессор после прогрева; "
              "сеть, БД и загрузка весов исключены. Время обучения не участвует в выборе; "
              "оно сохраняется только в техническом протоколе.",
              "Показатели MSE проверки получены до какого-либо обучения на этой части данных.", "",
              f"Общая подготовка признаков из БД: {report['preprocessing']['feature_seconds']:.3f} с.",
              f"Векторизация {report['preprocessing']['raw_sample_rows']} полных текстов "
              f"с прогретой SBERT: {report['preprocessing']['raw_sample_seconds']:.3f} с.", "",
              "Данные относятся к имеющимся ресурсам и оценкам формулы метрик. "
              "Разделение случайное по группам одинаковых текстов; оно не проверяет прогноз будущих дат."]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


async def run():
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--refresh-scores", action="store_true")
    parser.add_argument("--reembed", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--test-size", type=float, default=.2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--timing-repeats", type=int, default=9)
    parser.add_argument("--single-repeats", type=int, default=21)
    parser.add_argument("--max-response-seconds", type=float, default=10.0)
    args = parser.parse_args()
    if (args.folds < 2 or args.threads < 1 or args.timing_repeats < 1 or args.single_repeats < 1
            or not 0 < args.test_size < 1 or not np.isfinite(args.max_response_seconds) or args.max_response_seconds <= 0):
        parser.error("Positive counts, at least two folds and test-size in (0, 1) are required")
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise SystemExit("Set DATABASE_URL in .env")
    output = args.output or root / "model_tests" / "results" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")
    if (output / "report.json").exists():
        raise SystemExit("This output already contains a report; choose another --output to preserve results")
    output.mkdir(parents=True, exist_ok=True)
    if args.refresh_scores:
        print(f"Reassessed {await initialize_database(dsn)} articles", flush=True)
    started = time.perf_counter()
    dataset = await load_training_dataset(dsn)
    database_seconds = time.perf_counter() - started
    train, test = split_dataset(dataset, args.test_size, args.seed)
    if len(set(dataset.groups[train])) < args.folds or len(test) < 2:
        raise ValueError("Too few distinct training groups or test rows for the requested comparison")
    folds = list(GroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed).split(
        train, groups=dataset.groups[train]))
    for local_train, local_validation in folds:
        assert not set(dataset.groups[train[local_train]]) & set(dataset.groups[train[local_validation]])
    assert not set(dataset.groups[train]) & set(dataset.groups[test])
    print(f"Data: {dataset.report}; train={len(train)}, test={len(test)}; output={output}", flush=True)
    encoder = TextTopicEncoder()
    previous_threads = torch.get_num_threads()
    benchmark_started = time.perf_counter()
    try:
        torch.set_num_threads(args.threads)
        with threadpool_limits(limits=args.threads):
            started = time.perf_counter()
            features = encoder.transform(dataset.texts, dataset.topics,
                                         None if args.reembed else dataset.embeddings)
            feature_seconds = time.perf_counter() - started
            raw_sample = test[:min(8, len(test))]
            started = time.perf_counter()
            raw_features = encoder.transform([dataset.texts[i] for i in raw_sample],
                                             [dataset.topics[i] for i in raw_sample])
            raw_seconds = time.perf_counter() - started
            feature_difference = float(np.max(np.abs(raw_features - features[raw_sample])))
            np.savez_compressed(output / "dataset.npz", features=features, scores=dataset.scores,
                                article_ids=dataset.article_ids, groups=dataset.groups, topics=dataset.topics)
            write_json(output / "splits.json", dict(seed=args.seed, train_indices=train.tolist(),
                       test_indices=test.tolist(), cv_folds=[dict(train_indices=train[a].tolist(),
                       validation_indices=train[b].tolist()) for a, b in folds]))
            report = dict(status="running", created_at=datetime.now(timezone.utc).isoformat(),
                          dataset=dataset.report, dataset_fingerprint=dataset.fingerprint,
                          train_rows=len(train), test_rows=len(test), seed=args.seed, folds=args.folds,
                          threads=args.threads, feature_dimensions=features.shape[1],
                          feature_model=MODEL_NAME, split="groups of normalized identical texts",
                          target_formula_version=AssesmentConformPPmetricsBased.FORMULA_VERSION,
                          target_formula_parameters=asdict(AssesmentConformPPmetricsBased()),
                          preprocessing=dict(database_seconds=database_seconds, feature_seconds=feature_seconds,
                                             raw_sample_rows=len(raw_sample), raw_sample_seconds=raw_seconds,
                                             raw_vs_cached_max_difference=feature_difference),
                          runtime=dict(python=platform.python_version(), platform=platform.platform(),
                                       processor=platform.processor(), logical_cpu_count=os.cpu_count(), device="CPU"),
                          package_versions={name: importlib.metadata.version(name) for name in
                                            ["scikit-learn", "catboost", "torch", "sentence-transformers", "numpy"]},
                          timing_method="perf_counter wall time; models sequential; warmed predictions; SBERT excluded",
                          selection_rule="lowest grouped CV MSE with measured maximum warm end-to-end response within limit; fit time ignored",
                          max_response_seconds=args.max_response_seconds,
                          results={}, limitations=["Targets are scores of the module 2 heuristic formula",
                                                   "Random text-group split; not a future-time or unseen-channel test",
                                                   "Near-duplicate paraphrases can cross text groups"])
            predictions = {}
            history = None
            latency_cases = representative_cases(dataset.texts, train)
            report["response_case_lengths"] = [len(dataset.texts[i]) for i in latency_cases]
            for name in MODEL_CLASSES:
                fold_results = []
                for index, (local_train, local_validation) in enumerate(folds, start=1):
                    print(f"{name}: CV {index}/{args.folds}", flush=True)
                    fit_indices, validation_indices = train[local_train], train[local_validation]
                    assessment = make_assessment(name, encoder, args.seed, args.threads)
                    X_fit, y_fit = features[fit_indices], dataset.scores[fit_indices]
                    X_validation = features[validation_indices]
                    started = time.perf_counter()
                    assessment.train_features(X_fit, y_fit)
                    fit_seconds = time.perf_counter() - started
                    started = time.perf_counter()
                    predicted = assessment.model.predict(X_validation)
                    predict_seconds = time.perf_counter() - started
                    fold_results.append(dict(**metrics(dataset.scores[validation_indices], predicted),
                                             fit_seconds=fit_seconds, predict_seconds=predict_seconds))
                print(f"{name}: train on {len(train)} rows and evaluate held-out test", flush=True)
                assessment = make_assessment(name, encoder, args.seed, args.threads)
                X_fit, y_fit = features[train], dataset.scores[train]
                X_test = features[test]
                started = time.perf_counter()
                assessment.train_features(X_fit, y_fit)
                fit_seconds = time.perf_counter() - started
                predictions[name] = assessment.model.predict(X_test)
                result = dict(cv_folds=fold_results,
                              cv_mse=summarize([row["mse"] for row in fold_results]),
                              cv_fit_seconds=summarize([row["fit_seconds"] for row in fold_results]),
                              train=metrics(y_fit, assessment.model.predict(X_fit)),
                              test=metrics(dataset.scores[test], predictions[name]), fit_seconds=fit_seconds,
                              batch_prediction=prediction_timing(assessment.model, X_test, args.timing_repeats),
                              single_prediction=prediction_timing(assessment.model, X_test[:1], args.single_repeats),
                              parameters=assessment.model.estimator.get_params())
                result["response_time"] = response_timing(
                    lambda i: assessment.assess(dataset.texts[i], dataset.topics[i]), latency_cases)
                assessment.metadata.update(dataset_fingerprint=dataset.fingerprint,
                                           purpose="evaluation; held-out test excluded")
                assessment.save(output / "evaluation_models" / f"{name}.joblib")
                if name == "neural_network":
                    history = assessment.model.estimator.history_
                    write_json(output / "neural_network_history.json", history)
                report["results"][name] = result
                write_json(output / "report.json", report)
                print(f"{name}: CV MSE={result['cv_mse']['mean']:.4f}; "
                      f"test MSE={result['test']['mse']:.4f}; fit={fit_seconds:.3f}s", flush=True)
            # The baseline is an accuracy reference, not a candidate architecture.
            baseline_folds = []
            for a, b in folds:
                predicted = np.full(len(b), np.mean(dataset.scores[train[a]]))
                baseline_folds.append(metrics(dataset.scores[train[b]], predicted))
            from sklearn.dummy import DummyRegressor
            baseline = DummyRegressor(strategy="mean")
            X_fit, y_fit, X_test = features[train], dataset.scores[train], features[test]
            started = time.perf_counter()
            baseline.fit(X_fit, y_fit)
            fit_seconds = time.perf_counter() - started
            predictions["mean_baseline"] = baseline.predict(X_test)
            report["results"]["mean_baseline"] = dict(cv_folds=baseline_folds,
                cv_mse=summarize([row["mse"] for row in baseline_folds]),
                cv_fit_seconds=None, fit_seconds=fit_seconds,
                train=metrics(y_fit, baseline.predict(X_fit)),
                test=metrics(dataset.scores[test], predictions["mean_baseline"]),
                batch_prediction=prediction_timing(baseline, X_test, args.timing_repeats),
                single_prediction=prediction_timing(baseline, X_test[:1], args.single_repeats))
            report["results"]["mean_baseline"]["response_time"] = response_timing(
                lambda i: baseline.predict(encoder.transform([dataset.texts[i]], [dataset.topics[i]])), latency_cases)
            report["selected_model"] = select_model(report["results"], MODEL_CLASSES, "cv_mse", args.max_response_seconds)
            report["best_including_baseline"] = select_model(report["results"], report["results"], "cv_mse", args.max_response_seconds)
            with (output / "test_predictions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["article_id", "topic", "actual", *predictions])
                for index, article_index in enumerate(test):
                    writer.writerow([dataset.article_ids[article_index], dataset.topics[article_index],
                                     dataset.scores[article_index], *[values[index] for values in predictions.values()]])
            from .plots import create_plots
            create_plots(output, report, dataset.scores[test], predictions, history)
            report.update(status="complete", completed_at=datetime.now(timezone.utc).isoformat(),
                          benchmark_seconds=time.perf_counter() - benchmark_started)
            write_json(output / "report.json", report)
            write_summary(output, report)
            print(f"Selected by CV MSE: {report['selected_model']}", flush=True)
            print(f"Reports and graphs: {output.resolve()}", flush=True)
    finally:
        torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop if sys.platform == "win32" else None) as runner:
        runner.run(run())
