"""Nested upstream cross-fitting, model-family selection and untouched test evaluation."""

import argparse
import asyncio
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch
from dotenv import load_dotenv
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from src.assesment_conform_of_PP_text_based import TextTopicEncoder
from src.social_metrics_prediction.dataset import load_metric_dataset
from src.social_metrics_prediction.features import METRICS
from src.social_metrics_prediction.training import cross_fitted_scores, train_text_head

from .benchmark_utils import metrics, summarize, representative_cases, response_timing
from .model_selection import select_model
from .compare_text_models import write_json
from .data_split import split_dataset
from .social_metric_candidates import MODEL_NAMES, CandidateSocialMetricsPredictor, ResourceMeanBaseline
from .social_metric_plots import create_plots


def evaluate(actual, predicted):
    return {metric: metrics(actual[:, i], predicted[:, i]) for i, metric in enumerate(METRICS)}


def normalized_mse(actual, predicted, train_counts):
    # Per-target variance is fitted on this training fold only. Views cannot
    # dominate selection simply because their numerical scale is larger.
    scale = np.maximum(np.var(train_counts, axis=0), 1.0)
    return float(np.mean(np.mean((actual-predicted)**2, axis=0) / scale))


def timed_prediction(model, rows, repeats):
    model.predict_rows(rows)
    durations = []
    for _ in range(repeats):
        started = time.perf_counter()
        model.predict_rows(rows)
        durations.append(time.perf_counter()-started)
    return summarize(durations)


def progress(message):
    print(message, flush=True)


def prepare_fold(dataset, features, encoder, train, validation, args, title):
    progress(f"{title}: preparing independent module-3 scores")
    started = time.perf_counter()
    scores, audit = cross_fitted_scores(features[train], dataset.scores[train],
        dataset.groups[train], folds=args.folds, seed=args.seed, threads=args.threads, on_progress=progress)
    head = train_text_head(encoder, features[train], dataset.scores[train],
                          seed=args.seed, threads=args.threads)
    validation_scores = head.model.predict(features[validation])
    audit = [dict(train_indices=train[a["train_indices"]].tolist(),
                  validation_indices=train[a["validation_indices"]].tolist()) for a in audit]
    return dict(train_rows=dataset.inputs(train, scores),
                validation_rows=dataset.inputs(validation, validation_scores),
                train=train, validation=validation, audit=audit,
                upstream_seconds=time.perf_counter()-started,
                upstream_validation=metrics(dataset.scores[validation], validation_scores), upstream_head=head)


def write_summary(output, report):
    lines = ["# Прогнозирование социальных метрик", "",
        f"Статей: {report['dataset']['training_rows']}; обучение {report['train_rows']}; проверка {report['test_rows']}.",
        "", "Оценка популярности во входе — прогноз нейросети по тексту и теме. "
        "Для каждой строки обучения он получен без её группы текстов. "
        "Для CV эта процедура повторяется внутри каждой обучающей части.", "",
        "| Модель | CV, норм. MSE | MSE просмотров | MSE реакций | MSE комментариев | Ответ с E5, медиана, с | p95, с | Максимум, с |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, r in report["results"].items():
        mse = [r["test"][m]["mse"] for m in METRICS]
        lines.append(f"| {name} | {r['cv_selection']['mean']:.4f} | {mse[0]:.2f} | {mse[1]:.2f} | {mse[2]:.2f} "
                     f"| {r['response_time']['median']:.4f} | {r['response_time']['p95']:.4f} | {r['response_time']['max']:.4f} |")
    lines += ["", f"Выбрана семья моделей: **{report['selected_model']}**.",
        f"Лучший результат вместе с базовым прогнозом: **{report['best_including_baseline']}**.", "",
        "Критерий точности: среднее по трём метрикам MSE, делённое на дисперсию соответствующей "
        "метрики обучения, затем среднее по CV-разбиениям. Проверка не участвует в выборе. "
        f"Допускаются модели с максимальным измеренным ответом не более {report['max_response_seconds']:g} с. "
        "Время обучения не учитывается при выборе.",
        "MSE в таблице рассчитан на исходных целочисленных прогнозах; RMSE, MAE и R² есть в metrics.csv.",
        "Полный ответ включает E5, прогноз оценки текстовой нейросетью, кодирование и все три регрессора; "
        "измеряется после загрузки и прогрева, без сети и БД. Технические времена обучения сохранены отдельно.", "",
        "Случайное разделение по точным нормализованным текстам проверяет имеющиеся каналы. "
        "Это не проверка будущих дат или новых платформ. Метрики измерены у статей возраста 2–14 дней; "
        "модель не обещает результат через фиксированное число дней."]
    (output / "summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    with (output / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model", "metric", "cv_mse_mean", "test_mse", "test_rmse", "test_mae", "test_r2",
                         "fit_seconds", "batch_prediction_ms", "single_prediction_ms", "response_median_seconds", "response_p95_seconds", "response_max_seconds"])
        for name, r in report["results"].items():
            for metric in METRICS:
                writer.writerow([name, metric, r["cv_metrics"][metric]["mse"]["mean"],
                    *[r["test"][metric][key] for key in ("mse", "rmse", "mae", "r2")],
                    r["fit_seconds"]["median"], r["batch_predict_seconds"]["median"]*1000,
                    r["single_predict_seconds"]["median"]*1000, r["response_time"]["median"], r["response_time"]["p95"], r["response_time"]["max"]])


async def run():
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--test-size", type=float, default=.2)
    parser.add_argument("--prepared-run", type=Path, help="Reuse upstream cross-fit scores from an identical dataset and split")
    parser.add_argument("--max-response-seconds", type=float, default=10.0)
    args = parser.parse_args()
    if args.folds < 2 or args.threads < 1 or not np.isfinite(args.max_response_seconds) or args.max_response_seconds <= 0:
        parser.error("At least two folds and positive threads required")
    output = args.output or root / "model_tests/results" / datetime.now(timezone.utc).strftime("social_metrics_%Y%m%d_%H%M%S_UTC")
    if (output / "report.json").exists():
        raise ValueError("Report already exists; use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    dataset = await load_metric_dataset(os.environ["DATABASE_URL"])
    train, test = split_dataset(dataset, args.test_size, args.seed)
    progress(f"Module 4: {len(dataset.scores)} articles; {len(train)} training; {len(test)} test; output={output}")
    encoder = TextTopicEncoder()
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(args.threads)
        with threadpool_limits(limits=args.threads):
            if args.prepared_run:
                cached = json.loads((args.prepared_run / "report.json").read_text(encoding="utf-8"))
                splits = json.loads((args.prepared_run / "splits.json").read_text(encoding="utf-8"))
                inputs = json.loads((args.prepared_run / "inputs.json").read_text(encoding="utf-8"))
                if (cached["dataset_fingerprint"] != dataset.fingerprint or cached["seed"] != args.seed
                        or cached["folds"] != args.folds or splits["train_indices"] != train.tolist()
                        or splits["test_indices"] != test.tolist()):
                    raise ValueError("Prepared run does not match the current data and split")
                with np.load(args.prepared_run / "dataset.npz") as snapshot:
                    features = snapshot["text_features"]
                feature_seconds = cached["feature_seconds"]
                cv = [dict(train=np.asarray(s["train_indices"]), validation=np.asarray(s["validation_indices"]),
                    train_rows=inp["train"], validation_rows=inp["validation"], audit=s["upstream_crossfit"],
                    upstream_seconds=cached["upstream_seconds"][i], upstream_validation=cached["upstream_validation"][i])
                    for i, (s, inp) in enumerate(zip(splits["cv"], inputs["cv"]))]
                heldout = dict(train=train, validation=test, train_rows=inputs["heldout"]["train"],
                    validation_rows=inputs["heldout"]["test"], audit=splits["upstream_crossfit"],
                    upstream_seconds=cached["upstream_seconds"][-1], upstream_validation=cached["upstream_validation"][-1])
                progress(f"Reused independent module-3 scores from {args.prepared_run}")
            else:
                started = time.perf_counter()
                features = encoder.transform(dataset.text.texts, dataset.text.topics, dataset.text.embeddings)
                feature_seconds = time.perf_counter()-started
                splitter = GroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
                cv = []
                for i, (a, b) in enumerate(splitter.split(train, groups=dataset.groups[train]), 1):
                    cv.append(prepare_fold(dataset, features, encoder, train[a], train[b], args, f"CV {i}/{args.folds}"))
                heldout = prepare_fold(dataset, features, encoder, train, test, args, "Held-out test")
            np.savez_compressed(output / "dataset.npz", article_ids=dataset.text.article_ids,
                topics=dataset.text.topics, groups=dataset.groups, counts=dataset.counts,
                text_features=features, metric_scores=dataset.scores)
            split_audit = dict(train_indices=train.tolist(), test_indices=test.tolist(),
                cv=[dict(train_indices=f["train"].tolist(), validation_indices=f["validation"].tolist(),
                         upstream_crossfit=f["audit"]) for f in cv], upstream_crossfit=heldout["audit"])
            write_json(output / "splits.json", split_audit)
            write_json(output / "inputs.json", dict(cv=[dict(train=f["train_rows"], validation=f["validation_rows"]) for f in cv],
                heldout=dict(train=heldout["train_rows"], test=heldout["validation_rows"])))
            report = dict(status="running", dataset=dataset.text.report, dataset_fingerprint=dataset.fingerprint,
                created_at=datetime.now(timezone.utc).isoformat(), train_rows=len(train), test_rows=len(test),
                folds=args.folds, seed=args.seed, threads=args.threads,
                selection_rule="lowest grouped CV normalized MSE with measured maximum warm end-to-end response within limit; fit time ignored",
                max_response_seconds=args.max_response_seconds,
                score_input="nested cross-fitted module-3 neural-network predictions; module-2 labels excluded from inputs",
                target_transform="counts per 1000 subscribers; loss weighted by audience squared to optimize count MSE",
                prepared_run=None if args.prepared_run is None else str(args.prepared_run.resolve()),
                feature_seconds=feature_seconds, upstream_seconds=[f["upstream_seconds"] for f in [*cv, heldout]],
                upstream_validation=[f["upstream_validation"] for f in [*cv, heldout]],
                versions={p: importlib.metadata.version(p) for p in ["numpy", "scikit-learn", "catboost", "torch"]},
                limitations=["Random text-group split; not future-time or unseen-resource evaluation",
                             "Counts observed at publication ages 2-14 days, not a fixed forecast horizon",
                             "Only Telegram is represented", "Near duplicate paraphrases can cross splits"], results={})
            predictions = {}
            latency_cases = representative_cases(dataset.text.texts, train)
            report["response_case_lengths"] = [len(dataset.text.texts[i]) for i in latency_cases]
            if "upstream_head" not in heldout:
                heldout["upstream_head"] = train_text_head(encoder, features[train], dataset.scores[train], seed=args.seed, threads=args.threads)
            def predict_case(model, index):
                score = heldout["upstream_head"].assess(dataset.text.texts[index], dataset.text.topics[index])
                return model.predict_rows(dataset.inputs(np.asarray([index]), np.asarray([score])))
            for name in [*MODEL_NAMES, "resource_mean"]:
                factory = (lambda: ResourceMeanBaseline()) if name == "resource_mean" else (
                    lambda: CandidateSocialMetricsPredictor(name, args.seed, args.threads))
                fold_results, selection = [], []
                for i, f in enumerate(cv, 1):
                    progress(f"{name}: CV {i}/{args.folds}")
                    model = factory()
                    started = time.perf_counter()
                    model.fit(f["train_rows"], dataset.counts[f["train"]])
                    fit_seconds = time.perf_counter()-started
                    predicted = model.predict_rows(f["validation_rows"])
                    fold_results.append(dict(metrics=evaluate(dataset.counts[f["validation"]], predicted), fit_seconds=fit_seconds))
                    selection.append(normalized_mse(dataset.counts[f["validation"]], predicted, dataset.counts[f["train"]]))
                progress(f"{name}: held-out test")
                model = factory()
                started = time.perf_counter()
                model.fit(heldout["train_rows"], dataset.counts[train])
                fit_seconds = time.perf_counter()-started
                predictions[name] = model.predict_rows(heldout["validation_rows"])
                result = dict(cv_folds=fold_results, cv_selection=summarize(selection),
                    cv_metrics={m: {k: summarize([f["metrics"][m][k] for f in fold_results]) for k in ("mse", "rmse", "mae", "r2")} for m in METRICS},
                    test=evaluate(dataset.counts[test], predictions[name]),
                    test_normalized_mse=normalized_mse(dataset.counts[test], predictions[name], dataset.counts[train]),
                    fit_seconds=summarize([fit_seconds]),
                    batch_predict_seconds=timed_prediction(model, heldout["validation_rows"], 9),
                    single_predict_seconds=timed_prediction(model, heldout["validation_rows"][:1], 21))
                result["response_time"] = response_timing(lambda i: predict_case(model,i), latency_cases)
                if name != "resource_mean":
                    model.set_resources(dataset.resources)
                    model.metadata.update(purpose="evaluation; held-out test excluded", dataset_fingerprint=dataset.fingerprint)
                    model.save(output / "evaluation_models" / f"{name}.joblib")
                    result["parameters"] = [repr(e) for e in model.estimators_]
                report["results"][name] = result
                write_json(output / "report.json", report)
                progress(f"{name}: CV normalized MSE={result['cv_selection']['mean']:.4f}; fit={fit_seconds:.3f}s")
            report["selected_model"] = select_model(report["results"], MODEL_NAMES, "cv_selection", args.max_response_seconds)
            report["best_including_baseline"] = select_model(report["results"], report["results"], "cv_selection", args.max_response_seconds)
            np.savez_compressed(output / "test_predictions.npz", actual=dataset.counts[test], **predictions)
            with (output / "test_predictions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["article_id", "topic", "resource_id", "predicted_popularity_score",
                                 *[f"{n}_{m}" for n in ["actual", *predictions] for m in METRICS]])
                for j, i in enumerate(test):
                    writer.writerow([dataset.text.article_ids[i], dataset.text.topics[i], dataset.rows[i]["resource_id"],
                        heldout["validation_rows"][j]["score"], *dataset.counts[i],
                        *[v for p in predictions.values() for v in p[j]]])
            create_plots(output, report, dataset.counts[test], predictions)
            report.update(status="complete", completed_at=datetime.now(timezone.utc).isoformat())
            write_json(output / "report.json", report)
            write_summary(output, report)
            progress(f"Selected: {report['selected_model']}; results: {output}")
    finally:
        torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop if sys.platform == "win32" else None) as runner:
        runner.run(run())
