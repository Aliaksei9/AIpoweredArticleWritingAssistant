"""Export module-4 errors and timings as standalone PNG and SVG figures."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = {"gradient_boosting": "Градиентный\nбустинг", "random_forest": "Случайный\nлес",
          "catboost": "CatBoost", "neural_network": "Нейросеть", "resource_mean": "Среднее\nресурса"}
METRIC_LABELS = {"views": "Просмотры", "reactions": "Реакции", "comments": "Комментарии"}


def create_plots(output, report, actual, predictions):
    output = Path(output)
    names = list(report["results"])
    labels = [LABELS[n] for n in names]
    x = np.arange(len(names))
    def save(fig, name):
        fig.savefig(output / f"{name}.png", dpi=170)
        fig.savefig(output / f"{name}.svg")
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    for ax, metric in zip(axes, METRIC_LABELS):
        cv = [report["results"][n]["cv_metrics"][metric]["mse"]["mean"] for n in names]
        test = [report["results"][n]["test"][metric]["mse"] for n in names]
        ax.bar(x-.18, cv, .36, label="CV", color="#4779b0")
        ax.bar(x+.18, test, .36, label="Отложенная проверка", color="#dfac51")
        ax.set(title=METRIC_LABELS[metric], ylabel="MSE, количество²", xticks=x, xticklabels=labels)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=.2)
        ax.legend(fontsize=8)
    save(fig, "mse_comparison")

    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    cv = [report["results"][n]["cv_selection"]["mean"] for n in names]
    test = [report["results"][n]["test_normalized_mse"] for n in names]
    ax.bar(x-.18, cv, .36, label="CV", color="#4779b0")
    ax.bar(x+.18, test, .36, label="Отложенная проверка", color="#dfac51")
    ax.set(ylabel="Среднее MSE / дисперсия метрики обучения", xticks=x, xticklabels=labels,
           title="Общий критерий: равный вес трёх социальных метрик")
    ax.grid(axis="y", alpha=.2)
    ax.legend()
    save(fig, "normalized_mse_comparison")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout="constrained")
    for ax, key, title, scale in zip(axes,
        ["fit_seconds", "batch_predict_seconds", "single_predict_seconds"],
        ["Обучение трёх регрессоров, с", f"Прогноз {report['test_rows']} статей, мс", "Прогноз одной статьи, мс"],
        [1, 1000, 1000]):
        values = [report["results"][n][key]["median"]*scale for n in names]
        ax.bar(x, values, color="#65a278")
        ax.set(title=title, xticks=x, xticklabels=labels)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=.2)
    save(fig, "runtime_comparison")

    for name, values in predictions.items():
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), layout="constrained")
        for i, (metric, label) in enumerate(METRIC_LABELS.items()):
            ax = axes[i]
            ax.scatter(actual[:, i], values[:, i], s=12, alpha=.45)
            high = max(float(actual[:, i].max()), float(values[:, i].max()), 1)
            ax.plot([0, high], [0, high], "--", color="#777777")
            ax.set(xscale="symlog", yscale="symlog", title=label,
                   xlabel="Факт", ylabel="Прогноз")
            ax.grid(alpha=.15)
        fig.suptitle(LABELS[name].replace("\n", " "))
        save(fig, f"diagnostics_{name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    report = json.loads((args.directory / "report.json").read_text(encoding="utf-8"))
    with np.load(args.directory / "test_predictions.npz") as data:
        create_plots(args.directory, report, data["actual"],
                     {n: data[n] for n in report["results"]})
