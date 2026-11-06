"""Standalone scientific figures for comparing popularity regressors."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LABELS = {"gradient_boosting": "Градиентный\nбустинг", "random_forest": "Случайный\nлес",
          "catboost": "CatBoost", "neural_network": "Нейросеть\nMLP",
          "mean_baseline": "Среднее\n(база)"}
COLORS = ["#4779b0", "#65a278", "#dfac51", "#a173b0", "#969da6"]


def _save(fig, output, name):
    fig.savefig(output / f"{name}.png", dpi=160)
    fig.savefig(output / f"{name}.svg")
    plt.close(fig)


def create_plots(output: Path, report: dict, actual, predictions: dict, history=None):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    results = report["results"]
    names = list(results)
    positions = np.arange(len(names))
    labels = [LABELS[name] for name in names]
    fig, ax = plt.subplots(figsize=(11, 5.3), layout="constrained")
    cv = [results[name]["cv_mse"]["mean"] for name in names]
    test = [results[name]["test"]["mse"] for name in names]
    ax.bar(positions - .2, cv, .4, label=f"CV: {report['folds']} части обучения",
           yerr=[results[name]["cv_mse"]["std"] for name in names], capsize=4, color="#4779b0")
    ax.bar(positions + .2, test, .4, label=f"Отложенная проверка: {report['test_rows']} статей",
           color="#dfac51")
    for x, value in zip(positions + .2, test):
        ax.text(x, value + .08, f"{value:.3f}", ha="center", fontsize=9)
    upper = max([value + results[name]["cv_mse"]["std"] for name, value in zip(names, cv)] + test)
    ax.set(xticks=positions, xticklabels=labels, ylabel="MSE, баллы²", ylim=(0, upper * 1.25),
           title="Среднеквадратичная ошибка · меньше — лучше")
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)
    _save(fig, output, "mse_comparison")

    models = [name for name in names if name != "mean_baseline"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    specifications = [
        ([results[name]["fit_seconds"] for name in models], "Обучение", "Секунды"),
        ([results[name]["batch_prediction"]["median"] * 1000 for name in models],
         f"Прогноз для {report['test_rows']} статей", "Медиана, мс"),
        ([results[name]["single_prediction"]["median"] * 1000 for name in models],
         "Прогноз для одной статьи", "Медиана, мс"),
    ]
    for ax, (values, title, ylabel) in zip(axes, specifications):
        ax.bar(range(len(models)), values, color=COLORS[:len(models)])
        for x, value in enumerate(values):
            ax.text(x, value + max(values) * .025, f"{value:.3f}", ha="center", fontsize=9)
        ax.set(xticks=range(len(models)), xticklabels=[LABELS[name] for name in models],
               title=title, ylabel=ylabel, ylim=(0, max(values) * 1.2))
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle(f"Время на CPU · {report['threads']} потока · готовые эмбеддинги", fontsize=13)
    _save(fig, output, "execution_time")

    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    for index, name in enumerate(models):
        ax.scatter(results[name]["fit_seconds"], results[name]["test"]["mse"],
                   s=100, color=COLORS[index], label=LABELS[name].replace("\n", " "))
    ax.set(xlabel="Время обучения, с", ylabel="MSE на отложенной проверке, баллы²",
           title="Ошибка и время обучения · ближе к левому нижнему углу — лучше")
    ax.grid(alpha=.2)
    ax.legend()
    _save(fig, output, "mse_vs_training_time")

    for index, name in enumerate(models):
        predicted = predictions[name]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), layout="constrained")
        axes[0].scatter(actual, predicted, s=9, alpha=.3, color=COLORS[index])
        axes[0].plot([-10, 10], [-10, 10], "--", color="#333333", linewidth=1)
        axes[0].set(xlabel="Оценка по метрикам", ylabel="Прогноз по тексту",
                    xlim=(-10, 10), ylim=(-10, 10), title="Оценки и прогнозы")
        axes[1].hist(predicted - actual, bins=35, color=COLORS[index], edgecolor="white")
        axes[1].axvline(0, color="#333333", linestyle="--", linewidth=1)
        axes[1].set(xlabel="Прогноз − оценка по метрикам", ylabel="Количество статей",
                    title="Распределение ошибок")
        fig.suptitle(f"{LABELS[name].replace(chr(10), ' ')} · MSE = {results[name]['test']['mse']:.3f}")
        _save(fig, output, f"{name}_diagnostics")

    if history:
        fig, ax = plt.subplots(figsize=(8, 4.4), layout="constrained")
        ax.plot([row["epoch"] for row in history], [row["train_mse"] for row in history], color=COLORS[3])
        ax.set(xlabel="Эпоха", ylabel="MSE на обучающей выборке, баллы²",
               title="Обучение MLP · история не используется для оценки тестовой ошибки")
        ax.grid(alpha=.2)
        _save(fig, output, "neural_network_learning_curve")
