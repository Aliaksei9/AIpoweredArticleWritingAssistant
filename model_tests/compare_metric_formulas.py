"""Read-only comparison of metric formulas on the current PostgreSQL snapshot."""

import argparse
import asyncio
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys

from dotenv import load_dotenv
import numpy as np

from src.database.connection import connect_database
from src.database.table_classes import ArticleTextTable
from src.assesment_conform_of_PP_metric_based import AssesmentConformPPmetricsBased


def old_score(v, r, c, a):
    def signal(x, reference):
        return -1.0 if x == 0 else math.tanh((math.log(x)-math.log(a)-math.log(reference))/1.5)
    return 10*(.2*signal(v, .1)+.4*signal(r, .0008)+.4*signal(c, .00005))


def describe(values):
    return dict(zip(["min", "p25", "median", "p75", "max"], np.percentile(values, [0, 25, 50, 75, 100]).tolist()),
                mean=float(np.mean(values)), std=float(np.std(values)),
                negative=int(np.count_nonzero(np.asarray(values) < 0)))


async def run():
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    async with await connect_database(os.environ["DATABASE_URL"]) as conn:
        rows = await ArticleTextTable(conn).list_training_data()
    assessor = AssesmentConformPPmetricsBased()
    old, new, invalid = [], [], []
    for row in rows:
        values = [row[key] for key in ("views", "reactions", "comments", "audience")]
        score = assessor.assess(*values)
        if score is None:
            invalid.append(str(row["article_id"])); continue
        new.append(score); old.append(old_score(*values))
    if not new:
        raise ValueError("No valid articles for comparing formulas")
    report = dict(created_at=datetime.now(timezone.utc).isoformat(), rows=len(rows), eligible=len(new),
        excluded=len(invalid), excluded_article_ids=invalid,
        zeros={key: sum(row[key] == 0 for row in rows) for key in ("views", "reactions", "comments", "audience")},
        old_version="log-relative-v2", new_version=assessor.FORMULA_VERSION,
        old=describe(old), new=describe(new))
    (args.output/"report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output/"previous_scores.json").write_text(json.dumps([
        dict(article_id=str(row["article_id"]), score=row["score"]) for row in rows], indent=2), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(9, 4.8))
    bins = np.linspace(-10, 10, 41)
    axis.hist(old, bins=bins, alpha=.45, label="Прежняя формула", color="#8d97ad")
    axis.hist(new, bins=bins, alpha=.65, label="Новая формула", color="#4263eb")
    axis.axvline(0, color="#50596a", linewidth=.8, linestyle="--")
    axis.set(xlim=(-10, 10), xlabel="Оценка соответствия паттернам популярности",
             ylabel="Количество статей", title=f"Сравнение на {len(new)} статьях с корректными метриками")
    axis.legend(); axis.grid(axis="y", alpha=.15); figure.tight_layout()
    figure.savefig(args.output/"distribution.png", dpi=150)
    plt.close(figure)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop if sys.platform == "win32" else None) as runner:
        runner.run(run())
