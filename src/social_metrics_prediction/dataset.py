"""Read a consistent training snapshot through database table classes."""

from dataclasses import dataclass
import hashlib
import json
import numpy as np

from src.database.connection import connect_database
from src.database.table_classes import ArticleTextTable, ResourceTable
from src.assesment_conform_of_PP_text_based.dataset import dataset_from_rows, TrainingDataset

from .features import METRICS, validate_counts
from .resources import ResourceInfo, normalize_source


@dataclass
class MetricDataset:
    text: TrainingDataset
    rows: list[dict]
    counts: np.ndarray
    resources: list[ResourceInfo]

    @property
    def groups(self):
        return self.text.groups

    @property
    def scores(self):
        return self.text.scores

    @property
    def fingerprint(self):
        payload = [self.text.fingerprint, self.rows, self.counts.tolist()]
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def inputs(self, indices, scores):
        if len(indices) != len(scores):
            raise ValueError("Score count does not match requested rows")
        return [dict(self.rows[int(i)], score=float(score)) for i, score in zip(indices, scores)]


def metric_dataset_from_rows(rows, resources):
    text = dataset_from_rows(rows)
    by_id = {str(r["article_id"]): r for r in rows}
    accepted = [by_id[i] for i in text.article_ids]
    # The module-2 score is deliberately absent here; it is a training label only.
    inputs = [dict(topic=r["topic"].strip(), resource_id=str(r["resource_id"]),
                   source=normalize_source(r["source"]), audience=int(r["audience"])) for r in accepted]
    if any(r["resource_id"] == "None" for r in inputs):
        raise ValueError("Training article has no resource")
    counts = validate_counts([[r[m] for m in METRICS] for r in accepted], len(accepted))
    return MetricDataset(text, inputs, counts, resources)


async def load_metric_dataset(dsn):
    async with await connect_database(dsn) as conn:
        rows = await ArticleTextTable(conn).list_training_data(include_resource_metadata=True)
        catalog = await ResourceTable(conn).list_prediction_resources()
    resources = [ResourceInfo.from_database(r) for r in catalog if r["reader_count"] and r["reader_count"] > 0]
    return metric_dataset_from_rows(rows, resources)
