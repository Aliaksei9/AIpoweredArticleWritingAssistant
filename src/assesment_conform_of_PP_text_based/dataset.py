"""Read training labels through the database module and prevent duplicate leakage."""

import hashlib
import json
import unicodedata
from dataclasses import dataclass

import numpy as np

from src.assesment_conform_of_PP_metric_based import AssesmentConformPPmetricsBased
from src.database.connection import connect_database
from src.database.table_classes import ArticleTextTable


def text_group(text: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@dataclass
class TrainingDataset:
    article_ids: list[str]
    texts: list[str]
    topics: list[str]
    scores: np.ndarray
    embeddings: list
    groups: np.ndarray
    report: dict

    @property
    def fingerprint(self) -> str:
        records = list(zip(self.article_ids, self.groups.tolist(), self.topics,
                           self.scores.tolist()))
        return hashlib.sha256(json.dumps(records, ensure_ascii=False).encode("utf-8")).hexdigest()


def dataset_from_rows(rows: list[dict]) -> TrainingDataset:
    assessor = AssesmentConformPPmetricsBased()
    accepted = []
    missing_labels = empty_texts = 0
    for row in rows:
        expected = assessor.assess(row["views"], row["reactions"], row["comments"], row["audience"])
        score = row["score"]
        if ((expected is None) != (score is None)
                or (expected is not None and abs(expected - float(score)) > 1e-5)):
            raise ValueError("Stale metric scores: run python -m src.database init-db first")
        if score is None:
            missing_labels += 1
            continue
        if not np.isfinite(score) or not -10 <= score <= 10:
            raise ValueError("Database contains an invalid target score")
        if not row["text"] or not row["text"].strip() or not row["topic"] or not row["topic"].strip():
            empty_texts += 1
            continue
        accepted.append(row)
    if not accepted:
        raise ValueError("No training rows with text, topic and a complete metric score")
    texts = [row["text"] for row in accepted]
    groups = np.asarray([text_group(text) for text in texts])
    return TrainingDataset(
        article_ids=[str(row["article_id"]) for row in accepted], texts=texts,
        topics=[row["topic"].strip() for row in accepted],
        scores=np.asarray([row["score"] for row in accepted], dtype=np.float64),
        embeddings=[row["embedding"] for row in accepted], groups=groups,
        report=dict(database_rows=len(rows), training_rows=len(accepted),
                    excluded_missing_labels=missing_labels, excluded_empty_texts=empty_texts,
                    duplicate_rows=len(accepted) - len(set(groups)),
                    resource_count=len({str(row["resource_id"]) for row in accepted}),
                    target_audience="per-article metric observation snapshot",
                    rows_by_resource={resource: sum(str(row["resource_id"]) == resource for row in accepted)
                                      for resource in sorted({str(row["resource_id"]) for row in accepted})}),
    )


async def load_training_dataset(dsn: str) -> TrainingDataset:
    async with await connect_database(dsn) as conn:
        rows = await ArticleTextTable(conn).list_training_data()
    return dataset_from_rows(rows)
