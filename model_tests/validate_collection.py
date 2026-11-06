"""Audit all collected rows against the exact window and audience in a report."""

import argparse
import asyncio
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from src.assesment_conform_of_PP_metric_based import AssesmentConformPPmetricsBased
from src.database.connection import connect_database
from src.database.table_classes import ArticleTextTable, ResourceTable


async def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collection_reports", type=Path, nargs="+",
                        help="Reports for all DB resources; later reports replace earlier ones for the same resource")
    parser.add_argument("--output", type=Path, default=Path("model_tests/results/collection_validation.json"))
    args = parser.parse_args()
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    records = {}
    for path in args.collection_reports:
        collection = json.loads(path.read_text(encoding="utf-8"))
        assert collection["status"] == "complete"
        for record in collection["resources"]:
            records[record["resource"]["id"]] = dict(record, limit=collection["limit_per_resource"])
    assessor = AssesmentConformPPmetricsBased()
    async with await connect_database(os.environ["DATABASE_URL"]) as conn:
        rows = await ArticleTextTable(conn).list_training_data()
        resources = await ResourceTable(conn).list_all()
    assert {str(resource["id"]) for resource in resources} == records.keys()
    counts, scored = Counter(), Counter()
    for row in rows:
        resource_id = str(row["resource_id"])
        record = records[resource_id]
        assert datetime.fromisoformat(record["window"]["oldest"]) < row["published_at"] < datetime.fromisoformat(record["window"]["newest"])
        assert row["audience"] == record["resource"]["audience"]
        assert row["collected_at"] is not None
        expected = assessor.assess(row["views"], row["reactions"], row["comments"], row["audience"])
        assert (expected is None and row["score"] is None) or (expected is not None and row["score"] is not None and abs(expected - row["score"]) < 1e-6)
        vector = row["embedding"]
        assert vector is not None and len(vector) == 384
        assert all(math.isfinite(value) for value in vector)
        assert abs(math.sqrt(sum(value * value for value in vector)) - 1) < 1e-5
        counts[resource_id] += 1
        scored[resource_id] += row["score"] is not None
    for resource_id, record in records.items():
        assert counts[resource_id] == record["processed"]
        assert counts[resource_id] <= record["limit"]
    output = dict(status="passed", total_articles=len(rows), labeled_articles=sum(scored.values()),
                  resources=[dict(address=record["address"], source=record["source"],
                                 articles=counts[resource_id],
                                 labeled_articles=scored[resource_id], audience=record["resource"]["audience"],
                                 oldest=min((row["published_at"].isoformat() for row in rows if str(row["resource_id"]) == resource_id), default=None),
                                 newest=max((row["published_at"].isoformat() for row in rows if str(row["resource_id"]) == resource_id), default=None))
                            for resource_id, record in records.items()],
                  checks=["all_expected_resources", "strict_publication_window", "per_resource_limit",
                          "audience_snapshot", "metric_scores", "normalized_vectors", "article_counts"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop if sys.platform == "win32" else None) as runner:
        runner.run(run())
