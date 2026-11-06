"""The dataset auditor reads the unified report for multiple sources."""

import json
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from model_tests import validate_collection
from src.assesment_conform_of_PP_metric_based import AssesmentConformPPmetricsBased
from src.database.vector_utils import EMBEDDING_DIMENSIONS


class CollectionValidationTests(unittest.IsolatedAsyncioTestCase):
    async def test_audits_unified_resources_from_multiple_sources(self):
        as_of = datetime(2026, 10, 23, tzinfo=timezone.utc)
        identifiers = [uuid4(), uuid4()]
        records, rows = [], []
        score = AssesmentConformPPmetricsBased().assess(100, 10, 1, 1000)
        for identifier, source, address in zip(identifiers, ("TG", "WEB"), ("@news", "feed")):
            records.append(dict(
                source=source, address=address, processed=1,
                resource=dict(id=str(identifier), audience=1000),
                window=dict(oldest=(as_of - timedelta(days=14)).isoformat(),
                            newest=(as_of - timedelta(days=2)).isoformat()),
            ))
            rows.append(dict(
                resource_id=identifier, published_at=as_of - timedelta(days=3),
                audience=1000, collected_at=as_of, views=100, reactions=10, comments=1,
                score=score, embedding=[1.0] + [0.0] * (EMBEDDING_DIMENSIONS - 1),
            ))
        connection = AsyncMock()
        connection.__aenter__.return_value = connection
        with TemporaryDirectory() as directory:
            report_path = Path(directory) / "update.json"
            output_path = Path(directory) / "audit.json"
            report_path.write_text(json.dumps(dict(
                status="complete", selected_sources=["TG", "WEB"],
                resources=records, limit_per_resource=1000,
            )), encoding="utf-8")
            with patch.object(validate_collection, "load_dotenv"), patch.dict(
                "os.environ", {"DATABASE_URL": "fake-dsn"}
            ), patch("sys.argv", ["audit", str(report_path), "--output", str(output_path)]), patch.object(
                validate_collection, "connect_database", AsyncMock(return_value=connection)
            ), patch.object(validate_collection, "ArticleTextTable") as texts, patch.object(
                validate_collection, "ResourceTable"
            ) as resources, patch("builtins.print"):
                texts.return_value.list_training_data = AsyncMock(return_value=rows)
                resources.return_value.list_all = AsyncMock(return_value=[dict(id=value) for value in identifiers])
                await validate_collection.run()
            output = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(output["status"], "passed")
        self.assertEqual(output["total_articles"], 2)
        self.assertEqual(output["labeled_articles"], 2)
        self.assertEqual([(row["source"], row["address"]) for row in output["resources"]],
                         [("TG", "@news"), ("WEB", "feed")])
        self.assertNotIn("channels", output)
