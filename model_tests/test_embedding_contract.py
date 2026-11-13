"""A fixed model and explicit DB provenance prevent mixing vector spaces."""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

import joblib
import numpy as np
import psycopg

from src.database.model_embedding import MODEL_NAME, IncompatibleEmbeddingError, TextEmbedder
from src.database.table_classes import ArticleTextTable, ArticleVectorTable
from src.database.embeddings import reembed_database
from src.assesment_conform_of_PP_text_based import AssesmentConformPPtextBased, TextTopicEncoder
from src.assesment_conform_of_PP_text_based.dataset import dataset_from_rows


VECTOR = [1.0] + [0.0] * 383


class FixedModelTests(unittest.TestCase):
    def test_environment_cannot_select_another_embedding_model(self):
        factory = Mock(return_value=SimpleNamespace(get_embedding_dimension=lambda: 384))
        module = SimpleNamespace(SentenceTransformer=factory)
        with patch.dict(os.environ, {"EMBEDDING_MODEL": "another/model"}), patch.dict(
            "sys.modules", {"sentence_transformers": module}
        ):
            TextEmbedder()
        factory.assert_called_once_with(MODEL_NAME)
        with self.assertRaises(TypeError):
            TextEmbedder("another/model")
        with self.assertRaises(TypeError):
            TextTopicEncoder("another/model")

    def test_artifact_cannot_select_a_foreign_model_even_with_matching_header(self):
        model = AssesmentConformPPtextBased(epochs=1)
        model.train_features(np.ones((4, 768)), [-1, 0, 1, 2])
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            model.save(path)
            self.assertEqual(AssesmentConformPPtextBased.load(path).encoder.model_name, MODEL_NAME)
            payload = joblib.load(path)
            payload["embedding_model"] = "another/model"
            payload["assessment"].encoder.model_name = "another/model"
            joblib.dump(payload, path)
            with self.assertRaisesRegex(ValueError, "Invalid assessment artifact"):
                AssesmentConformPPtextBased.load(path)

    def test_cached_training_vectors_require_known_model_provenance(self):
        row = dict(article_id="article", resource_id="resource", text="Article", topic="Topic",
                   views=10000, reactions=80, comments=5, audience=100000, score=0.0,
                   embedding=VECTOR, embedding_model=MODEL_NAME)
        self.assertEqual(len(dataset_from_rows([row]).scores), 1)
        for name in (None, "another/model"):
            with self.subTest(model=name), self.assertRaises(IncompatibleEmbeddingError):
                dataset_from_rows([dict(row, embedding_model=name)])
        # A missing vector is safe: the fixed encoder will compute it from text.
        self.assertEqual(dataset_from_rows([dict(row, embedding=None, embedding_model=None)]).embeddings, [None])


class DatabaseEmbeddingTests(unittest.IsolatedAsyncioTestCase):
    async def test_old_schema_reports_the_migration_command(self):
        conn = SimpleNamespace(execute=AsyncMock(side_effect=psycopg.errors.UndefinedColumn()))
        with self.assertRaisesRegex(IncompatibleEmbeddingError, "python -m src.database reembed"):
            await ArticleVectorTable(conn).ensure_compatible()

    async def test_training_and_all_search_paths_reject_foreign_or_unknown_vectors(self):
        for model in (None, "another/model"):
            cursor = SimpleNamespace(fetchone=AsyncMock(return_value={
                "article_id": "foreign-article", "embedding_model": model,
            }))
            for operation in ("search", "search_popular", "best_by_topic", "training"):
                with self.subTest(model=model, operation=operation):
                    conn = SimpleNamespace(execute=AsyncMock(return_value=cursor))
                    vectors = ArticleVectorTable(conn)
                    with self.assertRaisesRegex(IncompatibleEmbeddingError, "python -m src.database reembed"):
                        if operation == "training":
                            await ArticleTextTable(conn).list_training_data()
                        elif operation == "best_by_topic":
                            await vectors.best_by_topic(VECTOR, topic="Topic")
                        else:
                            await getattr(vectors, operation)(VECTOR)
                    # The unsafe comparison/data query must never run.
                    self.assertEqual(conn.execute.await_count, 1)

    async def test_rebuild_preserves_concurrently_edited_and_deleted_articles(self):
        @asynccontextmanager
        async def transaction():
            yield

        conn = SimpleNamespace(transaction=transaction)
        context = AsyncMock()
        context.__aenter__.return_value = conn
        rows = [dict(id="unchanged", text="Text 1"), dict(id="edited", text="Old text"),
                dict(id="deleted", text="Deleted text")]
        texts = SimpleNamespace(list_embedding_inputs=AsyncMock(return_value=rows),
            get=AsyncMock(side_effect=[dict(text="Text 1"), dict(text="New text"), None]))
        vectors = SimpleNamespace(upsert=AsyncMock(), ensure_compatible=AsyncMock())
        embedder = SimpleNamespace(embed_document=Mock(return_value=VECTOR))
        with patch("src.database.embeddings.connect_database", AsyncMock(return_value=context)), \
                patch("src.database.embeddings.ArticleTextTable", return_value=texts), \
                patch("src.database.embeddings.ArticleVectorTable", return_value=vectors), \
                patch("src.database.embeddings.load_embedder", AsyncMock(return_value=embedder)):
            result = await reembed_database("fake-dsn", log=Mock())
        self.assertEqual(result, {"updated": 1, "skipped_changed_or_deleted": 2})
        vectors.upsert.assert_awaited_once_with("unchanged", VECTOR)
        vectors.ensure_compatible.assert_awaited_once()
        self.assertTrue(all(call.kwargs == {"for_update": True} for call in texts.get.await_args_list))


if __name__ == "__main__":
    unittest.main()
