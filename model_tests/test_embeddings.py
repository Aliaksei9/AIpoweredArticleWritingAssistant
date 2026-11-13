"""Offline regression checks for token boundaries and shared vector operations."""

import math
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock
from src.database.model_embedding import EmptyTextError, TextEmbedder
from src.database.table_classes import ArticleVectorTable
from src.database.vector_utils import EMBEDDING_DIMENSIONS, normalize_vector, validate_vector


class ExpandingTokenizer:
    """Decoded input can grow when the prefix is added, as in real tokenizers."""

    def encode(self, text, *, add_special_tokens=False, **_):
        if text.startswith(("passage: ", "query: ")):
            _, body = text.split(": ", 1)
            tokens = [0] * (2 + len(body) + body.count("x"))
        else:
            tokens = [ord(char) for char in text if char != "\u200b"]
        # Deliberately use three tokens: callers must measure, not assume two.
        return [-1, *tokens, -2, -3] if add_special_tokens else tokens

    def decode(self, tokens):
        return "".join(chr(token) for token in tokens)


class ByteTokenizer:
    def encode(self, text, *, add_special_tokens=False, **_):
        tokens = list(text.encode("utf-8"))
        return [-1, *tokens, -2] if add_special_tokens else tokens

    def decode(self, tokens):
        return bytes(tokens).decode("utf-8", errors="replace")


class RecordingModel:
    def __init__(self, tokenizer, capacity=12):
        self.tokenizer = tokenizer
        self.max_seq_length = capacity
        self.inputs = []

    def encode(self, chunks, **_):
        self.inputs.extend(chunks)
        for chunk in chunks:
            length = len(self.tokenizer.encode(chunk, add_special_tokens=True))
            if length > min(self.max_seq_length or 512, 512):
                raise AssertionError(f"Model input would be truncated: {length}")
        return [[3.0, 4.0] + [0.0] * (EMBEDDING_DIMENSIONS - 2) for _ in chunks]


def make_embedder(tokenizer, capacity=12):
    embedder = TextEmbedder.__new__(TextEmbedder)
    embedder.model = RecordingModel(tokenizer, capacity)
    return embedder


class EmbeddingTests(unittest.TestCase):
    def test_retokenized_chunks_fit_and_keep_every_original_token(self):
        text = "xxabcxxxxxdefxxghijklxxx"
        embedder = make_embedder(ExpandingTokenizer())
        result = embedder.embed_document(text)
        self.assertGreater(len(embedder.model.inputs), 1)
        # Concatenating contiguous decoded slices proves no gap or overlap.
        recovered = "".join(chunk.removeprefix("passage: ") for chunk in embedder.model.inputs)
        self.assertEqual(recovered, text)
        self.assertEqual(len(result), EMBEDDING_DIMENSIONS)
        self.assertAlmostEqual(math.hypot(*result), 1.0)
        self.assertAlmostEqual(result[0], 0.6)
        self.assertAlmostEqual(result[1], 0.8)

    def test_query_prefix_uses_the_same_capacity_checks(self):
        embedder = make_embedder(ExpandingTokenizer())
        embedder.embed_query("xxxxxxxxxxxx")
        self.assertTrue(all(chunk.startswith("query: ") for chunk in embedder.model.inputs))
        self.assertEqual("".join(chunk.removeprefix("query: ") for chunk in embedder.model.inputs),
                         "xxxxxxxxxxxx")

    def test_unicode_byte_tokens_are_not_split_inside_a_character(self):
        text = "a😀б🐈c😀"
        embedder = make_embedder(ByteTokenizer(), capacity=16)
        embedder.embed_document(text)
        recovered = "".join(chunk.removeprefix("passage: ") for chunk in embedder.model.inputs)
        self.assertEqual(recovered, text)
        self.assertNotIn("\ufffd", recovered)

    def test_blank_or_tokenizer_empty_text_has_specific_error(self):
        for text in ("", " \n\t", "\u200b", "\u200b\u200b"):
            with self.subTest(text=repr(text)):
                embedder = make_embedder(ExpandingTokenizer())
                with self.assertRaises(EmptyTextError):
                    embedder.embed_document(text)
                self.assertEqual(embedder.model.inputs, [])

    def test_model_must_have_capacity_for_a_prefixed_token(self):
        embedder = make_embedder(ExpandingTokenizer(), capacity=5)
        with self.assertRaisesRegex(ValueError, "too little input capacity"):
            embedder.embed_document("abc")

    def test_no_complete_unicode_character_fits(self):
        embedder = make_embedder(ByteTokenizer(), capacity=13)
        with self.assertRaisesRegex(ValueError, "complete text token"):
            embedder.embed_document("😀")

    def test_zero_model_output_is_rejected(self):
        embedder = make_embedder(ExpandingTokenizer())
        embedder.model.encode = lambda *args, **kwargs: [[0.0] * EMBEDDING_DIMENSIONS]
        with self.assertRaisesRegex(ValueError, "zero vector"):
            embedder.embed_document("abc")


class VectorUtilityTests(unittest.TestCase):
    def test_validation_preserves_magnitude_and_normalization_returns_a_unit_vector(self):
        original = [3.0, 4.0] + [0.0] * (EMBEDDING_DIMENSIONS - 2)
        self.assertEqual(validate_vector(original), original)
        normalized = normalize_vector(original)
        self.assertEqual(normalized[:2], [0.6, 0.8])
        self.assertAlmostEqual(math.hypot(*normalized), 1.0)
        # The database adapter must validate without normalizing stored values.
        stored = ArticleVectorTable._vector(original)
        stored_values = stored.to_list() if hasattr(stored, "to_list") else list(stored)
        self.assertEqual(stored_values, original)

    def test_shared_validation_rejects_invalid_vectors(self):
        cases = ([1.0], [0.0] * EMBEDDING_DIMENSIONS,
                 [math.nan] + [0.0] * (EMBEDDING_DIMENSIONS - 1),
                 [math.inf] + [0.0] * (EMBEDDING_DIMENSIONS - 1))
        for vector in cases:
            for operation in (validate_vector, normalize_vector, ArticleVectorTable._vector):
                with self.subTest(operation=operation.__name__, vector=vector[:2]):
                    with self.assertRaises(ValueError):
                        operation(vector)

    def test_normalization_avoids_overflow_for_large_finite_coordinates(self):
        vector = [1e308, 1e308] + [0.0] * (EMBEDDING_DIMENSIONS - 2)
        normalized = normalize_vector(vector)
        self.assertTrue(all(math.isfinite(value) for value in normalized))
        self.assertAlmostEqual(math.hypot(*normalized), 1.0)


class VectorSearchValidationTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_search_methods_reject_invalid_limits_before_database_access(self):
        for limit in (0, -1, True, 1.5, None):
            for operation in ("search", "search_popular", "best_by_topic"):
                with self.subTest(limit=limit, operation=operation):
                    conn = SimpleNamespace(execute=AsyncMock())
                    vectors = ArticleVectorTable(conn)
                    with self.assertRaises(ValueError):
                        if operation == "best_by_topic":
                            await vectors.best_by_topic([1.] + [0.] * 383, topic="Topic", limit=limit)
                        else:
                            await getattr(vectors, operation)([1.] + [0.] * 383, limit=limit)
                    conn.execute.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
