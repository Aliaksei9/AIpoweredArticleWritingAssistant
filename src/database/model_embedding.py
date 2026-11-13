"""Configured sentence-transformer embeddings for articles and RAG queries."""

import math

from .vector_utils import EMBEDDING_DIMENSIONS, normalize_vector
from .embedding_config import MODEL_NAME,SETTINGS


class IncompatibleEmbeddingError(ValueError):
    """Stored vectors cannot be used with the fixed embedding model."""


class EmptyTextError(ValueError):
    """The input contains no text that the embedding tokenizer can process."""


class TextEmbedder:
    def __init__(self):
        # Import lazily so schema initialization does not load PyTorch.
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(MODEL_NAME)
        get_dimensions = getattr(self.model, "get_embedding_dimension", None)
        if get_dimensions is None:
            get_dimensions = self.model.get_sentence_embedding_dimension
        dimensions = get_dimensions()
        if dimensions != EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"Model produces {dimensions} dimensions; EMBEDDING_DIMENSIONS={EMBEDDING_DIMENSIONS}. "
                "Set EMBEDDING_DIMENSIONS to the model output size and restart."
            )

    def _chunks(self, prefix: str, text: str) -> list[str]:
        """Split all original tokens into inputs that fit after re-tokenization."""
        if not text.strip():
            raise EmptyTextError("Cannot embed empty text")
        tokenizer = self.model.tokenizer
        prefix_text = f"{prefix}: " if prefix else ""
        # Read all tokens before splitting; this sequence is not sent to the model.
        token_ids = tokenizer.encode(
            text, add_special_tokens=False, truncation=False, verbose=False
        )
        if not token_ids:
            raise EmptyTextError("Text contains no embeddable tokens")
        capacity = min(self.model.max_seq_length or 512, 512)
        prefix_size = len(tokenizer.encode(
            prefix_text, add_special_tokens=True, truncation=False, verbose=False
        ))
        chunk_size = capacity - prefix_size
        if chunk_size <= 0:
            raise ValueError("Embedding model has too little input capacity")
        check_unicode_boundaries = "\ufffd" not in text
        chunks = []
        start = 0
        while start < len(token_ids):
            end = min(start + chunk_size, len(token_ids))
            while end > start:
                body = tokenizer.decode(token_ids[start:end])
                chunk = prefix_text + body
                size = len(tokenizer.encode(
                    chunk, add_special_tokens=True, truncation=False, verbose=False
                ))
                # Byte-level tokenizers may split a Unicode character between
                # tokens. Prefer a boundary that decodes without replacement.
                broken_character = check_unicode_boundaries and "\ufffd" in body
                if size <= capacity and not broken_character:
                    break
                end -= 1
            else:
                raise ValueError("Embedding model cannot fit a complete text token")
            chunks.append(chunk)
            start = end
        return chunks

    def _encode(self, prefix: str, text: str) -> list[float]:
        chunks = self._chunks(prefix, text)
        encoded = self.model.encode(
            chunks, normalize_embeddings=True, convert_to_numpy=True, batch_size=8
        )
        mean = [
            math.fsum(float(vector[index]) for vector in encoded) / len(encoded)
            for index in range(EMBEDDING_DIMENSIONS)
        ]
        return normalize_vector(mean)

    def embed_document(self, text: str) -> list[float]:
        return self._encode(SETTINGS.document_prefix, text)

    def embed_query(self, text: str) -> list[float]:
        return self._encode(SETTINGS.query_prefix, text)
