"""Multilingual E5 text encoder."""

import numpy as np
from sentence_transformers import SentenceTransformer


class TextEncoder:
    """Encode documents and queries using a multilingual E5 model."""

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
    ):
        self.model = SentenceTransformer(model_name)

    def encode_documents(
        self,
        texts: list[str],
    ) -> np.ndarray:
        """Encode document texts with the E5 passage prefix."""
        prefixed_texts = [
            f"passage: {text}" for text in texts
        ]
        return self.model.encode(
            prefixed_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def encode_queries(
        self,
        queries: list[str],
    ) -> np.ndarray:
        """Encode search queries with the E5 query prefix."""
        prefixed_queries = [
            f"query: {query}" for query in queries
        ]
        return self.model.encode(
            prefixed_queries,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
