"""FAISS vector index using inner-product similarity."""

from pathlib import Path

import faiss
import numpy as np


class FaissIndex:
    """Vector index for normalized text embeddings."""

    def __init__(self, dimension: int):
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.index = faiss.IndexFlatIP(dimension)

    def add(self, embeddings: np.ndarray) -> None:
        """Add embeddings to the index."""
        embeddings = np.asarray(embeddings, dtype=np.float32)

        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D array")
        if embeddings.shape[1] != self.index.d:
            raise ValueError("embedding dimension mismatch")
        if embeddings.shape[0] == 0:
            return
        if not np.isfinite(embeddings).all():
            raise ValueError("embeddings must contain only finite values")

        self.index.add(np.ascontiguousarray(embeddings))

    def search(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return similarity scores and matching vector indices."""
        query_embeddings = np.asarray(
            query_embeddings,
            dtype=np.float32,
        )

        if query_embeddings.ndim != 2:
            raise ValueError("query_embeddings must be a 2D array")
        if query_embeddings.shape[1] != self.index.d:
            raise ValueError("query embedding dimension mismatch")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.index.ntotal == 0:
            raise ValueError("cannot search an empty index")
        if not np.isfinite(query_embeddings).all():
            raise ValueError(
                "query embeddings must contain only finite values"
            )

        top_k = min(top_k, self.index.ntotal)
        return self.index.search(
            np.ascontiguousarray(query_embeddings),
            top_k,
        )

    def save(self, path: str | Path) -> None:
        """Save the FAISS index to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, path: str | Path) -> "FaissIndex":
        """Load a FAISS index from disk."""
        path = Path(path)

        if not path.is_file():
            raise FileNotFoundError(f"FAISS index not found: {path}")

        index = faiss.read_index(str(path))

        if not isinstance(index, faiss.IndexFlatIP):
            raise ValueError("Unsupported FAISS index type")

        instance = cls(dimension=index.d)
        instance.index = index
        return instance

    @property
    def size(self) -> int:
        """Number of indexed vectors."""
        return self.index.ntotal
