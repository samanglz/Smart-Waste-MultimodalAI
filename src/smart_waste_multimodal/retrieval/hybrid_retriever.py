"""Hybrid retrieval combining FAISS and BM25 with RRF."""

from smart_waste_multimodal.embeddings.encoder import TextEncoder
from smart_waste_multimodal.indexing.faiss_index import FaissIndex
from smart_waste_multimodal.retrieval.bm25_retriever import BM25Retriever
from smart_waste_multimodal.retrieval.reciprocal_rank_fusion import (
    reciprocal_rank_fusion,
)


class HybridRetriever:
    """Combine dense and lexical retrieval using RRF."""

    def __init__(
        self,
        chunks: list[dict[str, str]],
        encoder: TextEncoder,
        vector_index: FaissIndex,
        bm25_retriever: BM25Retriever,
    ):
        if not chunks:
            raise ValueError("chunks must not be empty")

        self.chunks = chunks
        self.encoder = encoder
        self.vector_index = vector_index
        self.bm25_retriever = bm25_retriever

    def search(
        self,
        query: str,
        top_k: int = 5,
        rrf_k: int = 60,
    ) -> list[dict[str, str | int | float]]:
        """Retrieve and fuse results from FAISS and BM25."""
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query_embedding = self.encoder.encode_queries([query])

        _, vector_indices = self.vector_index.search(
            query_embedding,
            top_k=top_k,
        )
        vector_ranking = vector_indices[0].tolist()

        bm25_results = self.bm25_retriever.search(
            query,
            top_k=top_k,
        )
        bm25_ranking = [idx for idx, _ in bm25_results]

        hybrid_ranking, fused_scores = reciprocal_rank_fusion(
            [vector_ranking, bm25_ranking],
            k=rrf_k,
        )

        return [
            {
                "chunk_id": self.chunks[idx]["chunk_id"],
                "source": self.chunks[idx]["source"],
                "text": self.chunks[idx]["text"],
                "metadata": self.chunks[idx].get("metadata", {}).copy(),
                "score": fused_scores[idx],
                "rank": rank,
            }
            for rank, idx in enumerate(hybrid_ranking[:top_k], start=1)
        ]
