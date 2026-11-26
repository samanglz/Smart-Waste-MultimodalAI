"""Lexical retrieval using BM25Okapi."""

from rank_bm25 import BM25Okapi


class BM25Retriever:
    """Lexical retrieval using BM25Okapi."""

    def __init__(self, documents: list[dict[str, str]]):
        if not documents:
            raise ValueError("documents must not be empty")

        self.documents = documents
        self.tokenized_documents = [
            self._tokenize(document["text"])
            for document in documents
        ]

        if any(not tokens for tokens in self.tokenized_documents):
            raise ValueError("documents must contain non-empty text")

        self.bm25 = BM25Okapi(self.tokenized_documents)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Basic whitespace tokenization."""
        return text.lower().split()

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        """Return (document index, BM25 score) pairs."""
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        scores = self.bm25.get_scores(self._tokenize(query))
        ranked_indices = scores.argsort()[::-1][:top_k]

        return [
            (int(index), float(scores[index]))
            for index in ranked_indices
        ]
