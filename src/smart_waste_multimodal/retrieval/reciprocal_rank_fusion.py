"""Reciprocal Rank Fusion implementation."""


def reciprocal_rank_fusion(
    rankings: list[list[int]],
    k: int = 60,
) -> tuple[list[int], dict[int, float]]:
    """Combine ranked document indices using Reciprocal Rank Fusion."""
    if k <= 0:
        raise ValueError("k must be positive")

    fused_scores: dict[int, float] = {}

    for ranking in rankings:
        for rank, doc_idx in enumerate(ranking, start=1):
            fused_scores[doc_idx] = (
                fused_scores.get(doc_idx, 0.0)
                + 1.0 / (k + rank)
            )

    fused_ranking = sorted(
        fused_scores,
        key=fused_scores.get,
        reverse=True,
    )

    return fused_ranking, fused_scores
