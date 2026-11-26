"""Retrieval-stage output contracts."""

import math
from typing import Literal

from pydantic import Field, model_validator

from .common import ContractModel
from .query_rewrite import QueryRewriteTrace

RetrievalStatus = Literal[
    "success",
    "no_evidence",
]


class EvidenceItem(ContractModel):
    chunk_id: str = Field(min_length=1, max_length=300)
    source: str = Field(min_length=1, max_length=1000)
    text: str = Field(min_length=1, max_length=20000)
    # This is a retrieval ranking score, not answer confidence.
    score: float

    @model_validator(mode="after")
    def validate_score(self):
        if not math.isfinite(self.score):
            raise ValueError("Retrieval score must be finite.")
        return self

    rank: int = Field(ge=1)


class RetrievalResult(ContractModel):
    status: RetrievalStatus
    query: str = Field(min_length=1, max_length=4000)
    results: list[EvidenceItem] = Field(default_factory=list)
    index_version: str | None = Field(default=None, max_length=200)
    # Internal transport only; exposed at AnalyzeResponse level.
    query_rewrite_trace: QueryRewriteTrace | None = Field(
        default=None, exclude=True
    )

    @model_validator(mode="after")
    def validate_result(self):
        if self.status == "success" and not self.results:
            raise ValueError("success requires at least one evidence item.")

        if self.status == "no_evidence" and self.results:
            raise ValueError("no_evidence requires an empty results list.")

        chunk_ids = [item.chunk_id for item in self.results]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("Retrieval results must not contain duplicate chunk_id values.")

        ranks = [item.rank for item in self.results]
        expected_ranks = list(range(1, len(self.results) + 1))
        if ranks != expected_ranks:
            raise ValueError(
                "Retrieval result ranks must be unique, ordered, and consecutive from 1."
            )

        return self
