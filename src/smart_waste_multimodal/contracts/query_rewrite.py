"""Typed observability contract for query rewriting."""

from typing import Literal

from pydantic import Field

from .common import ContractModel

QueryRewriteAction = Literal[
    "direct",
    "rewritten",
    "rewrite_unchanged",
    "fallback_original",
    "natural_only",
    "bypassed",
]

QueryRewriteFailureCategory = Literal[
    "provider_failure",
    "invalid_response",
    "rewriter_unavailable",
]


class QueryRewriteTrace(ContractModel):
    """Safe, request-scoped metadata describing query rewrite execution."""

    action: QueryRewriteAction
    reason: str = Field(min_length=1, max_length=200)
    original_query: str | None = Field(default=None, max_length=4000)
    retrieval_query: str = Field(min_length=1, max_length=4000)
    duration_ms: float = Field(ge=0)
    failure_category: QueryRewriteFailureCategory | None = None
    failure_type: str | None = Field(default=None, min_length=1, max_length=200)
