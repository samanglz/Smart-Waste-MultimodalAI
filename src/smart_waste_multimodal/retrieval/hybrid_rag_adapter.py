"""Adapter connecting the legacy hybrid retriever to pipeline contracts."""

from collections.abc import Callable
from typing import Any, Protocol

from smart_waste_multimodal.contracts import (
    AnalyzeRequest,
    EvidenceItem,
    QueryRewriteTrace,
    RetrievalResult,
    VisionResult,
)
from smart_waste_multimodal.contracts.context import ContextAssessment


class HybridSearch(Protocol):
    """Minimal interface required from a hybrid retriever."""

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]: ...


class QueryRewrite(Protocol):
    """Minimal interface required from the query rewriting service."""

    def rewrite_for_retrieval(self, original_query: str, context_assessment: ContextAssessment | None = None) -> Any: ...


class HybridRAGAdapter:
    """Adapt a hybrid retriever and optionally rewrite question-based queries.

    Rewriting is opt-in. Detection-derived queries always bypass query rewriting.
    """

    def __init__(
        self,
        *,
        retriever: HybridSearch | Callable[..., list[dict[str, Any]]],
        top_k: int = 5,
        index_version: str | None = None,
        rewrite_service: QueryRewrite | None = None,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        self.retriever = retriever
        self.top_k = top_k
        self.index_version = index_version
        self.rewrite_service = rewrite_service

    @staticmethod
    def _build_detection_query(
        vision: VisionResult | None,
    ) -> str:
        if vision is None:
            raise ValueError(
                "Vision results are required for detection-based retrieval."
            )

        class_names = list(
            dict.fromkeys(
                detection.class_name.strip()
                for detection in vision.detections
                if detection.class_name.strip()
            )
        )

        if not class_names:
            raise ValueError(
                "Detection-based retrieval requires at least one detection."
            )

        return " ".join(class_names)

    def __call__(
        self,
        request: AnalyzeRequest,
        vision: VisionResult | None,
        context_assessment: ContextAssessment | None = None,
    ) -> RetrievalResult:
        """Run retrieval and translate results into stable contracts."""
        query_rewrite_trace = None
        if request.options.retrieval_query_source == "question":
            original_query = (request.question or "").strip()
            if not original_query:
                raise ValueError(
                    "A non-empty question is required for question-based retrieval."
                )

            query = original_query

            if self.rewrite_service is None:
                query_rewrite_trace = QueryRewriteTrace(
                    action="bypassed",
                    reason="rewriter_not_configured",
                    original_query=original_query,
                    retrieval_query=original_query,
                    duration_ms=0.0,
                )
            else:
                outcome = self.rewrite_service.rewrite_for_retrieval(
                    original_query,
                    context_assessment=context_assessment,
                )

                action = getattr(outcome, "action", None)
                allowed_actions = {
                    "direct",
                    "rewritten",
                    "rewrite_unchanged",
                    "fallback_original",
                    "natural_only",
                }

                if action not in allowed_actions:
                    raise ValueError(
                        f"Invalid query rewrite action: {action!r}"
                    )

                outcome_original = getattr(outcome, "original_query", None)
                if outcome_original != original_query:
                    raise ValueError(
                        "Query rewrite outcome does not preserve the original query."
                    )

                candidate_query = getattr(outcome, "retrieval_query", None)

                if (
                    not isinstance(candidate_query, str)
                    or not candidate_query.strip()
                ):
                    raise ValueError(
                        "Query rewrite outcome contains an empty or invalid "
                        "retrieval_query."
                    )

                query = candidate_query.strip()

                reason_value = getattr(outcome, "reason", None)
                if not isinstance(reason_value, str) or not reason_value.strip():
                    reason_value = "unspecified"

                duration_value = getattr(outcome, "duration_ms", None)
                if (
                    isinstance(duration_value, bool)
                    or not isinstance(duration_value, (int, float))
                    or not (0 <= duration_value < float("inf"))
                ):
                    duration_value = 0.0

                failure_category_value = getattr(
                    outcome, "failure_category", None
                )
                if failure_category_value not in {
                    "provider_failure",
                    "invalid_response",
                    "rewriter_unavailable",
                }:
                    failure_category_value = None

                failure_type_value = getattr(outcome, "failure_type", None)
                if (
                    not isinstance(failure_type_value, str)
                    or not failure_type_value.strip()
                ):
                    failure_type_value = None

                query_rewrite_trace = QueryRewriteTrace(
                    action=action,
                    reason=reason_value,
                    original_query=original_query,
                    retrieval_query=query,
                    duration_ms=float(duration_value),
                    failure_category=failure_category_value,
                    failure_type=failure_type_value,
                )

        else:
            query = self._build_detection_query(vision)
            query_rewrite_trace = QueryRewriteTrace(
                action="bypassed",
                reason="detection_based_retrieval",
                original_query=None,
                retrieval_query=query,
                duration_ms=0.0,
            )

        raw_results = self.retriever.search(
            query,
            top_k=self.top_k,
        )

        if not isinstance(raw_results, list):
            raise TypeError("Retriever output must be a list of evidence records.")

        if len(raw_results) > self.top_k:
            raise ValueError(
                f"Retriever returned {len(raw_results)} records, exceeding top_k={self.top_k}."
            )

        required_fields = {"chunk_id", "source", "text", "score", "rank"}
        evidence: list[EvidenceItem] = []

        for index, item in enumerate(raw_results):
            if not isinstance(item, dict):
                raise TypeError(
                    f"Retriever record at position {index} must be a dictionary."
                )

            missing_fields = required_fields.difference(item)
            if missing_fields:
                missing_names = ", ".join(sorted(missing_fields))
                raise ValueError(
                    f"Retriever record at position {index} is missing required fields: "
                    f"{missing_names}."
                )

            evidence.append(
                EvidenceItem(
                    chunk_id=item["chunk_id"],
                    source=item["source"],
                    text=item["text"],
                    score=item["score"],
                    rank=item["rank"],
                )
            )

        return RetrievalResult(
            status="success" if evidence else "no_evidence",
            query=query,
            results=evidence,
            index_version=self.index_version,
            query_rewrite_trace=query_rewrite_trace,
        )
