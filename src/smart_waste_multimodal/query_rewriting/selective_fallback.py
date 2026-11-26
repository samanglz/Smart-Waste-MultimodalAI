"""Conservative two-action policy for query rewriting."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from smart_waste_multimodal.contracts.context import (
    ContextAssessment,
    ContextStatus,
)


class RewriteAction(str, Enum):
    """Action selected by the query rewriting policy."""

    DIRECT = "direct"
    REWRITE = "rewrite"


@dataclass(frozen=True)
class RewritePolicyDecision:
    """Final policy decision."""

    action: RewriteAction
    reason: str
    signals: tuple[str, ...] = ()


class SelectiveRewritePolicy:
    """Choose whether a query should go direct or be rewritten.

    This policy runs before retrieval. It does not estimate retrieval
    quality and does not use retrieval results.

    Context sufficiency is supplied through ContextAssessment.
    Possible context-dependent queries without an assessment fail closed.
    """

    CONTEXT_DEPENDENT_MARKERS = (
        "what about",
        "how about",
        "same thing",
        "same material",
        "same item",
        "same one",
        "that one",
        "this one",
        "those items",
        "these items",
        "the former",
        "the latter",
        "as mentioned above",
        "as discussed",
    )

    @staticmethod
    def _normalize(query: str) -> str:
        """Normalize punctuation and whitespace for phrase matching."""
        normalized = query.lower()
        normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
        return " ".join(normalized.split())

    def _find_context_markers(
        self,
        query: str,
    ) -> tuple[str, ...]:
        normalized = self._normalize(query)

        if not normalized:
            return ()

        padded = f" {normalized} "

        return tuple(
            marker
            for marker in self.CONTEXT_DEPENDENT_MARKERS
            if f" {marker} " in padded
        )

    def decide(
        self,
        query: str,
        context_assessment: ContextAssessment | None = None,
    ) -> RewritePolicyDecision:
        if not isinstance(query, str):
            raise TypeError("query must be a string")

        if not self._normalize(query):
            raise ValueError("query must not be empty")

        if context_assessment is not None:
            if not isinstance(context_assessment, ContextAssessment):
                raise TypeError(
                    "context_assessment must be a ContextAssessment "
                    "instance or None"
                )

            status = context_assessment.status

            if not isinstance(status, ContextStatus):
                raise TypeError(
                    "context_assessment.status must be a ContextStatus"
                )

            if status == ContextStatus.INDEPENDENT:
                return RewritePolicyDecision(
                    action=RewriteAction.DIRECT,
                    reason="independent_query",
                    signals=context_assessment.signals,
                )

            if status == ContextStatus.RESOLVED:
                return RewritePolicyDecision(
                    action=RewriteAction.REWRITE,
                    reason="context_dependency_resolved",
                    signals=context_assessment.signals,
                )

            raise ValueError(
                "Context must be resolved before selective query rewriting; "
                f"received status={status.value!r}."
            )

        # Lexical markers cannot resolve references. Do not send an
        # ambiguous query to retrieval or ask the rewriting model to guess.
        matched_markers = self._find_context_markers(query)

        if matched_markers:
            raise ValueError(
                "Context assessment is required for a potentially "
                "context-dependent query; matched markers: "
                + ", ".join(matched_markers)
            )

        return RewritePolicyDecision(
            action=RewriteAction.DIRECT,
            reason="no_context_dependency_detected",
            signals=(),
        )
