"""Contracts for semantic conversation-context analysis."""

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from pydantic import Field, model_validator

from .common import ContractModel
from .evidence import EvidenceSourceType


class ContextStatus(str, Enum):
    """Validated status of a query's dependency on conversation context."""

    INDEPENDENT = "independent"
    RESOLVED = "resolved"
    INSUFFICIENT = "insufficient"
    AMBIGUOUS = "ambiguous"
    CONFLICT = "conflict"
    STALE = "stale"


@dataclass(frozen=True)
class ContextAssessment:
    """Validated context-analysis result supplied to downstream stages."""

    status: ContextStatus
    signals: tuple[str, ...] = ()
    referenced_turn_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    evidence_turn_ids: tuple[str, ...] = ()
    resolved_context: str | None = None
    reason: str = ""
    analysis_error: bool = False


ContextAnalysisStatus = Literal[
    "independent", "resolved", "insufficient", "ambiguous", "conflict"
]


class ContextAnalysisOutput(ContractModel):
    """Semantic output expected from the context analyzer.

    STALE is determined by Python after applying the configured TTL.
    """

    status: ContextAnalysisStatus
    referenced_turn_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    required_evidence_types: list[EvidenceSourceType] = Field(
        default_factory=list
    )
    resolved_context: str | None = None
    signals: list[str] = Field(default_factory=list)
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_semantic_fields(self):
        if len(self.referenced_turn_ids) != len(set(self.referenced_turn_ids)):
            raise ValueError("referenced_turn_ids must be unique.")

        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("evidence_ids must be unique.")

        if len(self.required_evidence_types) != len(
            set(self.required_evidence_types)
        ):
            raise ValueError("required_evidence_types must be unique.")

        if self.status == "independent":
            if self.resolved_context is not None:
                raise ValueError(
                    "Independent analysis must not contain resolved_context."
                )
            if self.evidence_ids or self.required_evidence_types:
                raise ValueError(
                    "Independent analysis must not require evidence."
                )

        elif self.status == "resolved":
            if not self.resolved_context or not self.resolved_context.strip():
                raise ValueError(
                    "Resolved analysis requires non-empty resolved_context."
                )
            if not self.evidence_ids:
                raise ValueError(
                    "Resolved analysis requires supporting evidence."
                )
            if not self.required_evidence_types:
                raise ValueError(
                    "Resolved analysis requires evidence types."
                )

        else:
            if self.resolved_context is not None:
                raise ValueError(
                    "Unresolved analysis must not contain resolved_context."
                )
            if self.evidence_ids or self.required_evidence_types:
                raise ValueError(
                    "Unresolved analysis must not contain evidence."
                )

        return self
