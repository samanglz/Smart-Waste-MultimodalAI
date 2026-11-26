"""Contracts for the final evidence-based decision."""

from typing import Literal

from pydantic import Field, model_validator

from .common import ContractModel

DecisionExecutionStatus = Literal["DECIDED", "UNAVAILABLE"]

DecisionAction = Literal[
    "DIRECT",
    "INSUFFICIENT_EVIDENCE",
    "HUMAN_REVIEW",
]

DecisionReasonCode = Literal[
    "SUFFICIENT_EVIDENCE",
    "INSUFFICIENT_EVIDENCE",
    "HUMAN_ACTION_REQUIRED",
    "DECISION_DEPENDENCY_UNAVAILABLE",
]


class DecisionTraceEntry(ContractModel):
    """Auditable decision metadata; never a model chain-of-thought."""

    stage: str = Field(min_length=1, max_length=100)
    outcome: str = Field(min_length=1, max_length=100)
    reason_code: str | None = Field(default=None, max_length=100)
    evidence_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_trace_entry(self):
        if not self.stage.strip() or not self.outcome.strip():
            raise ValueError("Trace stage and outcome must not be blank.")

        if any(not item.strip() for item in self.evidence_ids):
            raise ValueError("Trace evidence IDs must not be blank.")

        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("Trace evidence IDs must not contain duplicates.")

        return self


class FinalDecisionResult(ContractModel):
    """Final pipeline decision and its auditable outcome."""

    execution_status: DecisionExecutionStatus
    action: DecisionAction | None = None
    reason_code: DecisionReasonCode
    reason: str = Field(min_length=1, max_length=1000)
    evidence_ids: list[str] = Field(default_factory=list)
    clarification_required: bool = False
    clarification_reason: str | None = Field(default=None, max_length=1000)
    decision_trace: list[DecisionTraceEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_decision_consistency(self):
        if not self.reason.strip():
            raise ValueError("Decision reason must not be blank.")

        if any(not item.strip() for item in self.evidence_ids):
            raise ValueError("Evidence IDs must not be blank.")

        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("Evidence IDs must not contain duplicates.")

        if self.execution_status == "UNAVAILABLE":
            if self.action is not None:
                raise ValueError(
                    "UNAVAILABLE decisions must not contain an action."
                )
            if self.reason_code != "DECISION_DEPENDENCY_UNAVAILABLE":
                raise ValueError(
                    "UNAVAILABLE requires DECISION_DEPENDENCY_UNAVAILABLE."
                )
            if self.evidence_ids:
                raise ValueError(
                    "UNAVAILABLE decisions must not expose evidence IDs."
                )
            if self.clarification_required or self.clarification_reason:
                raise ValueError(
                    "UNAVAILABLE must not request user clarification."
                )
            return self

        if self.action is None:
            raise ValueError("DECIDED requires a decision action.")

        expected_reason_codes = {
            "DIRECT": "SUFFICIENT_EVIDENCE",
            "INSUFFICIENT_EVIDENCE": "INSUFFICIENT_EVIDENCE",
            "HUMAN_REVIEW": "HUMAN_ACTION_REQUIRED",
        }
        expected_code = expected_reason_codes[self.action]

        if self.reason_code != expected_code:
            raise ValueError(
                f"{self.action} requires reason code {expected_code}."
            )

        if self.action == "DIRECT" and not self.evidence_ids:
            raise ValueError("DIRECT requires at least one evidence ID.")

        if self.action == "HUMAN_REVIEW":
            if not self.clarification_required:
                raise ValueError(
                    "HUMAN_REVIEW requires clarification_required=True."
                )
            if (
                not self.clarification_reason
                or not self.clarification_reason.strip()
            ):
                raise ValueError(
                    "HUMAN_REVIEW requires a non-empty clarification reason."
                )
        else:
            if self.clarification_required or self.clarification_reason:
                raise ValueError(
                    "Only HUMAN_REVIEW may request clarification."
                )

        return self
