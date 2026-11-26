"""Unit tests for Final Decision contracts."""

import pytest
from pydantic import ValidationError

from smart_waste_multimodal.contracts.final_decision import (
    DecisionTraceEntry,
    FinalDecisionResult,
)


def test_direct_decision_accepts_valid_evidence_ids():
    result = FinalDecisionResult(
        execution_status="DECIDED",
        action="DIRECT",
        reason_code="SUFFICIENT_EVIDENCE",
        reason="The available vision evidence is sufficient for this request.",
        evidence_ids=["vision:artifact-1:snapshot"],
    )
    assert result.action == "DIRECT"
    assert result.execution_status == "DECIDED"


def test_direct_decision_requires_evidence():
    with pytest.raises(ValidationError):
        FinalDecisionResult(
            execution_status="DECIDED",
            action="DIRECT",
            reason_code="SUFFICIENT_EVIDENCE",
            reason="Evidence is sufficient.",
        )


def test_insufficient_evidence_can_retain_relevant_evidence_ids():
    result = FinalDecisionResult(
        execution_status="DECIDED",
        action="INSUFFICIENT_EVIDENCE",
        reason_code="INSUFFICIENT_EVIDENCE",
        reason="Available evidence does not adequately support an answer.",
        evidence_ids=["retrieval:index-1:1:0001:chunk-1"],
    )
    assert result.evidence_ids
    assert not result.clarification_required


def test_insufficient_evidence_can_have_no_evidence_ids():
    result = FinalDecisionResult(
        execution_status="DECIDED",
        action="INSUFFICIENT_EVIDENCE",
        reason_code="INSUFFICIENT_EVIDENCE",
        reason="No usable evidence is available.",
    )
    assert result.evidence_ids == []


def test_human_review_requires_explicit_clarification_reason():
    result = FinalDecisionResult(
        execution_status="DECIDED",
        action="HUMAN_REVIEW",
        reason_code="HUMAN_ACTION_REQUIRED",
        reason="A human decision is required before proceeding.",
        clarification_required=True,
        clarification_reason="Please confirm the intended disposal jurisdiction.",
    )
    assert result.clarification_required


def test_human_review_rejects_missing_clarification_reason():
    with pytest.raises(ValidationError):
        FinalDecisionResult(
            execution_status="DECIDED",
            action="HUMAN_REVIEW",
            reason_code="HUMAN_ACTION_REQUIRED",
            reason="A human decision is required.",
            clarification_required=True,
        )


def test_non_human_action_rejects_clarification_request():
    with pytest.raises(ValidationError):
        FinalDecisionResult(
            execution_status="DECIDED",
            action="INSUFFICIENT_EVIDENCE",
            reason_code="INSUFFICIENT_EVIDENCE",
            reason="Evidence is insufficient.",
            clarification_required=True,
            clarification_reason="Please provide more information.",
        )


def test_unavailable_is_not_a_fourth_decision_action():
    result = FinalDecisionResult(
        execution_status="UNAVAILABLE",
        action=None,
        reason_code="DECISION_DEPENDENCY_UNAVAILABLE",
        reason="The evidence registry could not be accessed safely.",
    )
    assert result.action is None
    assert result.execution_status == "UNAVAILABLE"


def test_unavailable_rejects_decision_action():
    with pytest.raises(ValidationError):
        FinalDecisionResult(
            execution_status="UNAVAILABLE",
            action="INSUFFICIENT_EVIDENCE",
            reason_code="DECISION_DEPENDENCY_UNAVAILABLE",
            reason="Decision dependencies are unavailable.",
        )


def test_unavailable_rejects_evidence_ids():
    with pytest.raises(ValidationError):
        FinalDecisionResult(
            execution_status="UNAVAILABLE",
            action=None,
            reason_code="DECISION_DEPENDENCY_UNAVAILABLE",
            reason="Registry access failed.",
            evidence_ids=["vision:artifact-1:snapshot"],
        )


def test_decision_rejects_duplicate_evidence_ids():
    with pytest.raises(ValidationError):
        FinalDecisionResult(
            execution_status="DECIDED",
            action="DIRECT",
            reason_code="SUFFICIENT_EVIDENCE",
            reason="Evidence is sufficient.",
            evidence_ids=["evidence-1", "evidence-1"],
        )


def test_decision_rejects_inconsistent_reason_code():
    with pytest.raises(ValidationError):
        FinalDecisionResult(
            execution_status="DECIDED",
            action="DIRECT",
            reason_code="INSUFFICIENT_EVIDENCE",
            reason="Evidence is sufficient.",
            evidence_ids=["evidence-1"],
        )


def test_decision_trace_is_structured_audit_metadata():
    entry = DecisionTraceEntry(
        stage="evidence_validation",
        outcome="completed",
        reason_code="VALID_EVIDENCE_SET",
        evidence_ids=["evidence-1"],
    )
    result = FinalDecisionResult(
        execution_status="DECIDED",
        action="DIRECT",
        reason_code="SUFFICIENT_EVIDENCE",
        reason="The available evidence is sufficient.",
        evidence_ids=["evidence-1"],
        decision_trace=[entry],
    )
    assert result.decision_trace[0].stage == "evidence_validation"


def test_trace_rejects_duplicate_evidence_ids():
    with pytest.raises(ValidationError):
        DecisionTraceEntry(
            stage="evidence_validation",
            outcome="completed",
            evidence_ids=["evidence-1", "evidence-1"],
        )


def test_trace_rejects_blank_stage():
    with pytest.raises(ValidationError):
        DecisionTraceEntry(stage="  ", outcome="completed")
