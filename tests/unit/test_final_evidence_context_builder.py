from datetime import UTC, datetime

import pytest

from smart_waste_multimodal.contracts.conversation import Conversation
from smart_waste_multimodal.contracts.evidence import (
    Evidence,
    EvidenceProvenance,
    EvidenceSourceType,
)
from smart_waste_multimodal.contracts.final_decision import (
    FinalDecisionResult,
)
from smart_waste_multimodal.evidence import (
    ConversationScopedInMemoryEvidenceRegistryFactory,
    EvidenceContextIntegrityError,
    EvidenceContextUnavailableError,
    FinalEvidenceContextBuilder,
)

NOW = datetime.now(UTC)
CONVERSATION_ID = "conversation-001"
OWNER_ID = "test-owner"


def make_evidence(
    evidence_id,
    request_id="request-001",
    conversation_id=CONVERSATION_ID,
):
    return Evidence(
        evidence_id=evidence_id,
        request_id=request_id,
        source_type=EvidenceSourceType.CONVERSATION_TEXT,
        provenance=EvidenceProvenance(source_id=f"source-{evidence_id}"),
        content={"text": f"content-{evidence_id}"},
        created_at=NOW,
        conversation_id=conversation_id,
        created_by_request_id=request_id,
        origin_turn_id="turn-001",
    )


def make_decision(evidence_ids=None, action="DIRECT", status="DECIDED"):
    if evidence_ids is None:
        evidence_ids = ["e-001"]

    reason_codes = {
        "DIRECT": "SUFFICIENT_EVIDENCE",
        "INSUFFICIENT_EVIDENCE": "INSUFFICIENT_EVIDENCE",
        "HUMAN_REVIEW": "HUMAN_ACTION_REQUIRED",
    }

    values = {
        "execution_status": status,
        "action": action,
        "reason_code": (
            "DECISION_DEPENDENCY_UNAVAILABLE"
            if status == "UNAVAILABLE"
            else reason_codes[action]
        ),
        "reason": "Test decision",
        "evidence_ids": evidence_ids,
    }

    if action == "HUMAN_REVIEW":
        values["clarification_required"] = True
        values["clarification_reason"] = "Human clarification required."

    if status == "UNAVAILABLE":
        values["action"] = None
        values["evidence_ids"] = []

    return FinalDecisionResult(**values)


class TestRegistryAdapter:
    __test__ = False
    """Adapt the conversation registry to concise test helpers."""

    def __init__(self, registry):
        self._registry = registry

    def __getattr__(self, name):
        return getattr(self._registry, name)

    def register(self, evidence):
        return self._registry.register(
            evidence,
            actor_id=OWNER_ID,
        )


def make_registry(conversation_id=CONVERSATION_ID):
    conversation = Conversation(
        conversation_id=conversation_id,
        owner_id=OWNER_ID,
        created_at=NOW,
        updated_at=NOW,
    )

    factory = ConversationScopedInMemoryEvidenceRegistryFactory()
    registry = factory.create(
        conversation,
        actor_id=OWNER_ID,
    )
    return TestRegistryAdapter(registry)


class FakeRegistry:
    def __init__(
        self,
        conversation_id=CONVERSATION_ID,
        records=None,
        error=None,
    ):
        self.conversation_id = conversation_id
        self.owner_id = OWNER_ID
        self.status = "ACTIVE"
        self.records = records or []
        self.error = error
        self.requested_ids = None

    def get_many(self, evidence_ids, *, actor_id):
        if actor_id != self.owner_id:
            raise PermissionError("Unexpected registry actor.")

        self.requested_ids = list(evidence_ids)

        if self.error:
            raise self.error

        return self.records


def test_builds_context_from_exact_authorized_evidence():
    registry = make_registry()
    registry.register(make_evidence("e-001"))
    registry.register(make_evidence("e-002"))

    decision = make_decision(["e-002", "e-001"])

    result = FinalEvidenceContextBuilder().build(
        request_id="request-001",
        conversation_id=CONVERSATION_ID,
        final_decision=decision,
        evidence_registry=registry,
    )

    assert result.request_id == "request-001"
    assert result.conversation_id == CONVERSATION_ID
    assert result.evidence_ids == ("e-002", "e-001")
    assert [item.evidence_id for item in result.evidence] == [
        "e-002",
        "e-001",
    ]


@pytest.mark.parametrize(
    ("status", "action"),
    [
        ("DECIDED", "INSUFFICIENT_EVIDENCE"),
        ("DECIDED", "HUMAN_REVIEW"),
        ("UNAVAILABLE", None),
    ],
)
def test_rejects_non_direct_decisions(status, action):
    decision = make_decision(action=action, status=status)
    registry = make_registry()

    with pytest.raises(EvidenceContextUnavailableError):
        FinalEvidenceContextBuilder().build(
            request_id="request-001",
            conversation_id=CONVERSATION_ID,
            final_decision=decision,
            evidence_registry=registry,
        )


def test_rejects_registry_from_another_conversation():
    registry = make_registry(conversation_id="conversation-002")
    decision = make_decision()

    with pytest.raises(EvidenceContextIntegrityError):
        FinalEvidenceContextBuilder().build(
            request_id="request-001",
            conversation_id=CONVERSATION_ID,
            final_decision=decision,
            evidence_registry=registry,
        )


def test_rejects_missing_authorized_evidence():
    registry = make_registry()
    registry.register(make_evidence("e-001"))
    decision = make_decision(["e-001", "e-002"])

    with pytest.raises(EvidenceContextIntegrityError):
        FinalEvidenceContextBuilder().build(
            request_id="request-001",
            conversation_id=CONVERSATION_ID,
            final_decision=decision,
            evidence_registry=registry,
        )


def test_rejects_wrong_registry_return_order():
    records = [make_evidence("e-002"), make_evidence("e-001")]
    registry = FakeRegistry(records=records)
    decision = make_decision(["e-001", "e-002"])

    with pytest.raises(EvidenceContextIntegrityError):
        FinalEvidenceContextBuilder().build(
            request_id="request-001",
            conversation_id=CONVERSATION_ID,
            final_decision=decision,
            evidence_registry=registry,
        )


def test_rejects_evidence_from_another_conversation():
    registry = FakeRegistry(
        records=[
            make_evidence(
                "e-001",
                conversation_id="conversation-002",
            )
        ],
    )
    decision = make_decision()

    with pytest.raises(EvidenceContextIntegrityError):
        FinalEvidenceContextBuilder().build(
            request_id="request-001",
            conversation_id=CONVERSATION_ID,
            final_decision=decision,
            evidence_registry=registry,
        )


def test_accepts_evidence_created_by_another_request_in_same_conversation():
    historical_evidence = make_evidence(
        "e-001",
        request_id="request-previous",
    )
    registry = FakeRegistry(records=[historical_evidence])

    result = FinalEvidenceContextBuilder().build(
        request_id="request-current",
        conversation_id=CONVERSATION_ID,
        final_decision=make_decision(),
        evidence_registry=registry,
    )

    assert result.request_id == "request-current"
    assert result.conversation_id == CONVERSATION_ID
    assert result.evidence[0].created_by_request_id == "request-previous"


def test_registry_failure_is_reported_as_unavailable():
    registry = FakeRegistry(
        error=RuntimeError("storage failure"),
    )

    with pytest.raises(EvidenceContextUnavailableError):
        FinalEvidenceContextBuilder().build(
            request_id="request-001",
            conversation_id=CONVERSATION_ID,
            final_decision=make_decision(),
            evidence_registry=registry,
        )


def test_builder_requests_only_final_decision_evidence_ids():
    registry = FakeRegistry(
        records=[make_evidence("e-002"), make_evidence("e-001")],
    )
    decision = make_decision(["e-002", "e-001"])

    FinalEvidenceContextBuilder().build(
        request_id="request-001",
        conversation_id=CONVERSATION_ID,
        final_decision=decision,
        evidence_registry=registry,
    )

    assert registry.requested_ids == ["e-002", "e-001"]
