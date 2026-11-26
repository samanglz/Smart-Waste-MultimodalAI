from datetime import UTC, datetime, timedelta

import pytest

from smart_waste_multimodal.contracts.conversation import Conversation
from smart_waste_multimodal.contracts.evidence import (
    Evidence,
    EvidenceProvenance,
    EvidenceSourceType,
)
from smart_waste_multimodal.evidence import (
    ConversationClosedError,
    ConversationNotFoundError,
    ConversationOwnershipError,
    ConversationScopedInMemoryEvidenceRegistryFactory,
    DuplicateEvidenceError,
    EvidenceConversationMismatchError,
)


def make_conversation(
    conversation_id="conversation-001",
    owner_id="owner-001",
    created_at=None,
):
    created_at = created_at or datetime.now(UTC) - timedelta(minutes=5)
    return Conversation(
        conversation_id=conversation_id,
        owner_id=owner_id,
        status="active",
        created_at=created_at,
        updated_at=created_at,
    )


def make_evidence(
    evidence_id="evidence-001",
    conversation_id="conversation-001",
    request_id="request-001",
    created_by_request_id="request-001",
    origin_turn_id="turn-001",
):
    return Evidence(
        evidence_id=evidence_id,
        request_id=request_id,
        conversation_id=conversation_id,
        created_by_request_id=created_by_request_id,
        origin_turn_id=origin_turn_id,
        source_type=EvidenceSourceType.CONVERSATION_TEXT,
        provenance=EvidenceProvenance(
            source_id=origin_turn_id,
            source_timestamp=datetime.now(UTC),
        ),
        content={"text": "Original evidence"},
        created_at=datetime.now(UTC),
        metadata={"test": True},
    )


def test_factory_returns_same_registry_for_same_conversation():
    factory = ConversationScopedInMemoryEvidenceRegistryFactory()
    conversation = make_conversation()

    first = factory.create(conversation, actor_id="owner-001")
    second = factory.create(conversation, actor_id="owner-001")

    assert first is second


def test_factory_creates_isolated_registries_for_different_conversations():
    factory = ConversationScopedInMemoryEvidenceRegistryFactory()

    first = factory.create(
        make_conversation("conversation-001"),
        actor_id="owner-001",
    )
    second = factory.create(
        make_conversation("conversation-002"),
        actor_id="owner-001",
    )

    first.register(make_evidence(), actor_id="owner-001")

    assert first.get("evidence-001", actor_id="owner-001") is not None
    assert second.get("evidence-001", actor_id="owner-001") is None


def test_registry_rejects_evidence_from_another_conversation():
    factory = ConversationScopedInMemoryEvidenceRegistryFactory()
    registry = factory.create(make_conversation(), actor_id="owner-001")

    with pytest.raises(EvidenceConversationMismatchError):
        registry.register(
            make_evidence(conversation_id="conversation-002"),
            actor_id="owner-001",
        )


def test_evidence_is_reusable_across_requests_without_provenance_rewrite():
    factory = ConversationScopedInMemoryEvidenceRegistryFactory()
    conversation = make_conversation()

    first_request_registry = factory.create(conversation, actor_id="owner-001")
    original = make_evidence(
        request_id="request-001",
        created_by_request_id="request-001",
        origin_turn_id="turn-001",
    )
    first_request_registry.register(original, actor_id="owner-001")

    # A later request obtains the same conversation registry.
    later_request_registry = factory.create(conversation, actor_id="owner-001")
    reused = later_request_registry.get("evidence-001", actor_id="owner-001")

    assert later_request_registry is first_request_registry
    assert reused is not None
    assert reused.request_id == "request-001"
    assert reused.created_by_request_id == "request-001"
    assert reused.origin_turn_id == "turn-001"


def test_registry_does_not_rewrite_original_creator_request_id():
    factory = ConversationScopedInMemoryEvidenceRegistryFactory()
    registry = factory.create(make_conversation(), actor_id="owner-001")

    evidence = make_evidence(
        request_id="legacy-consumer-request",
        created_by_request_id="original-producing-request",
    )
    registry.register(evidence, actor_id="owner-001")

    restored = registry.get("evidence-001", actor_id="owner-001")
    assert restored is not None
    assert restored.created_by_request_id == "original-producing-request"
    assert restored.request_id == "legacy-consumer-request"


def test_registry_enforces_owner_on_all_access():
    factory = ConversationScopedInMemoryEvidenceRegistryFactory()
    registry = factory.create(make_conversation(), actor_id="owner-001")
    registry.register(make_evidence(), actor_id="owner-001")

    with pytest.raises(ConversationOwnershipError):
        registry.get("evidence-001", actor_id="another-owner")

    with pytest.raises(ConversationOwnershipError):
        registry.register(
            make_evidence(evidence_id="evidence-002"),
            actor_id="another-owner",
        )

    with pytest.raises(ConversationOwnershipError):
        factory.get("conversation-001", actor_id="another-owner")


def test_factory_rejects_non_owner_during_creation():
    factory = ConversationScopedInMemoryEvidenceRegistryFactory()

    with pytest.raises(ConversationOwnershipError):
        factory.create(make_conversation(), actor_id="another-owner")


def test_same_conversation_id_cannot_be_rebound_to_another_owner():
    factory = ConversationScopedInMemoryEvidenceRegistryFactory()
    factory.create(make_conversation(), actor_id="owner-001")

    conflicting = make_conversation(owner_id="owner-002")
    with pytest.raises(ConversationOwnershipError):
        factory.create(conflicting, actor_id="owner-002")


def test_register_rejects_duplicate_evidence_ids():
    factory = ConversationScopedInMemoryEvidenceRegistryFactory()
    registry = factory.create(make_conversation(), actor_id="owner-001")

    registry.register(make_evidence(), actor_id="owner-001")

    with pytest.raises(DuplicateEvidenceError):
        registry.register(
            make_evidence(),
            actor_id="owner-001",
        )


def test_get_many_preserves_order_and_omits_unknown_ids():
    factory = ConversationScopedInMemoryEvidenceRegistryFactory()
    registry = factory.create(make_conversation(), actor_id="owner-001")

    registry.register(make_evidence(evidence_id="e-002"), actor_id="owner-001")
    registry.register(make_evidence(evidence_id="e-001"), actor_id="owner-001")

    results = registry.get_many(["e-002", "missing", "e-001"], actor_id="owner-001")
    assert [item.evidence_id for item in results] == ["e-002", "e-001"]


def test_list_all_is_sorted_and_list_by_turn_is_scoped():
    factory = ConversationScopedInMemoryEvidenceRegistryFactory()
    registry = factory.create(make_conversation(), actor_id="owner-001")

    registry.register(
        make_evidence("e-003", origin_turn_id="turn-002"),
        actor_id="owner-001",
    )
    registry.register(
        make_evidence("e-001", origin_turn_id="turn-001"),
        actor_id="owner-001",
    )
    registry.register(
        make_evidence("e-002", origin_turn_id="turn-001"),
        actor_id="owner-001",
    )

    assert [item.evidence_id for item in registry.list_all(actor_id="owner-001")] == [
        "e-001",
        "e-002",
        "e-003",
    ]

    assert [
        item.evidence_id
        for item in registry.list_by_turn("turn-001", actor_id="owner-001")
    ] == ["e-001", "e-002"]


def test_registry_snapshot_isolated_from_input_and_returned_mutations():
    factory = ConversationScopedInMemoryEvidenceRegistryFactory()
    registry = factory.create(make_conversation(), actor_id="owner-001")
    evidence = make_evidence()

    registry.register(evidence, actor_id="owner-001")
    evidence.content["text"] = "mutated input"

    returned = registry.get("evidence-001", actor_id="owner-001")
    assert returned is not None
    returned.content["text"] = "mutated returned object"

    again = registry.get("evidence-001", actor_id="owner-001")
    assert again is not None
    assert again.content["text"] == "Original evidence"


def test_close_blocks_register_and_read_access():
    factory = ConversationScopedInMemoryEvidenceRegistryFactory()
    registry = factory.create(make_conversation(), actor_id="owner-001")
    registry.register(make_evidence(), actor_id="owner-001")

    factory.close_conversation("conversation-001", actor_id="owner-001")

    assert registry.status == "closed"

    with pytest.raises(ConversationClosedError):
        registry.get("evidence-001", actor_id="owner-001")

    with pytest.raises(ConversationClosedError):
        registry.register(
            make_evidence(evidence_id="evidence-002"),
            actor_id="owner-001",
        )

    with pytest.raises(ConversationClosedError):
        factory.create(make_conversation(), actor_id="owner-001")


def test_close_rejects_naive_timestamp():
    factory = ConversationScopedInMemoryEvidenceRegistryFactory()
    factory.create(make_conversation(), actor_id="owner-001")

    with pytest.raises(ValueError):
        factory.close_conversation(
            "conversation-001",
            actor_id="owner-001",
            closed_at=datetime(2026, 1, 1),
        )


def test_delete_removes_registry_and_all_evidence():
    factory = ConversationScopedInMemoryEvidenceRegistryFactory()
    registry = factory.create(make_conversation(), actor_id="owner-001")
    registry.register(make_evidence(), actor_id="owner-001")

    assert factory.delete_conversation("conversation-001", actor_id="owner-001") is True

    with pytest.raises(ConversationNotFoundError):
        factory.get("conversation-001", actor_id="owner-001")

    with pytest.raises(ConversationNotFoundError):
        registry.list_all(actor_id="owner-001")

    assert (
        factory.delete_conversation("conversation-001", actor_id="owner-001") is False
    )


def test_delete_requires_owner():
    factory = ConversationScopedInMemoryEvidenceRegistryFactory()
    factory.create(make_conversation(), actor_id="owner-001")

    with pytest.raises(ConversationOwnershipError):
        factory.delete_conversation("conversation-001", actor_id="another-owner")


def test_closed_conversation_cannot_be_created_from_closed_contract():
    created_at = datetime.now(UTC) - timedelta(minutes=5)
    closed_at = datetime.now(UTC)

    closed_conversation = Conversation(
        conversation_id="closed-conversation",
        owner_id="owner-001",
        status="closed",
        created_at=created_at,
        updated_at=closed_at,
        closed_at=closed_at,
    )

    factory = ConversationScopedInMemoryEvidenceRegistryFactory()
    with pytest.raises(ConversationClosedError):
        factory.create(closed_conversation, actor_id="owner-001")
