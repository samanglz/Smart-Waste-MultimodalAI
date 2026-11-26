from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from smart_waste_multimodal.config.context import ContextConfig
from smart_waste_multimodal.contracts.context import (
    ContextAnalysisOutput,
    ContextStatus,
)
from smart_waste_multimodal.contracts.conversation import (
    Conversation,
    ConversationTurn,
)
from smart_waste_multimodal.contracts.evidence import (
    Evidence,
    EvidenceProvenance,
    EvidenceSourceType,
)
from smart_waste_multimodal.evidence import (
    ConversationScopedInMemoryEvidenceRegistryFactory,
)
from smart_waste_multimodal.query_rewriting.context_validator import (
    ContextAnalysisValidator,
)

NOW = datetime(2030, 5, 10, 12, 0, tzinfo=UTC)
ACTOR_ID = "owner-001"
CONVERSATION_ID = "conversation-001"


def user_turn(turn_id="turn-1", timestamp=None, vision=None):
    return ConversationTurn(
        turn_id=turn_id,
        role="user",
        content="I have a glass bottle.",
        timestamp=timestamp or NOW - timedelta(minutes=5),
        vision=vision,
    )


def assistant_turn(turn_id="turn-1", timestamp=None):
    return ConversationTurn(
        turn_id=turn_id,
        role="assistant",
        content="The item is glass.",
        timestamp=timestamp or NOW - timedelta(minutes=1),
    )


def make_evidence(
    evidence_id="evidence-1",
    turn_id="turn-1",
    source_type=EvidenceSourceType.CONVERSATION_TEXT,
    created_at=None,
):
    created_at = created_at or NOW - timedelta(minutes=5)

    return Evidence(
        evidence_id=evidence_id,
        request_id="request-001",
        conversation_id=CONVERSATION_ID,
        created_by_request_id="request-001",
        origin_turn_id=turn_id,
        source_type=source_type,
        provenance=EvidenceProvenance(
            source_id=turn_id,
            source_timestamp=created_at,
        ),
        content={"text": "The item is a glass bottle."},
        created_at=created_at,
        metadata={"test": True},
    )


def make_registry(evidence_items=()):
    conversation = Conversation(
        conversation_id=CONVERSATION_ID,
        owner_id=ACTOR_ID,
        status="active",
        created_at=NOW - timedelta(minutes=10),
        updated_at=NOW - timedelta(minutes=10),
    )

    factory = ConversationScopedInMemoryEvidenceRegistryFactory()
    registry = factory.create(conversation, actor_id=ACTOR_ID)

    for evidence in evidence_items:
        registry.register(evidence, actor_id=ACTOR_ID)

    return registry


def resolved(
    turn_id="turn-1",
    evidence_id="evidence-1",
    required_types=None,
):
    if required_types is None:
        required_types = [EvidenceSourceType.CONVERSATION_TEXT]

    return ContextAnalysisOutput(
        status="resolved",
        referenced_turn_ids=[turn_id],
        evidence_ids=[evidence_id],
        required_evidence_types=required_types,
        resolved_context="The item is a glass bottle.",
        reason="The user identified the item.",
    )


def validate(
    output,
    turns,
    registry,
    validator=None,
    current_turn_id=None,
    current_turn_timestamp=None,
):
    return (validator or ContextAnalysisValidator()).validate(
        output,
        turns,
        registry,
        actor_id=ACTOR_ID,
        now=NOW,
        current_turn_id=current_turn_id,
        current_turn_timestamp=current_turn_timestamp,
    )


def test_current_turn_evidence_resolves_without_history_entry():
    current_turn_id = "turn-current"
    current_turn_timestamp = NOW - timedelta(seconds=30)

    evidence = make_evidence(
        evidence_id="evidence-current",
        turn_id=current_turn_id,
    )
    registry = make_registry([evidence])

    result = validate(
        resolved(
            turn_id=current_turn_id,
            evidence_id="evidence-current",
        ),
        [],
        registry,
        current_turn_id=current_turn_id,
        current_turn_timestamp=current_turn_timestamp,
    )

    assert result.status == ContextStatus.RESOLVED
    assert result.evidence_ids == ("evidence-current",)
    assert result.evidence_turn_ids == (current_turn_id,)
    assert result.resolved_context == "The item is a glass bottle."
    assert not result.analysis_error


@pytest.mark.parametrize(
    ("current_turn_id", "current_turn_timestamp"),
    [
        ("turn-current", None),
        (None, NOW - timedelta(seconds=30)),
    ],
)
def test_current_turn_metadata_must_be_provided_as_a_pair(
    current_turn_id,
    current_turn_timestamp,
):
    registry = make_registry()

    result = validate(
        ContextAnalysisOutput(
            status="independent",
            reason="Independent.",
        ),
        [],
        registry,
        current_turn_id=current_turn_id,
        current_turn_timestamp=current_turn_timestamp,
    )

    assert result.status == ContextStatus.INSUFFICIENT
    assert result.analysis_error


def test_empty_current_turn_id_fails_closed():
    registry = make_registry()

    result = validate(
        ContextAnalysisOutput(
            status="independent",
            reason="Independent.",
        ),
        [],
        registry,
        current_turn_id=" ",
        current_turn_timestamp=NOW - timedelta(seconds=30),
    )

    assert result.status == ContextStatus.INSUFFICIENT
    assert result.analysis_error


def test_current_turn_id_must_not_duplicate_history():
    turn = user_turn(turn_id="turn-current")
    registry = make_registry()

    result = validate(
        ContextAnalysisOutput(
            status="independent",
            reason="Independent.",
        ),
        [turn],
        registry,
        current_turn_id="turn-current",
        current_turn_timestamp=NOW,
    )

    assert result.status == ContextStatus.INSUFFICIENT
    assert result.analysis_error


def test_future_current_turn_timestamp_fails_closed():
    registry = make_registry()

    result = validate(
        ContextAnalysisOutput(
            status="independent",
            reason="Independent.",
        ),
        [],
        registry,
        current_turn_id="turn-current",
        current_turn_timestamp=NOW + timedelta(seconds=1),
    )

    assert result.status == ContextStatus.INSUFFICIENT
    assert result.analysis_error


def test_valid_content_evidence_resolves():
    evidence = make_evidence()
    registry = make_registry([evidence])

    result = validate(resolved(), [user_turn()], registry)

    assert result.status == ContextStatus.RESOLVED
    assert result.resolved_context == "The item is a glass bottle."
    assert result.evidence_ids == ("evidence-1",)
    assert result.evidence_turn_ids == ("turn-1",)
    assert not result.analysis_error


def test_registered_vision_evidence_resolves_without_turn_payload():
    evidence = make_evidence(
        source_type=EvidenceSourceType.VISION_DETECTION,
    )
    registry = make_registry([evidence])

    output = resolved(
        required_types=[EvidenceSourceType.VISION_DETECTION],
    )

    result = validate(output, [user_turn(vision=None)], registry)

    assert result.status == ContextStatus.RESOLVED
    assert result.evidence_ids == ("evidence-1",)
    assert not result.analysis_error


def test_unknown_referenced_id_fails_closed():
    output = ContextAnalysisOutput(
        status="independent",
        referenced_turn_ids=["fake"],
        reason="Independent.",
    )
    registry = make_registry()

    result = validate(output, [user_turn()], registry)

    assert result.analysis_error
    assert result.status == ContextStatus.INSUFFICIENT


def test_unknown_evidence_id_returns_insufficient():
    registry = make_registry()

    result = validate(
        resolved("turn-1", "missing-evidence"),
        [user_turn()],
        registry,
    )

    assert result.status == ContextStatus.INSUFFICIENT
    assert result.evidence_ids == ()
    assert result.resolved_context is None
    assert not result.analysis_error


def test_evidence_origin_turn_must_exist_in_history():
    evidence = make_evidence(turn_id="missing-turn")
    registry = make_registry([evidence])

    result = validate(
        resolved(),
        [user_turn()],
        registry,
    )

    assert result.analysis_error
    assert result.status == ContextStatus.INSUFFICIENT
    assert result.resolved_context is None


def test_required_vision_evidence_cannot_be_replaced_by_text():
    evidence = make_evidence(
        source_type=EvidenceSourceType.CONVERSATION_TEXT,
    )
    registry = make_registry([evidence])

    output = resolved(
        required_types=[EvidenceSourceType.VISION_DETECTION],
    )

    result = validate(output, [user_turn()], registry)

    assert result.status == ContextStatus.INSUFFICIENT
    assert result.resolved_context is None
    assert not result.analysis_error


def test_all_required_evidence_types_must_be_available():
    evidence = make_evidence(
        source_type=EvidenceSourceType.VISION_DETECTION,
    )
    registry = make_registry([evidence])

    output = resolved(
        required_types=[
            EvidenceSourceType.VISION_DETECTION,
            EvidenceSourceType.RETRIEVAL_CHUNK,
        ],
    )

    result = validate(output, [user_turn()], registry)

    assert result.status == ContextStatus.INSUFFICIENT
    assert result.resolved_context is None


def test_stale_required_evidence_returns_stale():
    old_evidence = make_evidence(
        created_at=NOW - timedelta(seconds=1801),
    )
    registry = make_registry([old_evidence])

    result = validate(resolved(), [user_turn()], registry)

    assert result.status == ContextStatus.STALE
    assert not result.analysis_error
    assert result.resolved_context is None
    assert result.evidence_ids == ("evidence-1",)


def test_exact_ttl_boundary_is_valid():
    boundary_evidence = make_evidence(
        created_at=NOW - timedelta(seconds=1800),
    )
    registry = make_registry([boundary_evidence])

    result = validate(resolved(), [user_turn()], registry)

    assert result.status == ContextStatus.RESOLVED
    assert not result.analysis_error


def test_old_non_evidence_reference_does_not_make_result_stale():
    old = user_turn("old", NOW - timedelta(days=2))
    fresh = user_turn("fresh", NOW - timedelta(minutes=2))

    evidence = make_evidence(
        turn_id="fresh",
        created_at=NOW - timedelta(minutes=2),
    )
    registry = make_registry([evidence])

    output = ContextAnalysisOutput(
        status="resolved",
        referenced_turn_ids=["old", "fresh"],
        evidence_ids=["evidence-1"],
        required_evidence_types=[
            EvidenceSourceType.CONVERSATION_TEXT
        ],
        resolved_context="The item is a glass bottle.",
        reason="Fresh evidence resolves the reference.",
    )

    result = validate(output, [old, fresh], registry)

    assert result.status == ContextStatus.RESOLVED
    assert result.evidence_turn_ids == ("fresh",)


@pytest.mark.parametrize(
    ("raw_status", "expected"),
    [
        ("insufficient", ContextStatus.INSUFFICIENT),
        ("ambiguous", ContextStatus.AMBIGUOUS),
        ("conflict", ContextStatus.CONFLICT),
    ],
)
def test_unresolved_semantic_statuses_are_preserved(raw_status, expected):
    output = ContextAnalysisOutput(
        status=raw_status,
        referenced_turn_ids=["turn-1"],
        reason="Cannot resolve.",
    )
    registry = make_registry()

    result = validate(output, [user_turn()], registry)

    assert result.status == expected
    assert not result.analysis_error


def test_evidence_turn_ids_are_derived_from_registry():
    evidence = make_evidence(
        evidence_id="evidence-1",
        turn_id="turn-1",
    )
    registry = make_registry([evidence])

    result = validate(
        resolved(
            turn_id="turn-1",
            evidence_id="evidence-1",
        ),
        [user_turn("turn-1")],
        registry,
    )

    assert result.status == ContextStatus.RESOLVED
    assert result.evidence_ids == ("evidence-1",)
    assert result.evidence_turn_ids == ("turn-1",)
    assert result.resolved_context == "The item is a glass bottle."


def test_independent_output_cannot_contain_evidence():
    with pytest.raises(ValidationError):
        ContextAnalysisOutput(
            status="independent",
            referenced_turn_ids=["turn-1"],
            evidence_ids=["evidence-1"],
            required_evidence_types=[
                EvidenceSourceType.CONVERSATION_TEXT
            ],
            reason="Independent.",
        )


def test_duplicate_history_ids_fail_closed():
    turn = user_turn()
    output = ContextAnalysisOutput(
        status="independent",
        referenced_turn_ids=["turn-1"],
        reason="Independent.",
    )
    registry = make_registry()

    result = validate(output, [turn, turn], registry)

    assert result.analysis_error


def test_future_timestamp_fails_closed():
    future = user_turn(timestamp=NOW + timedelta(seconds=1))
    output = ContextAnalysisOutput(
        status="independent",
        referenced_turn_ids=["turn-1"],
        reason="Independent.",
    )
    registry = make_registry()

    result = validate(output, [future], registry)

    assert result.analysis_error


def test_future_evidence_timestamp_fails_closed():
    future_evidence = make_evidence(
        created_at=NOW + timedelta(seconds=1),
    )
    registry = make_registry([future_evidence])

    result = validate(resolved(), [user_turn()], registry)

    assert result.analysis_error
    assert result.status == ContextStatus.INSUFFICIENT


def test_custom_ttl_is_respected():
    old_evidence = make_evidence(
        created_at=NOW - timedelta(seconds=301),
    )
    registry = make_registry([old_evidence])

    validator = ContextAnalysisValidator(
        ContextConfig(max_age_seconds=300)
    )

    result = validate(
        resolved(),
        [user_turn()],
        registry,
        validator,
    )

    assert result.status == ContextStatus.STALE


def test_wrong_output_type_fails_closed():
    registry = make_registry()

    result = validate(
        {"status": "resolved"},
        [user_turn()],
        registry,
    )

    assert result.analysis_error
    assert result.status == ContextStatus.INSUFFICIENT
