"""Unit tests for bounded, evidence-validated context recovery."""

from datetime import UTC, datetime, timedelta

from smart_waste_multimodal.contracts.context import (
    ContextAnalysisOutput,
    ContextAssessment,
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
from smart_waste_multimodal.orchestration.context_recovery import (
    ContextRecoveryOrchestrator,
)
from smart_waste_multimodal.query_rewriting.context_validator import (
    ContextAnalysisValidator,
)


NOW = datetime.now(UTC)
ACTOR_ID = "owner-001"
CONVERSATION_ID = "conversation-001"


def make_history():
    return [
        ConversationTurn(
            turn_id="turn-001",
            role="user",
            content="I found a glass bottle.",
            timestamp=NOW - timedelta(seconds=30),
        )
    ]


def make_evidence(
    evidence_id="evidence-001",
    turn_id="turn-001",
    source_type=EvidenceSourceType.CONVERSATION_TEXT,
):
    created_at = NOW - timedelta(seconds=20)

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
        created_at=NOW - timedelta(minutes=5),
        updated_at=NOW - timedelta(minutes=5),
    )

    factory = ConversationScopedInMemoryEvidenceRegistryFactory()
    registry = factory.create(conversation, actor_id=ACTOR_ID)

    for evidence in evidence_items:
        registry.register(evidence, actor_id=ACTOR_ID)

    return registry


def ambiguous_assessment():
    return ContextAssessment(
        status=ContextStatus.AMBIGUOUS,
        reason="The antecedent is ambiguous.",
    )


def resolved_output(
    turn_id="turn-001",
    evidence_id="evidence-001",
):
    return ContextAnalysisOutput(
        status="resolved",
        referenced_turn_ids=[turn_id],
        evidence_ids=[evidence_id],
        required_evidence_types=[
            EvidenceSourceType.CONVERSATION_TEXT
        ],
        resolved_context="The referenced item is a glass bottle.",
        reason="The user turn explicitly identifies the item.",
    )


class SequenceStrategy:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def propose(
        self,
        question,
        conversation_history,
        current_assessment,
        attempt,
        previous_proposal_fingerprints,
    ):
        self.calls.append(attempt)
        if not self.outputs:
            return None
        return self.outputs.pop(0)


def make_orchestrator(strategy, max_attempts=2):
    return ContextRecoveryOrchestrator(
        strategy=strategy,
        validator=ContextAnalysisValidator(),
        max_attempts=max_attempts,
    )


def resolve(
    strategy,
    *,
    history=None,
    registry=None,
    initial_assessment=None,
    max_attempts=2,
    current_turn_id=None,
    current_turn_timestamp=None,
):
    return make_orchestrator(
        strategy,
        max_attempts=max_attempts,
    ).resolve(
        question="What about it?",
        conversation_history=(
            make_history() if history is None else history
        ),
        initial_assessment=(
            initial_assessment or ambiguous_assessment()
        ),
        registry=registry or make_registry(),
        actor_id=ACTOR_ID,
        current_turn_id=current_turn_id,
        current_turn_timestamp=current_turn_timestamp,
    )


def test_recovery_accepts_validated_proposal():
    evidence = make_evidence()
    registry = make_registry([evidence])
    strategy = SequenceStrategy([resolved_output()])

    result = resolve(strategy, registry=registry)

    assert result.resolved is True
    assert result.assessment.status == ContextStatus.RESOLVED
    assert result.assessment.evidence_ids == ("evidence-001",)
    assert len(result.attempts) == 1
    assert result.attempts[0].outcome == "resolved"


def test_recovery_is_bounded():
    first = ContextAnalysisOutput(
        status="ambiguous",
        reason="Still ambiguous: first proposal.",
    )
    second = ContextAnalysisOutput(
        status="ambiguous",
        reason="Still ambiguous: second proposal.",
    )
    strategy = SequenceStrategy([first, second])

    result = resolve(strategy, max_attempts=2)

    assert result.resolved is False
    assert len(result.attempts) == 2
    assert strategy.calls == [1, 2]


def test_repeated_proposal_stops():
    unresolved = ContextAnalysisOutput(
        status="insufficient",
        reason="Not enough information.",
    )
    strategy = SequenceStrategy([unresolved, unresolved])

    result = resolve(strategy, max_attempts=2)

    assert result.resolved is False
    assert len(result.attempts) == 2
    assert result.attempts[-1].outcome == "repeated_proposal"
    assert strategy.calls == [1, 2]


def test_recovery_rejects_unknown_evidence():
    registry = make_registry()
    strategy = SequenceStrategy([
        resolved_output(evidence_id="missing-evidence")
    ])

    result = resolve(strategy, registry=registry)

    assert result.resolved is False
    assert result.assessment.status == ContextStatus.INSUFFICIENT
    assert result.attempts[0].outcome == "still_unresolved"


def test_recovery_accepts_current_turn_evidence_without_history():
    current_turn_id = "turn-current"
    current_turn_timestamp = NOW - timedelta(seconds=10)

    evidence = make_evidence(
        evidence_id="evidence-current",
        turn_id=current_turn_id,
    )
    registry = make_registry([evidence])
    strategy = SequenceStrategy([
        resolved_output(
            turn_id=current_turn_id,
            evidence_id="evidence-current",
        )
    ])

    result = resolve(
        strategy,
        history=[],
        registry=registry,
        current_turn_id=current_turn_id,
        current_turn_timestamp=current_turn_timestamp,
    )

    assert result.resolved is True
    assert result.assessment.status == ContextStatus.RESOLVED
    assert result.assessment.evidence_ids == ("evidence-current",)
    assert result.assessment.evidence_turn_ids == (current_turn_id,)
    assert result.attempts[0].outcome == "resolved"


def test_missing_strategy_fails_closed():
    result = resolve(None)

    assert result.resolved is False
    assert result.attempts == ()
    assert result.assessment.status == ContextStatus.AMBIGUOUS


def test_valid_context_bypasses_recovery():
    strategy = SequenceStrategy([])
    initial = ContextAssessment(
        status=ContextStatus.INDEPENDENT,
        reason="The query is self-contained.",
    )

    result = resolve(
        strategy,
        initial_assessment=initial,
    )

    assert result.resolved is True
    assert result.assessment.status == ContextStatus.INDEPENDENT
    assert result.attempts == ()
    assert strategy.calls == []
