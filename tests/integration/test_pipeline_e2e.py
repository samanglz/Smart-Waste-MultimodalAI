"""End-to-end integration tests using deterministic mock adapters."""

from smart_waste_multimodal.adapters.mock_adapters import (
    MockGenerationAdapter,
    MockRetrievalAdapter,
    MockVisionAdapter,
)
from smart_waste_multimodal.contracts import (
    AnalyzeRequest,
    ImageArtifact,
    RequestOptions,
)
from smart_waste_multimodal.evidence import (
    ConversationScopedInMemoryEvidenceRegistryFactory,
)
from smart_waste_multimodal.orchestration.orchestrator import (
    PipelineOrchestrator,
)


def make_e2e_request() -> AnalyzeRequest:
    return AnalyzeRequest(
        request_id="e2e-mock-request-001",
        conversation_id="test-conversation-test_pipeline_e2e-22",
        turn_id="test-turn-test_pipeline_e2e-22",
        image=ImageArtifact(
            artifact_id="e2e-image-001",
            media_type="image/jpeg",
            width=100,
            height=100,
        ),
        question="How should I recycle this item?",
        locale="en",
        options=RequestOptions(
            run_vision=True,
            run_retrieval=True,
            generate_answer=True,
            retrieval_query_source="question",
        ),
    )


def test_pipeline_end_to_end_with_mock_adapters():
    orchestrator = PipelineOrchestrator(
        vision_stage=MockVisionAdapter(),
        retrieval_stage=MockRetrievalAdapter(),
        generation_stage=MockGenerationAdapter(),
        evidence_registry_factory=ConversationScopedInMemoryEvidenceRegistryFactory(),
    )

    response = orchestrator.run(
        (_test_request := make_e2e_request()),
        conversation=_test_conversation_for_request(_test_request),
    )

    # The top-level contract and all requested stages must succeed.
    assert response.request_id == "e2e-mock-request-001"
    assert response.status == "success"
    assert response.errors == []
    assert response.warnings == []

    assert response.vision is not None
    assert response.vision.status == "success"
    assert response.vision.detections[0].class_name == "plastic"

    assert response.retrieval is not None
    assert response.retrieval.status == "success"
    assert response.retrieval.results[0].chunk_id == ("mock-recycling-chunk-001")

    assert response.generation is not None
    assert response.generation.status == "success"
    assert "plastic" in response.generation.answer.lower()
    assert response.generation.citation_chunk_ids == ["mock-recycling-chunk-001"]

    # Serialization should preserve a response that validates on parsing.
    serialized = response.model_dump_json()
    restored = type(response).model_validate_json(serialized)
    assert restored == response


def test_detection_based_retrieval_receives_vision_output():
    request = make_e2e_request().model_copy(
        update={
            "options": RequestOptions(
                run_vision=True,
                run_retrieval=True,
                generate_answer=True,
                retrieval_query_source="vision_detections",
            )
        }
    )

    orchestrator = PipelineOrchestrator(
        vision_stage=MockVisionAdapter(),
        retrieval_stage=MockRetrievalAdapter(),
        generation_stage=MockGenerationAdapter(),
        evidence_registry_factory=ConversationScopedInMemoryEvidenceRegistryFactory(),
    )

    response = orchestrator.run(
        (_test_request := request),
        conversation=_test_conversation_for_request(_test_request),
    )

    assert response.status == "success"
    assert response.vision is not None
    assert response.retrieval is not None
    assert "plastic" in response.retrieval.query.lower()
    assert response.generation is not None
    assert response.generation.citation_chunk_ids == [
        response.retrieval.results[0].chunk_id
    ]


def test_pipeline_recovers_ambiguous_context_before_retrieval():
    from datetime import UTC, datetime, timedelta

    from smart_waste_multimodal.contracts.context import (
        ContextAnalysisOutput,
        ContextStatus,
    )
    from smart_waste_multimodal.contracts.evidence import (
        Evidence,
        EvidenceProvenance,
        EvidenceSourceType,
    )
    from smart_waste_multimodal.contracts.conversation import ConversationTurn
    from smart_waste_multimodal.orchestration.context_recovery import (
        ContextRecoveryOrchestrator,
    )
    from smart_waste_multimodal.orchestration.final_decision import (
        FinalDecisionEngine,
    )
    from smart_waste_multimodal.query_rewriting.context_validator import (
        ContextAnalysisValidator,
    )

    history = [
        ConversationTurn(
            turn_id="turn-001",
            role="user",
            content="I found a glass bottle.",
            timestamp=datetime.now(UTC) - timedelta(seconds=30),
        )
    ]

    request = make_e2e_request().model_copy(
        update={
            "question": "What about it?",
            "conversation_history": history,
        }
    )

    class AmbiguousContextAnalyzer:
        def analyze(self, query, conversation_history):
            assert query == "What about it?"
            assert len(conversation_history) == 1
            return ContextAnalysisOutput(
                status="ambiguous",
                reason="The antecedent is ambiguous.",
            )

    class DeterministicRecoveryStrategy:
        def __init__(self):
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
            return ContextAnalysisOutput(
                status="resolved",
                referenced_turn_ids=["turn-001"],
                evidence_ids=["e2e-conversation-text-evidence-001"],
                required_evidence_types=[
                    EvidenceSourceType.CONVERSATION_TEXT
                ],
                resolved_context="The referenced item is a glass bottle.",
                reason="The referenced conversation turn identifies the item.",
            )
    class SeededEvidenceRegistryFactory(
        ConversationScopedInMemoryEvidenceRegistryFactory
    ):
        def create(self, conversation, *, actor_id):
            registry = super().create(
                conversation,
                actor_id=actor_id,
            )
            source_turn = history[0]
            evidence = Evidence(
                evidence_id="e2e-conversation-text-evidence-001",
                request_id="e2e-mock-request-001",
                conversation_id=conversation.conversation_id,
                created_by_request_id="e2e-mock-request-001",
                origin_turn_id="turn-001",
                source_type=EvidenceSourceType.CONVERSATION_TEXT,
                provenance=EvidenceProvenance(
                    source_id="turn-001",
                    source_timestamp=source_turn.timestamp,
                    source_locator={"turn_id": "turn-001"},
                ),
                content={"text": source_turn.content},
                created_at=source_turn.timestamp,
            )
            registry.register(evidence, actor_id=actor_id)
            return registry


    class RecordingRetrievalAdapter:
        def __init__(self):
            self.received_assessments = []
            self.delegate = MockRetrievalAdapter()

        def __call__(self, request, vision, context_assessment=None):
            self.received_assessments.append(context_assessment)
            return self.delegate(request, vision, context_assessment)

    class RecordingFinalDecisionEngine:
        def __init__(self):
            self.delegate = FinalDecisionEngine()
            self.calls = []

        def decide(self, **kwargs):
            self.calls.append(
                {
                    "context_recovery": kwargs["context_recovery"],
                    "context_assessment": kwargs["context_assessment"],
                    "registry_records": kwargs["evidence_registry"].list_all(
                        actor_id=kwargs["evidence_registry"].owner_id
                    ),
                }
            )
            return self.delegate.decide(**kwargs)

    strategy = DeterministicRecoveryStrategy()
    validator = ContextAnalysisValidator()
    recovery_orchestrator = ContextRecoveryOrchestrator(
        strategy=strategy,
        validator=validator,
        max_attempts=2,
    )
    retrieval = RecordingRetrievalAdapter()
    decision_engine = RecordingFinalDecisionEngine()

    orchestrator = PipelineOrchestrator(
        vision_stage=MockVisionAdapter(),
        retrieval_stage=retrieval,
        generation_stage=MockGenerationAdapter(),
        evidence_registry_factory=SeededEvidenceRegistryFactory(),
        context_analyzer=AmbiguousContextAnalyzer(),
        context_validator=validator,
        context_recovery_orchestrator=recovery_orchestrator,
        final_decision_engine=decision_engine,
    )

    response = orchestrator.run(
        (_test_request := request),
        conversation=_test_conversation_for_request(_test_request),
    )

    assert strategy.calls == [1]
    assert len(retrieval.received_assessments) == 1

    recovered_assessment = retrieval.received_assessments[0]
    assert recovered_assessment is not None
    assert recovered_assessment.status == ContextStatus.RESOLVED
    assert recovered_assessment.evidence_turn_ids == ("turn-001",)

    assert len(decision_engine.calls) == 1
    decision_call = decision_engine.calls[0]
    recovery_result = decision_call["context_recovery"]

    assert recovery_result is not None
    assert recovery_result.resolved is True
    assert len(recovery_result.attempts) == 1
    assert recovery_result.assessment is recovered_assessment
    assert decision_call["context_assessment"] is recovered_assessment
    assert decision_call["registry_records"]

    assert response.final_decision is not None
    assert response.final_decision.execution_status == "DECIDED"
    assert response.final_decision.action == "DIRECT"
    assert response.retrieval is not None
    assert response.generation is not None


def _test_conversation_for_request(request):
    """Build a valid test Conversation for the supplied AnalyzeRequest."""
    from datetime import UTC, datetime

    from smart_waste_multimodal.contracts.conversation import Conversation

    now = datetime.now(UTC)
    return Conversation(
        conversation_id=request.conversation_id,
        owner_id="test-owner",
        created_at=now,
        updated_at=now,
    )
