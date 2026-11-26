"""Integration tests for selective query rewriting through the real pipeline."""

import json
from datetime import UTC, datetime, timedelta

from smart_waste_multimodal.adapters.mock_adapters import (
    MockGenerationAdapter,
    MockVisionAdapter,
)
from smart_waste_multimodal.contracts import (
    AnalyzeRequest,
    ImageArtifact,
    RequestOptions,
)
from smart_waste_multimodal.contracts.context import (
    ContextAnalysisOutput,
    ContextAssessment,
    ContextStatus,
)

from smart_waste_multimodal.contracts.evidence import (
    Evidence,
    EvidenceProvenance,
    EvidenceSourceType,
)
from smart_waste_multimodal.contracts.conversation import ConversationTurn
from smart_waste_multimodal.evidence import ConversationScopedInMemoryEvidenceRegistryFactory
from smart_waste_multimodal.orchestration.context_recovery import (
    ContextRecoveryOrchestrator,
)
from smart_waste_multimodal.orchestration.orchestrator import PipelineOrchestrator
from smart_waste_multimodal.query_rewriting.context_validator import (
    ContextAnalysisValidator,
)
from smart_waste_multimodal.query_rewriting.service import QueryRewriteService
from smart_waste_multimodal.retrieval.hybrid_rag_adapter import HybridRAGAdapter


class RecordingRetriever:
    """Deterministic retriever that records the exact query it receives."""

    def __init__(self):
        self.queries = []

    def search(self, query: str, top_k: int = 5):
        self.queries.append((query, top_k))
        return [
            {
                "chunk_id": "integration-chunk-001",
                "source": "integration-test-source",
                "text": "Follow local recycling guidance for the identified material.",
                "score": 0.91,
                "rank": 1,
            }
        ]


class FixedContextAnalyzer:
    """Deterministic context analyzer for resolved-context integration tests."""

    def __init__(self, output):
        self.output = output
        self.calls = []

    def analyze(self, query, conversation_history):
        self.calls.append((query, tuple(turn.turn_id for turn in conversation_history)))
        return self.output


class RecordingRewriter:
    """Deterministic rewrite provider that records the prompt."""

    def __init__(self, rewritten_query):
        self.rewritten_query = rewritten_query
        self.messages = []

    def __call__(self, messages):
        self.messages.append(messages)
        return json.dumps(
            {
                "rewritten_query": self.rewritten_query,
                "preserved_aspects": [
                    "original disposal question",
                    "resolved material reference",
                    "safety concern",
                ],
            }
        )


class FailingRewriter:
    def __init__(self):
        self.calls = 0

    def __call__(self, messages):
        self.calls += 1
        raise RuntimeError("Simulated provider failure")


def make_request(
    *,
    question,
    history=None,
    run_vision=False,
    retrieval_query_source="question",
):
    return AnalyzeRequest(
        request_id="selective-rewrite-integration",
        conversation_id="test-conversation-test_selective_query_rewrite_pipeline-102",
        turn_id="test-turn-test_selective_query_rewrite_pipeline-102",
        image=ImageArtifact(
            artifact_id="selective-rewrite-image",
            media_type="image/jpeg",
            width=100,
            height=100,
        ),
        conversation_history=history or [],
        question=question,
        locale="en",
        options=RequestOptions(
            run_vision=run_vision,
            run_retrieval=True,
            generate_answer=False,
            retrieval_query_source=retrieval_query_source,
        ),
    )


def make_pipeline(
    retriever,
    rewrite_service,
    *,
    context_analyzer=None,
    context_recovery_orchestrator=None,
    evidence_registry_factory=None,
):
    retrieval_adapter = HybridRAGAdapter(
        retriever=retriever,
        top_k=3,
        index_version="integration-index-v1",
        rewrite_service=rewrite_service,
    )

    return PipelineOrchestrator(
        vision_stage=MockVisionAdapter(),
        retrieval_stage=retrieval_adapter,
        generation_stage=MockGenerationAdapter(),
        evidence_registry_factory=(
            evidence_registry_factory
            or ConversationScopedInMemoryEvidenceRegistryFactory()
        ),
        context_analyzer=context_analyzer,
        context_validator=ContextAnalysisValidator(),
        context_recovery_orchestrator=context_recovery_orchestrator,
    )


def make_resolved_history():
    return [
        ConversationTurn(
            turn_id="turn-glass-001",
            role="user",
            content="I found a broken glass bottle in the garage.",
            timestamp=datetime.now(UTC) - timedelta(seconds=30),
        )
    ]


def register_historical_evidence(factory, request, history):
    conversation = _test_conversation_for_request(request)
    actor_id = conversation.owner_id

    registry = factory.create(conversation, actor_id=actor_id)

    matching_turns = [
        turn for turn in history
        if turn.turn_id == "turn-glass-001"
    ]
    if len(matching_turns) != 1:
        raise AssertionError(
            "Expected exactly one historical glass turn."
        )

    turn = matching_turns[0]

    evidence = Evidence(
        evidence_id="evidence-glass-001",
        request_id="historical-request-glass-001",
        conversation_id=request.conversation_id,
        created_by_request_id="historical-request-glass-001",
        origin_turn_id=turn.turn_id,
        source_type=EvidenceSourceType.CONVERSATION_TEXT,
        provenance=EvidenceProvenance(
            source_id=turn.turn_id,
            source_timestamp=turn.timestamp,
        ),
        content={"text": turn.content},
        created_at=datetime.now(UTC),
    )

    registry.register(evidence, actor_id=actor_id)
    return conversation


def make_resolved_output():
    return ContextAnalysisOutput(
        status="resolved",
        referenced_turn_ids=["turn-glass-001"],
        evidence_ids=["evidence-glass-001"],
        required_evidence_types=[
            EvidenceSourceType.CONVERSATION_TEXT,
        ],
        resolved_context="The referenced item is a broken glass bottle.",
        signals=["pronoun_reference", "material_resolved"],
        reason="The previous user turn explicitly identifies the referenced item.",
    )


def test_independent_query_reaches_retriever_without_model_rewrite():
    retriever = RecordingRetriever()
    rewriter = RecordingRewriter("this must not be used")

    rewrite_service = QueryRewriteService(
        rewriter=rewriter,
        mode="selective",
    )

    pipeline = make_pipeline(retriever, rewrite_service)

    original_query = "How should I recycle cardboard packaging?"
    response = pipeline.run((_test_request := make_request(question=original_query)), conversation=_test_conversation_for_request(_test_request))

    assert response.status == "success"
    assert response.context_assessment is not None
    assert response.context_assessment.status == ContextStatus.INDEPENDENT

    assert retriever.queries == [(original_query, 3)]
    assert rewriter.messages == []

    assert response.retrieval is not None
    assert response.retrieval.query == original_query


def test_resolved_context_is_rewritten_before_real_adapter_retrieval():
    retriever = RecordingRetriever()
    rewritten_query = (
        "How to safely dispose of a broken glass bottle, "
        "including appropriate handling precautions"
    )
    rewriter = RecordingRewriter(rewritten_query)

    history = make_resolved_history()
    analyzer = FixedContextAnalyzer(make_resolved_output())

    rewrite_service = QueryRewriteService(
        rewriter=rewriter,
        mode="selective",
    )

    request = make_request(
        question="How should I dispose of it safely?",
        history=history,
    )
    registry_factory = ConversationScopedInMemoryEvidenceRegistryFactory()
    conversation = register_historical_evidence(
        registry_factory,
        request,
        history,
    )

    pipeline = make_pipeline(
        retriever,
        rewrite_service,
        context_analyzer=analyzer,
        evidence_registry_factory=registry_factory,
    )

    original_query = request.question
    response = pipeline.run(request, conversation=conversation)

    assert response.status == "success"
    assert response.context_assessment is not None
    assert response.context_assessment.status == ContextStatus.RESOLVED

    assert len(analyzer.calls) == 1
    assert len(rewriter.messages) == 1
    assert retriever.queries == [(rewritten_query, 3)]

    assert response.retrieval is not None
    assert response.retrieval.query == rewritten_query

    payload = json.loads(rewriter.messages[0][1]["content"])
    assert payload["original_query"] == original_query
    assert payload["resolved_context"] == (
        "The referenced item is a broken glass bottle."
    )
    assert payload["evidence_turn_ids"] == ["turn-glass-001"]

    assert history[0].content not in rewriter.messages[0][1]["content"]


def test_detection_based_retrieval_bypasses_context_analysis_and_rewriting():
    retriever = RecordingRetriever()
    rewriter = RecordingRewriter("this must not be used")

    rewrite_service = QueryRewriteService(
        rewriter=rewriter,
        mode="selective",
    )

    pipeline = make_pipeline(retriever, rewrite_service)

    request = make_request(
        question=None,
        run_vision=True,
        retrieval_query_source="vision_detections",
    )
    response = pipeline.run((_test_request := request), conversation=_test_conversation_for_request(_test_request))

    assert response.status == "success"
    assert response.vision is not None
    assert response.context_assessment is None

    assert len(retriever.queries) == 1
    assert "plastic" in retriever.queries[0][0].lower()
    assert rewriter.messages == []
    assert response.query_rewrite_trace is not None
    assert response.query_rewrite_trace.action == "bypassed"
    assert response.query_rewrite_trace.reason == "detection_based_retrieval"
    assert response.query_rewrite_trace.retrieval_query == response.retrieval.query


def test_unresolved_context_is_rejected_by_rewrite_service_before_retrieval():
    retriever = RecordingRetriever()
    rewriter = RecordingRewriter("unsafe speculative rewrite")

    rewrite_service = QueryRewriteService(
        rewriter=rewriter,
        mode="selective",
    )
    adapter = HybridRAGAdapter(
        retriever=retriever,
        rewrite_service=rewrite_service,
    )

    request = make_request(
        question="What about that one?",
    )
    unresolved_assessment = ContextAssessment(
        status=ContextStatus.AMBIGUOUS,
        reason="Multiple possible antecedents remain.",
    )

    import pytest

    with pytest.raises(ValueError, match="Context must be resolved"):
        adapter(request, None, unresolved_assessment)

    assert retriever.queries == []
    assert rewriter.messages == []


def test_rewriter_provider_failure_falls_back_to_original_query():
    retriever = RecordingRetriever()
    rewriter = FailingRewriter()

    history = make_resolved_history()
    analyzer = FixedContextAnalyzer(make_resolved_output())

    rewrite_service = QueryRewriteService(
        rewriter=rewriter,
        mode="selective",
    )

    request = make_request(
        question="How should I dispose of it safely?",
        history=history,
    )
    registry_factory = ConversationScopedInMemoryEvidenceRegistryFactory()
    conversation = register_historical_evidence(
        registry_factory,
        request,
        history,
    )

    pipeline = make_pipeline(
        retriever,
        rewrite_service,
        context_analyzer=analyzer,
        evidence_registry_factory=registry_factory,
    )

    original_query = request.question
    response = pipeline.run(request, conversation=conversation)

    assert response.status == "success"
    assert rewriter.calls == 1
    assert retriever.queries == [(original_query, 3)]

    assert response.retrieval is not None
    assert response.retrieval.query == original_query
    assert response.query_rewrite_trace is not None
    assert response.query_rewrite_trace.action == "fallback_original"
    assert response.query_rewrite_trace.failure_category == "provider_failure"
    assert response.query_rewrite_trace.failure_type == "RuntimeError"
    assert response.query_rewrite_trace.retrieval_query == original_query

class FixedRecoveryStrategy:
    """Deterministic strategy for the real pipeline integration test."""

    def __init__(self, output):
        self.output = output
        self.calls = []

    def propose(
        self,
        question,
        conversation_history,
        current_assessment,
        attempt,
        previous_proposal_fingerprints,
    ):
        self.calls.append(
            {
                "question": question,
                "turn_ids": tuple(
                    turn.turn_id for turn in conversation_history
                ),
                "current_status": (
                    current_assessment.status
                    if current_assessment is not None
                    else None
                ),
                "attempt": attempt,
            }
        )
        return self.output


def test_pipeline_recovers_ambiguous_context_before_retrieval():
    retriever = RecordingRetriever()
    rewritten_query = (
        "How to safely dispose of a broken glass bottle, "
        "including appropriate handling precautions"
    )
    rewriter = RecordingRewriter(rewritten_query)
    history = make_resolved_history()

    analyzer = FixedContextAnalyzer(
        ContextAnalysisOutput(
            status="ambiguous",
            reason="The initial analysis could not identify the antecedent.",
        )
    )
    strategy = FixedRecoveryStrategy(make_resolved_output())
    recovery = ContextRecoveryOrchestrator(
        strategy=strategy,
        validator=ContextAnalysisValidator(),
        max_attempts=2,
    )
    rewrite_service = QueryRewriteService(
        rewriter=rewriter,
        mode="selective",
    )

    request = make_request(
        question="How should I dispose of it safely?",
        history=history,
    )
    registry_factory = ConversationScopedInMemoryEvidenceRegistryFactory()
    conversation = register_historical_evidence(
        registry_factory,
        request,
        history,
    )

    pipeline = make_pipeline(
        retriever,
        rewrite_service,
        context_analyzer=analyzer,
        context_recovery_orchestrator=recovery,
        evidence_registry_factory=registry_factory,
    )

    original_query = request.question
    response = pipeline.run(request, conversation=conversation)

    assert response.status == "success"
    assert response.context_assessment is not None
    assert response.context_assessment.status == ContextStatus.RESOLVED
    assert len(analyzer.calls) == 1
    assert len(strategy.calls) == 1
    assert strategy.calls[0]["attempt"] == 1
    assert strategy.calls[0]["current_status"] == ContextStatus.AMBIGUOUS
    assert strategy.calls[0]["turn_ids"] == ("turn-glass-001",)
    assert len(rewriter.messages) == 1
    assert retriever.queries == [(rewritten_query, 3)]
    assert response.retrieval is not None
    assert response.retrieval.query == rewritten_query




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
