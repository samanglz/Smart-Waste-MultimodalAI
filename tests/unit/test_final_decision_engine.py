"""Unit tests for the final decision engine."""

from datetime import UTC, datetime

from smart_waste_multimodal.contracts.common import ErrorInfo
from smart_waste_multimodal.contracts.context import (
    ContextAssessment,
    ContextStatus,
)
from smart_waste_multimodal.contracts.conversation import Conversation
from smart_waste_multimodal.contracts.evidence import (
    Evidence,
    EvidenceProvenance,
    EvidenceSourceType,
)
from smart_waste_multimodal.contracts.request import (
    AnalyzeRequest,
    ImageArtifact,
    RequestOptions,
)
from smart_waste_multimodal.contracts.retrieval import (
    EvidenceItem,
    RetrievalResult,
)
from smart_waste_multimodal.contracts.vision import (
    Detection,
    VisionResult,
)
from smart_waste_multimodal.evidence.in_memory_registry import (
    ConversationScopedInMemoryEvidenceRegistryFactory,
)
from smart_waste_multimodal.orchestration.context_recovery import (
    ContextRecoveryResult,
)
from smart_waste_multimodal.orchestration.final_decision import (
    DecisionAssessment,
    FinalDecisionEngine,
)

NOW = datetime.now(UTC)


TEST_OWNER_ID = "test-owner"


class TestRegistryAdapter:
    __test__ = False
    """Bind test operations to the conversation registry owner."""

    def __init__(self, registry):
        self._registry = registry

    def __getattr__(self, name):
        return getattr(self._registry, name)

    def register(self, evidence):
        return self._registry.register(
            evidence,
            actor_id=TEST_OWNER_ID,
        )

    def list_all(self, *, actor_id):
        return self._registry.list_all(actor_id=actor_id)


def make_registry(request, *, conversation_id=None):
    conversation = Conversation(
        conversation_id=conversation_id or request.conversation_id,
        owner_id=TEST_OWNER_ID,
        created_at=NOW,
        updated_at=NOW,
    )
    factory = ConversationScopedInMemoryEvidenceRegistryFactory()
    registry = factory.create(
        conversation,
        actor_id=TEST_OWNER_ID,
    )
    return TestRegistryAdapter(registry)


def make_request(
    *,
    question="What material is visible?",
    query_source="question",
    run_retrieval=True,
    generate_answer=True,
    include_image=True,
):
    return AnalyzeRequest(
        request_id="request-1",
        conversation_id="test-conversation-test_final_decision_engine-49",
        turn_id="test-turn-test_final_decision_engine-49",
        image=(
            ImageArtifact(
                artifact_id="image-1",
                media_type="image/jpeg",
            )
            if include_image
            else None
        ),
        question=question,
        options=RequestOptions(
            run_vision=True,
            run_retrieval=run_retrieval,
            generate_answer=generate_answer,
            retrieval_query_source=query_source,
        ),
    )


def make_vision():
    return VisionResult(
        status="success",
        detections=[
            Detection(
                class_name="plastic",
                confidence=0.91,
                bbox_xyxy=(1, 2, 30, 40),
            )
        ],
        image_width=100,
        image_height=100,
        model_version="vision-v1",
        artifact_id="image-1",
    )


def register_vision(registry, request, vision):
    registry.register(
        Evidence(
            evidence_id="vision:image-1:snapshot",
            request_id=request.request_id,
            source_type=EvidenceSourceType.VISION_SNAPSHOT,
            provenance=EvidenceProvenance(
                source_id="image-1",
                source_version=vision.model_version,
                source_locator={"artifact_id": "image-1"},
            ),
            content=vision.model_dump(mode="json"),
            created_at=NOW,
            conversation_id=request.conversation_id,
            created_by_request_id=request.request_id,
            origin_turn_id=request.turn_id,
        )
    )
    for index, detection in enumerate(vision.detections):
        registry.register(
            Evidence(
                evidence_id=f"vision:image-1:detection:{index:04d}",
                request_id=request.request_id,
                source_type=EvidenceSourceType.VISION_DETECTION,
                provenance=EvidenceProvenance(
                    source_id="image-1",
                    source_version=vision.model_version,
                    source_locator={
                        "artifact_id": "image-1",
                        "detection_index": str(index),
                    },
                ),
                content=detection.model_dump(mode="json"),
                created_at=NOW,
                metadata={"detection_index": index},
                conversation_id=request.conversation_id,
                created_by_request_id=request.request_id,
                origin_turn_id=request.turn_id,
            )
        )


def make_retrieval():
    item = EvidenceItem(
        chunk_id="chunk-1",
        source="recycling-guide.md",
        text="Plastic bottles belong in the plastics recycling stream.",
        score=0.0,
        rank=1,
    )
    result = RetrievalResult(
        status="success",
        query="What material is visible?",
        results=[item],
        index_version="index-v1",
    )
    return result, item


def register_retrieval(registry, request, retrieval, item):
    registry.register(
        Evidence(
            evidence_id="retrieval:index-v1:1:0000:chunk-1",
            request_id=request.request_id,
            source_type=EvidenceSourceType.RETRIEVAL_CHUNK,
            provenance=EvidenceProvenance(
                source_id=item.chunk_id,
                source_version=retrieval.index_version,
                source_locator={"source": item.source},
            ),
            content=item.model_dump(mode="json"),
            created_at=NOW,
            metadata={"index_version": "index-v1", "rank": item.rank},
            conversation_id=request.conversation_id,
            created_by_request_id=request.request_id,
            origin_turn_id=request.turn_id,
        )
    )


def test_successful_retrieval_produces_direct_decision():
    request = make_request()
    registry = make_registry(request)
    retrieval, item = make_retrieval()
    register_retrieval(registry, request, retrieval, item)

    result = FinalDecisionEngine().decide(
        request=request,
        evidence_registry=registry,
        retrieval=retrieval,
    )

    assert result.execution_status == "DECIDED"
    assert result.action == "DIRECT"
    assert result.reason_code == "SUFFICIENT_EVIDENCE"
    assert result.evidence_ids == ["retrieval:index-v1:1:0000:chunk-1"]


def test_retrieval_score_is_not_used_as_answer_confidence():
    request = make_request()
    registry = make_registry(request)
    retrieval, item = make_retrieval()
    register_retrieval(registry, request, retrieval, item)

    result = FinalDecisionEngine().decide(
        request=request,
        evidence_registry=registry,
        retrieval=retrieval,
    )

    assert item.score == 0.0
    assert result.action == "DIRECT"


def test_successful_retrieval_without_registry_mapping_is_unavailable():
    request = make_request()
    registry = make_registry(request)
    retrieval, _ = make_retrieval()

    result = FinalDecisionEngine().decide(
        request=request,
        evidence_registry=registry,
        retrieval=retrieval,
    )

    assert result.execution_status == "UNAVAILABLE"
    assert result.action is None
    assert result.evidence_ids == []


def test_vision_only_detection_task_can_be_direct():
    request = make_request(
        question=None,
        query_source="vision_detections",
        run_retrieval=False,
    )
    registry = make_registry(request)
    vision = make_vision()
    register_vision(registry, request, vision)

    result = FinalDecisionEngine().decide(
        request=request,
        evidence_registry=registry,
        vision=vision,
    )

    assert result.execution_status == "DECIDED"
    assert result.action == "DIRECT"
    assert len(result.evidence_ids) == 2


def test_vision_only_question_task_is_conservatively_insufficient():
    request = make_request(
        question="Explain the environmental impact of this object.",
        run_retrieval=False,
    )
    registry = make_registry(request)
    vision = make_vision()
    register_vision(registry, request, vision)

    result = FinalDecisionEngine().decide(
        request=request,
        evidence_registry=registry,
        vision=vision,
    )

    assert result.execution_status == "DECIDED"
    assert result.action == "INSUFFICIENT_EVIDENCE"


def test_ambiguous_context_does_not_short_circuit_vision_assessment():
    request = make_request(
        question="Explain the environmental impact of this object.",
        run_retrieval=False,
    )
    registry = make_registry(request)
    vision = make_vision()
    register_vision(registry, request, vision)

    ambiguous_context = ContextAssessment(
        status=ContextStatus.AMBIGUOUS,
        reason="The conversational context is ambiguous.",
    )

    result = FinalDecisionEngine().decide(
        request=request,
        evidence_registry=registry,
        vision=vision,
        context_assessment=ambiguous_context,
    )

    assert result.execution_status == "DECIDED"
    assert result.action == "INSUFFICIENT_EVIDENCE"
    assert result.evidence_ids == []
    assert result.reason == (
        "The validated evidence set does not establish "
        "sufficient support for a direct answer."
    )


def test_empty_registry_produces_insufficient_evidence():
    request = make_request()
    registry = make_registry(request)

    result = FinalDecisionEngine().decide(
        request=request,
        evidence_registry=registry,
    )

    assert result.execution_status == "DECIDED"
    assert result.action == "INSUFFICIENT_EVIDENCE"


def test_unresolved_context_does_not_automatically_request_human_review():
    request = make_request()
    registry = make_registry(request)
    context = ContextAssessment(
        status=ContextStatus.AMBIGUOUS,
        referenced_turn_ids=("turn-1",),
        reason="The reference is ambiguous.",
    )

    result = FinalDecisionEngine().decide(
        request=request,
        evidence_registry=registry,
        context_assessment=context,
    )

    assert result.action == "INSUFFICIENT_EVIDENCE"
    assert result.clarification_required is False


def test_injected_policy_can_explicitly_request_human_review():
    class HumanReviewPolicy:
        def evaluate(self, decision_input):
            return DecisionAssessment(
                action="HUMAN_REVIEW",
                reason_code="HUMAN_ACTION_REQUIRED",
                reason="A human must confirm the requested action.",
                clarification_required=True,
                clarification_reason="Please confirm the intended material.",
            )

    request = make_request()
    registry = make_registry(request)

    result = FinalDecisionEngine(policy=HumanReviewPolicy()).decide(
        request=request,
        evidence_registry=registry,
    )

    assert result.execution_status == "DECIDED"
    assert result.action == "HUMAN_REVIEW"
    assert result.clarification_required is True


def test_generate_answer_false_does_not_suppress_decision():
    request = make_request(generate_answer=False)
    registry = make_registry(request)
    retrieval, item = make_retrieval()
    register_retrieval(registry, request, retrieval, item)

    result = FinalDecisionEngine().decide(
        request=request,
        evidence_registry=registry,
        retrieval=retrieval,
    )

    assert result.action == "DIRECT"


def test_registry_request_mismatch_is_unavailable():
    request = make_request()
    registry = make_registry(
        request,
        conversation_id="another-conversation",
    )

    result = FinalDecisionEngine().decide(
        request=request,
        evidence_registry=registry,
    )

    assert result.execution_status == "UNAVAILABLE"
    assert result.action is None


def test_registry_failure_is_unavailable():
    class BrokenRegistry:
        conversation_id = "test-conversation-test_final_decision_engine-49"
        owner_id = "test-owner"

        def list_all(self, *, actor_id):
            raise RuntimeError("storage unavailable")

    request = make_request()
    result = FinalDecisionEngine().decide(
        request=request,
        evidence_registry=BrokenRegistry(),
    )

    assert result.execution_status == "UNAVAILABLE"
    assert result.action is None
    assert result.reason_code == "DECISION_DEPENDENCY_UNAVAILABLE"


def test_vision_registry_mapping_is_validated():
    request = make_request(
        question=None,
        query_source="vision_detections",
        run_retrieval=False,
    )
    registry = make_registry(request)
    vision = make_vision()

    # Register a snapshot with altered content.
    altered = vision.model_dump(mode="json")
    altered["image_width"] = 99
    registry.register(
        Evidence(
            evidence_id="vision:image-1:snapshot",
            request_id=request.request_id,
            source_type=EvidenceSourceType.VISION_SNAPSHOT,
            provenance=EvidenceProvenance(
                source_id="image-1",
                source_version=vision.model_version,
                source_locator={"artifact_id": "image-1"},
            ),
            content=altered,
            created_at=NOW,
            conversation_id=request.conversation_id,
            created_by_request_id=request.request_id,
            origin_turn_id=request.turn_id,
        )
    )

    result = FinalDecisionEngine().decide(
        request=request,
        evidence_registry=registry,
        vision=vision,
    )

    assert result.execution_status == "UNAVAILABLE"


def test_resolved_context_evidence_is_included_with_retrieval():
    request = make_request()
    registry = make_registry(request)
    retrieval, item = make_retrieval()
    register_retrieval(registry, request, retrieval, item)

    registry.register(
        Evidence(
            evidence_id="conversation:turn-1:text",
            request_id=request.request_id,
            source_type=EvidenceSourceType.CONVERSATION_TEXT,
            provenance=EvidenceProvenance(
                source_id="turn-1",
                source_locator={"role": "user"},
            ),
            content={"text": "I mean the plastic bottle."},
            created_at=NOW,
            metadata={"role": "user"},
            conversation_id=request.conversation_id,
            created_by_request_id=request.request_id,
            origin_turn_id="turn-1",
        )
    )

    context = ContextAssessment(
        status=ContextStatus.RESOLVED,
        evidence_turn_ids=("turn-1",),
        resolved_context="The user refers to a plastic bottle.",
        reason="Reference resolved.",
    )

    result = FinalDecisionEngine().decide(
        request=request,
        evidence_registry=registry,
        retrieval=retrieval,
        context_assessment=context,
    )

    assert result.action == "DIRECT"
    assert "conversation:turn-1:text" in result.evidence_ids
    assert "retrieval:index-v1:1:0000:chunk-1" in result.evidence_ids


def test_resolved_context_with_missing_registry_evidence_is_unavailable():
    request = make_request()
    registry = make_registry(request)
    context = ContextAssessment(
        status=ContextStatus.RESOLVED,
        evidence_turn_ids=("missing-turn",),
        resolved_context="A resolved reference.",
        reason="Reference resolved.",
    )

    result = FinalDecisionEngine().decide(
        request=request,
        evidence_registry=registry,
        context_assessment=context,
    )

    assert result.execution_status == "UNAVAILABLE"


def test_detection_based_decision_can_use_vision_after_retrieval_failure():
    request = make_request(
        question=None,
        query_source="vision_detections",
        run_retrieval=True,
    )
    registry = make_registry(request)
    vision = make_vision()
    register_vision(registry, request, vision)

    result = FinalDecisionEngine().decide(
        request=request,
        evidence_registry=registry,
        vision=vision,
        retrieval_execution_status="failed",
    )

    assert result.execution_status == "DECIDED"
    assert result.action == "DIRECT"


def test_policy_cannot_reference_unvalidated_evidence():
    class InvalidPolicy:
        def evaluate(self, decision_input):
            return DecisionAssessment(
                action="DIRECT",
                reason_code="SUFFICIENT_EVIDENCE",
                reason="Invalid evidence reference.",
                evidence_ids=("not-in-registry",),
            )

    request = make_request()
    registry = make_registry(request)

    result = FinalDecisionEngine(policy=InvalidPolicy()).decide(
        request=request,
        evidence_registry=registry,
    )

    assert result.execution_status == "UNAVAILABLE"


def test_completed_retrieval_without_result_is_unavailable():
    request = make_request()
    registry = make_registry(request)

    result = FinalDecisionEngine().decide(
        request=request,
        evidence_registry=registry,
        retrieval_execution_status="completed",
    )

    assert result.execution_status == "UNAVAILABLE"


def test_final_decision_context_recovery_consistency():
    request = make_request()
    registry = make_registry(request)

    valid_context = ContextAssessment(
        status=ContextStatus.INDEPENDENT,
        reason="The question is independent.",
    )
    unresolved_context = ContextAssessment(
        status=ContextStatus.AMBIGUOUS,
        referenced_turn_ids=("turn-1",),
        reason="The reference is ambiguous.",
    )

    valid_recovery = ContextRecoveryResult(
        assessment=valid_context,
        resolved=True,
        attempts=(),
        reason="Already valid.",
    )
    result = FinalDecisionEngine().decide(
        request=request,
        evidence_registry=registry,
        context_recovery=valid_recovery,
    )
    assert result.execution_status == "DECIDED"
    assert result.action == "INSUFFICIENT_EVIDENCE"

    missing_assessment = ContextRecoveryResult(
        assessment=None,
        resolved=True,
        attempts=(),
        reason="Invalid resolved result.",
    )
    result = FinalDecisionEngine().decide(
        request=request,
        evidence_registry=registry,
        context_recovery=missing_assessment,
    )
    assert result.execution_status == "UNAVAILABLE"

    contradictory_recovery = ContextRecoveryResult(
        assessment=valid_context,
        resolved=False,
        attempts=(),
        reason="Contradictory result.",
    )
    result = FinalDecisionEngine().decide(
        request=request,
        evidence_registry=registry,
        context_recovery=contradictory_recovery,
    )
    assert result.execution_status == "UNAVAILABLE"

    consistent_unresolved = ContextRecoveryResult(
        assessment=unresolved_context,
        resolved=False,
        attempts=(),
        reason="Recovery did not resolve the reference.",
    )
    result = FinalDecisionEngine().decide(
        request=request,
        evidence_registry=registry,
        context_recovery=consistent_unresolved,
    )
    assert result.execution_status == "DECIDED"
    assert result.action == "INSUFFICIENT_EVIDENCE"


def test_resolved_recovery_rejects_assessment_with_analysis_error():
    request = make_request()
    registry = make_registry(request)

    invalid_context = ContextAssessment(
        status=ContextStatus.INDEPENDENT,
        reason="Analysis failed.",
        analysis_error="Context analysis failed.",
    )
    recovery = ContextRecoveryResult(
        assessment=invalid_context,
        resolved=True,
        attempts=(),
        reason="Invalid recovery result.",
    )

    result = FinalDecisionEngine().decide(
        request=request,
        evidence_registry=registry,
        context_recovery=recovery,
    )

    assert result.execution_status == "UNAVAILABLE"


def test_retrieval_execution_status_consistency():
    engine = FinalDecisionEngine()

    def decide(
        *,
        request=None,
        retrieval=None,
        context=None,
        status=None,
    ):
        request = request or make_request()
        registry = make_registry(request)
        return engine.decide(
            request=request,
            evidence_registry=registry,
            retrieval=retrieval,
            context_assessment=context,
            retrieval_execution_status=status,
        )

    retrieval, _ = make_retrieval()

    # A result cannot be paired with failed execution.
    result = decide(retrieval=retrieval, status="failed")
    assert result.execution_status == "UNAVAILABLE"

    # A result cannot be paired with context-skipped execution.
    ambiguous = ContextAssessment(
        status=ContextStatus.AMBIGUOUS,
        reason="The reference is ambiguous.",
    )
    result = decide(
        retrieval=retrieval,
        context=ambiguous,
        status="skipped_context",
    )
    assert result.execution_status == "UNAVAILABLE"

    # Disabled request cannot claim failed execution.
    disabled_request = make_request(run_retrieval=False)
    result = decide(request=disabled_request, status="failed")
    assert result.execution_status == "UNAVAILABLE"

    # Context skipping is not valid for detection-based retrieval.
    detection_request = make_request(
        question=None,
        query_source="vision_detections",
    )
    result = decide(
        request=detection_request,
        status="skipped_context",
    )
    assert result.execution_status == "UNAVAILABLE"

    # A usable context cannot justify skipping retrieval.
    independent = ContextAssessment(
        status=ContextStatus.INDEPENDENT,
        reason="The question is independent.",
    )
    result = decide(
        context=independent,
        status="skipped_context",
    )
    assert result.execution_status == "UNAVAILABLE"

    # Failed status cannot hide a context-blocked question retrieval.
    result = decide(
        context=ambiguous,
        status="failed",
    )
    assert result.execution_status == "UNAVAILABLE"

    # A genuinely unresolved context may be reported as skipped.
    result = decide(
        context=ambiguous,
        status="skipped_context",
    )
    assert result.execution_status == "DECIDED"
    assert result.action == "INSUFFICIENT_EVIDENCE"

    # No result with failed execution is valid when no context blocks it.
    result = decide(status="failed")
    assert result.execution_status == "DECIDED"
    assert result.action == "INSUFFICIENT_EVIDENCE"

    # Disabled retrieval with no result is a valid execution state.
    result = decide(request=disabled_request, status="disabled")
    assert result.execution_status == "DECIDED"


def test_evidence_registry_error_blocks_final_decision():
    request = make_request()
    registry = make_registry(request)
    error = ErrorInfo(
        code="REGISTRY_DOWN",
        stage="evidence_registry",
        message="Evidence registry is unavailable.",
    )

    result = FinalDecisionEngine().decide(
        request=request,
        evidence_registry=registry,
        errors=(error,),
    )

    assert result.execution_status == "UNAVAILABLE"
    assert result.action is None
    assert any(
        entry.stage == "error.evidence_registry" for entry in result.decision_trace
    )


def test_vision_error_does_not_invalidate_independent_retrieval():
    request = make_request()
    registry = make_registry(request)
    retrieval, item = make_retrieval()
    register_retrieval(registry, request, retrieval, item)

    error = ErrorInfo(
        code="VISION_FAILED",
        stage="vision",
        message="Vision execution failed.",
    )

    result = FinalDecisionEngine().decide(
        request=request,
        evidence_registry=registry,
        retrieval=retrieval,
        errors=(error,),
    )

    assert result.execution_status == "DECIDED"
    assert result.action == "DIRECT"
    assert result.evidence_ids == ["retrieval:index-v1:1:0000:chunk-1"]
    assert any(entry.stage == "error.vision" for entry in result.decision_trace)


def test_retrieval_error_does_not_invalidate_detection_vision():
    request = make_request(
        question=None,
        query_source="vision_detections",
        run_retrieval=True,
    )
    registry = make_registry(request)
    vision = make_vision()
    register_vision(registry, request, vision)

    error = ErrorInfo(
        code="RETRIEVAL_FAILED",
        stage="retrieval",
        message="Retrieval execution failed.",
    )

    result = FinalDecisionEngine().decide(
        request=request,
        evidence_registry=registry,
        vision=vision,
        errors=(error,),
    )

    assert result.execution_status == "DECIDED"
    assert result.action == "DIRECT"
    assert len(result.evidence_ids) == 2
    assert any(entry.stage == "error.retrieval" for entry in result.decision_trace)


def test_generation_error_is_non_blocking_for_final_decision():
    request = make_request()
    registry = make_registry(request)
    retrieval, item = make_retrieval()
    register_retrieval(registry, request, retrieval, item)

    error = ErrorInfo(
        code="GENERATION_FAILED",
        stage="generation",
        message="Generation failed after decision dependencies completed.",
    )

    result = FinalDecisionEngine().decide(
        request=request,
        evidence_registry=registry,
        retrieval=retrieval,
        errors=(error,),
    )

    assert result.execution_status == "DECIDED"
    assert result.action == "DIRECT"
    assert any(entry.stage == "error.generation" for entry in result.decision_trace)


def test_context_error_is_superseded_by_valid_successful_recovery():
    request = make_request()
    registry = make_registry(request)
    retrieval, item = make_retrieval()
    register_retrieval(registry, request, retrieval, item)

    recovered_context = ContextAssessment(
        status=ContextStatus.INDEPENDENT,
        reason="The recovered assessment confirms an independent question.",
    )
    recovery = ContextRecoveryResult(
        assessment=recovered_context,
        resolved=True,
        attempts=(),
        reason="Recovery produced a valid assessment.",
    )
    error = ErrorInfo(
        code="CONTEXT_ANALYSIS_FAILED",
        stage="context_analysis",
        message="Initial context analysis failed.",
    )

    result = FinalDecisionEngine().decide(
        request=request,
        evidence_registry=registry,
        retrieval=retrieval,
        context_recovery=recovery,
        errors=(error,),
    )

    assert result.execution_status == "DECIDED"
    assert result.action == "DIRECT"
    assert result.evidence_ids == ["retrieval:index-v1:1:0000:chunk-1"]
    assert any(
        entry.stage == "error.context_analysis" for entry in result.decision_trace
    )


def test_context_error_without_successful_recovery_blocks_question_retrieval():
    request = make_request()
    registry = make_registry(request)
    retrieval, item = make_retrieval()
    register_retrieval(registry, request, retrieval, item)

    error = ErrorInfo(
        code="CONTEXT_VALIDATION_FAILED",
        stage="context_validation",
        message="Context validation failed.",
    )

    result = FinalDecisionEngine().decide(
        request=request,
        evidence_registry=registry,
        retrieval=retrieval,
        errors=(error,),
    )

    assert result.execution_status == "DECIDED"
    assert result.action == "INSUFFICIENT_EVIDENCE"
    assert result.evidence_ids == []
    assert any(
        entry.reason_code == "CONTEXT_ERROR_WITHOUT_VALID_RECOVERY"
        for entry in result.decision_trace
    )


def test_input_and_orchestration_errors_are_traced_and_revalidated():
    request = make_request()
    registry = make_registry(request)
    retrieval, item = make_retrieval()
    register_retrieval(registry, request, retrieval, item)

    errors = (
        ErrorInfo(
            code="INPUT_WARNING",
            stage="input",
            message="An input-stage warning was reported.",
        ),
        ErrorInfo(
            code="ORCHESTRATION_WARNING",
            stage="orchestration",
            message="An orchestration-stage warning was reported.",
        ),
    )

    result = FinalDecisionEngine().decide(
        request=request,
        evidence_registry=registry,
        retrieval=retrieval,
        errors=errors,
    )

    assert result.execution_status == "DECIDED"
    assert result.action == "DIRECT"
    trace_stages = [entry.stage for entry in result.decision_trace]
    assert "error.input" in trace_stages
    assert "error.orchestration" in trace_stages


def test_repeated_modality_errors_are_traced_without_cross_modality_failure():
    request = make_request()
    registry = make_registry(request)
    retrieval, item = make_retrieval()
    register_retrieval(registry, request, retrieval, item)

    errors = (
        ErrorInfo(
            code="VISION_FAILED_A",
            stage="vision",
            message="First vision error.",
        ),
        ErrorInfo(
            code="VISION_FAILED_B",
            stage="vision",
            message="Second vision error.",
        ),
        ErrorInfo(
            code="GENERATION_FAILED",
            stage="generation",
            message="Generation error.",
        ),
    )

    result = FinalDecisionEngine().decide(
        request=request,
        evidence_registry=registry,
        retrieval=retrieval,
        errors=errors,
    )

    assert result.execution_status == "DECIDED"
    assert result.action == "DIRECT"

    vision_error_entries = [
        entry for entry in result.decision_trace if entry.stage == "error.vision"
    ]
    assert len(vision_error_entries) == 2
    assert any(entry.stage == "error.generation" for entry in result.decision_trace)


def test_skipped_dependency_is_limited_to_vision_dependency():
    request = make_request(query_source="vision_detections")
    no_detections = make_vision().model_copy(
        update={"status": "no_detections", "detections": []}
    )

    derive = FinalDecisionEngine._derive_retrieval_status
    assert derive(request, None, None, None, (), None) == "skipped_dependency"
    assert derive(request, None, None, no_detections, (), None) == "skipped_dependency"
    assert derive(request, None, None, make_vision(), (), None) == "failed"

    question_request = make_request(query_source="question")
    assert derive(question_request, None, None, None, (), None) == "failed"


def test_skipped_dependency_status_rejects_unrelated_or_available_vision():
    request = make_request(query_source="vision_detections")
    no_detections = make_vision().model_copy(
        update={"status": "no_detections", "detections": []}
    )
    validate = FinalDecisionEngine._validate_retrieval_execution_consistency

    validate(request, None, None, None, "skipped_dependency")

    for invalid_request, invalid_vision in (
        (request, make_vision()),
        (make_request(query_source="question"), no_detections),
    ):
        try:
            validate(
                invalid_request,
                None,
                None,
                invalid_vision,
                "skipped_dependency",
            )
        except ValueError:
            continue
        raise AssertionError("Invalid skipped_dependency status was accepted.")


def test_final_decision_returns_unavailable_when_registry_is_missing():
    result = FinalDecisionEngine().decide(
        request=make_request(),
        evidence_registry=None,
    )
    assert result.execution_status == "UNAVAILABLE"


# OPTIONAL_IMAGE_REGRESSION_TESTS


def test_vision_evidence_matching_handles_missing_request_image():
    request = make_request(include_image=False)
    vision = make_vision().model_copy(update={"artifact_id": None})

    try:
        FinalDecisionEngine._match_vision_evidence(
            request=request,
            vision=vision,
            records=[],
        )
    except ValueError as exc:
        assert "registered snapshot" in str(exc)
    else:
        raise AssertionError("Expected missing snapshot validation to fail.")
