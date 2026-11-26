"""Mock-based tests for pipeline orchestration."""

import inspect
from datetime import UTC, datetime
from unittest.mock import Mock

from smart_waste_multimodal.adapters.interfaces import (
    GenerationStage,
    RetrievalStage,
    VisionStage,
)
from smart_waste_multimodal.contracts import (
    AnalyzeRequest,
    AnalyzeResponse,
    Detection,
    EvidenceItem,
    GenerationResult,
    ImageArtifact,
    RequestOptions,
    RetrievalResult,
    VisionResult,
)
from smart_waste_multimodal.contracts.final_decision import (
    FinalDecisionResult,
)
from smart_waste_multimodal.evidence import (
    ConversationScopedInMemoryEvidenceRegistryFactory,
)
from smart_waste_multimodal.orchestration.orchestrator import (
    PipelineOrchestrator,
)


def make_request(
    *,
    options=None,
    question="How should I recycle this item?",
    include_image=True,
):
    return AnalyzeRequest(
        request_id="mock-request-001",
        conversation_id="test-conversation-test_orchestration-36",
        turn_id="test-turn-test_orchestration-36",
        image=(
            ImageArtifact(
                artifact_id="image-001",
                media_type="image/jpeg",
                width=100,
                height=100,
            )
            if include_image
            else None
        ),
        question=question,
        options=options or RequestOptions(),
    )


def make_vision():
    return VisionResult(
        status="success",
        detections=[
            Detection(
                class_name="plastic",
                confidence=0.95,
                bbox_xyxy=(10, 10, 40, 40),
            )
        ],
        image_width=100,
        image_height=100,
        model_version="mock-vision-v1",
    )


def make_no_detections():
    return VisionResult(
        status="no_detections",
        detections=[],
        image_width=100,
        image_height=100,
        model_version="mock-vision-v1",
    )


def make_retrieval():
    return RetrievalResult(
        status="success",
        query="How should I recycle this item?",
        results=[
            EvidenceItem(
                chunk_id="chunk-001",
                source="mock-recycling-guide",
                text="Clean plastic containers before recycling.",
                score=0.82,
                rank=1,
            )
        ],
        index_version="mock-index-v1",
    )


def make_generation():
    return GenerationResult(
        status="success",
        answer="Clean the plastic container before recycling.",
        citation_chunk_ids=[],
        model_version="mock-generator-v1",
    )


class _IndependentContextAnalyzer:
    """Deterministic analyzer for orchestration tests."""

    def analyze(self, query, conversation_history):
        from smart_waste_multimodal.contracts.context import (
            ContextAnalysisOutput,
        )

        return ContextAnalysisOutput(
            status="independent",
            reason="No conversational dependency is present.",
        )


def _make_independent_context_analyzer():
    return _IndependentContextAnalyzer()


def make_orchestrator(
    vision=None,
    retrieval=None,
    generation=None,
    registry_factory=None,
    final_decision_engine=None,
):
    return PipelineOrchestrator(
        vision_stage=vision or Mock(return_value=make_vision()),
        retrieval_stage=retrieval or Mock(return_value=make_retrieval()),
        generation_stage=generation or Mock(return_value=make_generation()),
        context_analyzer=_make_independent_context_analyzer(),
        evidence_registry_factory=(
            registry_factory or ConversationScopedInMemoryEvidenceRegistryFactory()
        ),
        final_decision_engine=final_decision_engine,
    )


def test_all_stages_succeed():
    vision = Mock(return_value=make_vision())
    retrieval = Mock(return_value=make_retrieval())
    generation = Mock(return_value=make_generation())

    response = make_orchestrator(vision, retrieval, generation).run(
        (_test_request := make_request()),
        conversation=_test_conversation_for_request(_test_request),
    )

    assert response.status == "success"
    assert response.vision is not None
    assert response.retrieval is not None
    assert response.generation is not None
    assert response.errors == []
    assert response.warnings == []
    vision.assert_called_once()
    retrieval.assert_called_once()
    generation.assert_called_once()
    assert len(generation.call_args.args) == 2
    assert generation.call_args.args[1] is response.final_evidence_context


def test_independent_retrieval_runs_even_if_vision_fails():
    vision = Mock(side_effect=RuntimeError("mock vision failure"))
    retrieval = Mock(return_value=make_retrieval())
    generation = Mock(return_value=make_generation())

    response = make_orchestrator(vision, retrieval, generation).run(
        (_test_request := make_request()),
        conversation=_test_conversation_for_request(_test_request),
    )

    assert response.status == "partial_success"
    assert response.vision is None
    assert response.retrieval is not None
    assert response.errors[0].stage == "vision"
    retrieval.assert_called_once()
    generation.assert_called_once()


def test_dependent_retrieval_runs_after_vision():
    options = RequestOptions(retrieval_query_source="vision_detections")
    request = make_request(options=options)
    events = []

    def vision_stage(_request):
        events.append("vision")
        return make_vision()

    def retrieval_stage(_request, vision_result):
        events.append("retrieval")
        assert vision_result is not None
        assert vision_result.detections
        return make_retrieval()

    response = make_orchestrator(
        Mock(side_effect=vision_stage),
        Mock(side_effect=retrieval_stage),
        Mock(return_value=make_generation()),
    ).run(
        (_test_request := request),
        conversation=_test_conversation_for_request(_test_request),
    )

    assert response.status == "success"
    assert events == ["vision", "retrieval"]


def test_dependent_retrieval_is_not_run_after_vision_failure():
    options = RequestOptions(retrieval_query_source="vision_detections")
    retrieval = Mock(return_value=make_retrieval())
    generation = Mock(
        return_value=GenerationResult(
            status="insufficient_evidence",
            answer="I do not have enough evidence to answer.",
            reason="Vision did not provide detections for retrieval.",
        )
    )

    response = make_orchestrator(
        Mock(side_effect=RuntimeError("vision unavailable")),
        retrieval,
        generation,
    ).run(
        (_test_request := make_request(options=options)),
        conversation=_test_conversation_for_request(_test_request),
    )

    assert response.status == "failed"
    assert response.vision is None
    assert response.retrieval is None
    assert response.generation is None
    assert response.errors[0].stage == "vision"
    assert response.warnings
    retrieval.assert_not_called()
    generation.assert_not_called()


def test_dependent_retrieval_is_not_run_when_no_detections():
    options = RequestOptions(retrieval_query_source="vision_detections")
    retrieval = Mock(return_value=make_retrieval())
    generation = Mock(
        return_value=GenerationResult(
            status="insufficient_evidence",
            answer="I do not have enough evidence to answer.",
            reason="No objects were detected.",
        )
    )

    response = make_orchestrator(
        Mock(return_value=make_no_detections()),
        retrieval,
        generation,
    ).run(
        (_test_request := make_request(options=options)),
        conversation=_test_conversation_for_request(_test_request),
    )

    assert response.status == "partial_success"
    assert response.vision is not None
    assert response.vision.status == "no_detections"
    assert response.retrieval is None
    assert response.generation is None
    assert response.warnings
    retrieval.assert_not_called()
    generation.assert_not_called()


def test_disabled_vision_is_never_called():
    options = RequestOptions(run_vision=False)
    vision = Mock(return_value=make_vision())

    response = make_orchestrator(vision=vision).run(
        (_test_request := make_request(options=options)),
        conversation=_test_conversation_for_request(_test_request),
    )

    assert response.status == "success"
    assert response.vision is None
    vision.assert_not_called()


def test_generation_failure_preserves_upstream_results():
    generation = Mock(side_effect=RuntimeError("mock generation failure"))

    response = make_orchestrator(generation=generation).run(
        (_test_request := make_request()),
        conversation=_test_conversation_for_request(_test_request),
    )

    assert response.status == "partial_success"
    assert response.vision is not None
    assert response.retrieval is not None
    assert response.generation is None
    assert any(error.stage == "generation" for error in response.errors)


def test_all_stages_failing_produces_failed_response():
    options = RequestOptions(retrieval_query_source="question")

    response = make_orchestrator(
        vision=Mock(side_effect=RuntimeError("vision failed")),
        retrieval=Mock(side_effect=RuntimeError("retrieval failed")),
        generation=Mock(side_effect=RuntimeError("generation failed")),
    ).run(
        (_test_request := make_request(options=options)),
        conversation=_test_conversation_for_request(_test_request),
    )

    assert response.status == "failed"
    assert response.vision is None
    assert response.retrieval is None
    assert response.generation is None
    assert {error.stage for error in response.errors} == {"vision", "retrieval"}


def test_nonblocking_warning_produces_partial_success():
    # Simulate a future non-blocking warning added by orchestration.
    response = AnalyzeResponse(
        request_id="warning-request",
        status="partial_success",
        requested_options=RequestOptions(),
        vision=make_vision(),
        retrieval=make_retrieval(),
        generation=make_generation(),
        warnings=["A non-blocking stage warning."],
    )

    assert response.status == "partial_success"
    assert response.warnings


def test_retrieval_failure_does_not_erase_vision_result():
    response = make_orchestrator(
        retrieval=Mock(side_effect=RuntimeError("retrieval unavailable")),
        generation=Mock(
            return_value=GenerationResult(
                status="insufficient_evidence",
                answer="I do not have enough evidence to answer.",
                reason="Retrieval failed.",
            )
        ),
    ).run(
        (_test_request := make_request()),
        conversation=_test_conversation_for_request(_test_request),
    )

    assert response.status == "partial_success"
    assert response.vision is not None
    assert response.retrieval is None
    assert any(error.stage == "retrieval" for error in response.errors)


def test_disabled_dependent_retrieval_does_not_emit_warning():
    options = RequestOptions(
        run_vision=False,
        run_retrieval=False,
        generate_answer=True,
        retrieval_query_source="vision_detections",
    )
    vision = Mock(return_value=make_vision())
    retrieval = Mock(return_value=make_retrieval())
    generation = Mock(
        return_value=GenerationResult(
            status="success",
            answer="The request was handled without running optional stages.",
        )
    )

    response = make_orchestrator(vision, retrieval, generation).run(
        (_test_request := make_request(options=options)),
        conversation=_test_conversation_for_request(_test_request),
    )

    assert response.status == "success"
    assert response.vision is None
    assert response.retrieval is None
    assert response.warnings == []
    assert response.generation is None
    vision.assert_not_called()
    retrieval.assert_not_called()
    generation.assert_not_called()


def test_adapter_protocol_signatures_are_stable():
    assert list(inspect.signature(VisionStage.__call__).parameters) == [
        "self",
        "request",
    ]
    assert list(inspect.signature(RetrievalStage.__call__).parameters) == [
        "self",
        "request",
        "vision",
        "context_assessment",
    ]
    assert list(inspect.signature(GenerationStage.__call__).parameters) == [
        "self",
        "request",
        "final_evidence_context",
    ]


def test_stage_error_does_not_expose_raw_exception_details():
    secret_detail = "/private/weights/model.pt API_TOKEN=example-secret"

    error = PipelineOrchestrator._error(
        "vision",
        RuntimeError(secret_detail),
    )

    assert error.stage == "vision"
    assert error.code == "vision_stage_failed"
    assert error.message == "The vision stage could not complete."
    assert secret_detail not in error.message
    assert "/private/weights/model.pt" not in error.message
    assert "example-secret" not in error.message


def test_wrong_vision_result_type_is_handled_and_independent_stages_continue():
    vision = Mock(return_value={"unexpected": "dictionary"})
    retrieval = Mock(return_value=make_retrieval())
    generation = Mock(return_value=make_generation())

    response = make_orchestrator(
        vision=vision,
        retrieval=retrieval,
        generation=generation,
    ).run(
        (
            _test_request := make_request(
                options=RequestOptions(
                    run_vision=True,
                    run_retrieval=True,
                    generate_answer=True,
                    retrieval_query_source="question",
                )
            )
        ),
        conversation=_test_conversation_for_request(_test_request),
    )

    assert response.status == "partial_success"
    assert response.vision is None
    assert response.retrieval is not None
    assert response.generation is not None
    assert any(error.stage == "vision" for error in response.errors)
    assert all(
        error.message == "The vision stage could not complete."
        for error in response.errors
        if error.stage == "vision"
    )
    retrieval.assert_called_once()
    generation.assert_called_once()


def test_wrong_retrieval_result_type_is_handled_as_stage_failure():
    retrieval = Mock(return_value={"unexpected": "dictionary"})

    response = make_orchestrator(
        retrieval=retrieval,
    ).run(
        (
            _test_request := make_request(
                options=RequestOptions(
                    run_vision=False,
                    run_retrieval=True,
                    generate_answer=False,
                    retrieval_query_source="question",
                )
            )
        ),
        conversation=_test_conversation_for_request(_test_request),
    )

    assert response.status == "failed"
    assert response.retrieval is None
    assert len(response.errors) == 1
    assert response.errors[0].stage == "retrieval"
    assert response.errors[0].code == "retrieval_stage_failed"


def test_wrong_generation_result_type_preserves_upstream_results():
    generation = Mock(return_value={"unexpected": "dictionary"})

    response = make_orchestrator(
        generation=generation,
    ).run(
        (_test_request := make_request()),
        conversation=_test_conversation_for_request(_test_request),
    )

    assert response.status == "partial_success"
    assert response.vision is not None
    assert response.retrieval is not None
    assert response.generation is None
    assert any(error.stage == "generation" for error in response.errors)
    assert all(
        error.message == "The generation stage could not complete."
        for error in response.errors
        if error.stage == "generation"
    )


def test_none_stage_result_is_handled_as_stage_failure():
    vision = Mock(return_value=None)

    response = make_orchestrator(
        vision=vision,
    ).run(
        (
            _test_request := make_request(
                options=RequestOptions(
                    run_vision=True,
                    run_retrieval=False,
                    generate_answer=False,
                )
            )
        ),
        conversation=_test_conversation_for_request(_test_request),
    )

    assert response.status == "failed"
    assert response.vision is None
    assert len(response.errors) == 1
    assert response.errors[0].stage == "vision"
    assert response.errors[0].code == "vision_stage_failed"
    assert response.errors[0].message == ("The vision stage could not complete.")
    vision.assert_called_once()


def test_retrieval_does_not_control_human_review_or_generation():
    options = RequestOptions(
        run_vision=False,
        run_retrieval=True,
        generate_answer=True,
        retrieval_query_source="question",
    )

    retrieval = Mock(return_value=make_retrieval())
    generation = Mock(return_value=make_generation())
    vision = Mock(return_value=make_vision())

    response = make_orchestrator(
        vision=vision,
        retrieval=retrieval,
        generation=generation,
    ).run(
        (
            _test_request := make_request(
                options=options,
                question="How should I recycle this item?",
            )
        ),
        conversation=_test_conversation_for_request(_test_request),
    )

    assert response.status == "success"
    assert response.final_decision is not None
    assert response.final_decision.execution_status == "DECIDED"
    assert response.final_decision.action == "DIRECT"
    assert response.retrieval is not None
    assert response.retrieval.status == "success"
    assert response.generation is not None
    assert response.clarification_message is None
    assert response.errors == []
    assert response.warnings == []

    vision.assert_not_called()
    retrieval.assert_called_once()
    generation.assert_called_once()


# CONTEXT HISTORY LIMIT TESTS


def _make_request_with_history(count):
    from datetime import datetime, timedelta

    from smart_waste_multimodal.contracts import ConversationTurn

    now = datetime.now(UTC)

    history = [
        ConversationTurn(
            turn_id=f"turn-{index:03d}",
            role="user",
            content=f"Conversation message {index}",
            timestamp=now - timedelta(minutes=count - index),
        )
        for index in range(count)
    ]

    return make_request().model_copy(update={"conversation_history": history})


def test_orchestrator_keeps_newest_twenty_turns_by_default():

    request = _make_request_with_history(22)

    vision = Mock(return_value=make_vision())
    retrieval = Mock(return_value=make_retrieval())
    generation = Mock(return_value=make_generation())

    orchestrator = PipelineOrchestrator(
        vision_stage=vision,
        retrieval_stage=retrieval,
        generation_stage=generation,
        context_analyzer=_make_independent_context_analyzer(),
        evidence_registry_factory=ConversationScopedInMemoryEvidenceRegistryFactory(),
    )

    response = orchestrator.run(
        (_test_request := request),
        conversation=_test_conversation_for_request(_test_request),
    )

    assert response.status == "success"

    # The caller's original request must remain unchanged.
    assert len(request.conversation_history) == 22

    # Every stage must receive the bounded request.
    for stage_mock in (vision, retrieval, generation):
        stage_request = stage_mock.call_args.args[0]
        assert len(stage_request.conversation_history) == 20
        assert [turn.turn_id for turn in stage_request.conversation_history] == [
            f"turn-{index:03d}" for index in range(2, 22)
        ]

    assert vision.call_count == 1
    assert retrieval.call_count == 1
    assert generation.call_count == 1


def test_orchestrator_uses_overridden_history_limit():
    from smart_waste_multimodal.config import ContextConfig

    request = _make_request_with_history(8)

    vision = Mock(return_value=make_vision())
    retrieval = Mock(return_value=make_retrieval())
    generation = Mock(return_value=make_generation())

    orchestrator = PipelineOrchestrator(
        vision_stage=vision,
        retrieval_stage=retrieval,
        generation_stage=generation,
        context_config=ContextConfig(max_history_turns=5),
        context_analyzer=_make_independent_context_analyzer(),
        evidence_registry_factory=ConversationScopedInMemoryEvidenceRegistryFactory(),
    )

    response = orchestrator.run(
        (_test_request := request),
        conversation=_test_conversation_for_request(_test_request),
    )

    assert response.status == "success"
    assert len(request.conversation_history) == 8

    for stage_mock in (vision, retrieval, generation):
        stage_request = stage_mock.call_args.args[0]

        assert len(stage_request.conversation_history) == 5
        assert [turn.turn_id for turn in stage_request.conversation_history] == [
            "turn-003",
            "turn-004",
            "turn-005",
            "turn-006",
            "turn-007",
        ]


def test_orchestrator_preserves_request_when_history_is_within_limit():
    request = _make_request_with_history(20)

    vision = Mock(return_value=make_vision())
    retrieval = Mock(return_value=make_retrieval())
    generation = Mock(return_value=make_generation())

    orchestrator = PipelineOrchestrator(
        vision_stage=vision,
        retrieval_stage=retrieval,
        generation_stage=generation,
        context_analyzer=_make_independent_context_analyzer(),
        evidence_registry_factory=ConversationScopedInMemoryEvidenceRegistryFactory(),
    )

    response = orchestrator.run(
        (_test_request := request),
        conversation=_test_conversation_for_request(_test_request),
    )

    assert response.status == "success"

    # No copy or truncation is needed when the history is within the limit.
    assert vision.call_args.args[0] is request
    assert retrieval.call_args.args[0] is request
    assert generation.call_args.args[0] is request
    assert len(request.conversation_history) == 20


# ------------------------------------------------------------
# Evidence Registry pipeline integration tests
# ------------------------------------------------------------


class _RecordingEvidenceRegistryFactory:
    def __init__(self):
        self.delegate = ConversationScopedInMemoryEvidenceRegistryFactory()
        self.registries = []

    def create(self, conversation, *, actor_id):
        registry = self.delegate.create(conversation, actor_id=actor_id)
        self.registries.append(registry)
        return registry


def test_registry_records_all_evidence_sources_and_provenance():
    from smart_waste_multimodal.contracts import ConversationTurn
    from smart_waste_multimodal.contracts.evidence import EvidenceSourceType

    factory = _RecordingEvidenceRegistryFactory()
    orchestrator = make_orchestrator(registry_factory=factory)

    historical_vision = make_vision().model_copy(
        update={"artifact_id": "historical-image-001"}
    )

    historical_turn = ConversationTurn(
        turn_id="turn-history-001",
        role="user",
        content="What material is visible in this image?",
        timestamp=datetime.now(UTC),
        vision=historical_vision,
    )

    request = make_request().model_copy(
        update={"conversation_history": [historical_turn]}
    )

    response = orchestrator.run(
        (_test_request := request),
        conversation=_test_conversation_for_request(_test_request),
    )

    assert response.generation is not None
    assert len(factory.registries) == 1

    registry = factory.registries[0]
    evidence_items = registry.list_all(actor_id=registry.owner_id)
    evidence_by_id = {item.evidence_id: item for item in evidence_items}

    expected_ids = {
        "conversation:turn-history-001:text",
        "conversation:turn-history-001:vision",
        "vision:image-001:snapshot",
        "vision:image-001:detection:0000",
        "retrieval:mock-index-v1:1:0000:chunk-001",
    }

    assert set(evidence_by_id) == expected_ids
    assert len(evidence_items) == 5

    # Conversation text
    text_evidence = evidence_by_id["conversation:turn-history-001:text"]
    assert text_evidence.source_type == EvidenceSourceType.CONVERSATION_TEXT
    assert text_evidence.provenance.source_id == "turn-history-001"
    assert text_evidence.content["text"] == historical_turn.content

    # Historical vision snapshot
    historical_snapshot = evidence_by_id["conversation:turn-history-001:vision"]
    assert historical_snapshot.source_type == EvidenceSourceType.VISION_SNAPSHOT
    assert historical_snapshot.provenance.source_id == "turn-history-001"
    assert historical_snapshot.provenance.source_version == (
        historical_vision.model_version
    )

    # Current vision snapshot
    current_snapshot = evidence_by_id["vision:image-001:snapshot"]
    assert current_snapshot.source_type == EvidenceSourceType.VISION_SNAPSHOT
    assert current_snapshot.provenance.source_id == "image-001"

    # Individual detection
    detection = evidence_by_id["vision:image-001:detection:0000"]
    assert detection.source_type == EvidenceSourceType.VISION_DETECTION
    assert detection.provenance.source_id == "image-001"
    assert detection.provenance.source_locator == {
        "artifact_id": "image-001",
        "detection_index": "0",
    }

    # Retrieval chunk
    retrieval_chunk = evidence_by_id["retrieval:mock-index-v1:1:0000:chunk-001"]
    assert retrieval_chunk.source_type == EvidenceSourceType.RETRIEVAL_CHUNK
    assert retrieval_chunk.provenance.source_id == "chunk-001"
    assert retrieval_chunk.provenance.source_version == "mock-index-v1"
    assert retrieval_chunk.content["text"] == (
        "Clean plastic containers before recycling."
    )

    # Historical evidence must preserve its original creation timestamp.
    historical_ids = {
        "conversation:turn-history-001:text",
        "conversation:turn-history-001:vision",
    }
    historical_items = [
        item for item in evidence_items if item.evidence_id in historical_ids
    ]
    current_items = [
        item for item in evidence_items if item.evidence_id not in historical_ids
    ]

    assert len(historical_items) == 2
    assert all(
        item.created_at == historical_turn.timestamp for item in historical_items
    )

    # Evidence produced during the current request shares its creation timestamp.
    assert current_items
    assert len({item.created_at for item in current_items}) == 1

    # All records belong to the same request.
    assert all(item.request_id == request.request_id for item in evidence_items)


def test_orchestrator_reuses_registry_across_requests_in_same_conversation():
    factory = _RecordingEvidenceRegistryFactory()
    orchestrator = make_orchestrator(registry_factory=factory)

    request_one = make_request().model_copy(
        update={"request_id": "conversation-request-001"}
    )
    conversation = _test_conversation_for_request(request_one)

    request_two = make_request().model_copy(
        update={
            "request_id": "conversation-request-002",
            "conversation_id": request_one.conversation_id,
            "turn_id": "conversation-turn-002",
        }
    )

    response_one = orchestrator.run(
        request_one,
        conversation=conversation,
    )
    response_two = orchestrator.run(
        request_two,
        conversation=conversation,
    )

    assert response_one.generation is not None
    assert response_two.generation is not None

    # One conversation must retain one shared registry.
    assert len(factory.registries) == 2
    registry_one, registry_two = factory.registries

    assert registry_one is registry_two
    assert registry_one.conversation_id == conversation.conversation_id

    evidence_items = registry_one.list_all(actor_id=conversation.owner_id)

    assert evidence_items

    # Both requests produce the same evidence IDs in this test fixture.
    # Existing records must be preserved without overwriting their provenance.
    creator_request_ids = {item.created_by_request_id for item in evidence_items}

    assert creator_request_ids == {"conversation-request-001"}
    assert len({item.evidence_id for item in evidence_items}) == len(evidence_items)


def test_registry_registration_failure_blocks_generation():
    secret_detail = "PRIVATE_REGISTRY_FAILURE_DETAIL"

    class FailingRegistry:
        def __init__(self, conversation):
            self.conversation_id = conversation.conversation_id
            self.owner_id = conversation.owner_id
            self.status = conversation.status

        def register(self, evidence, *, actor_id):
            raise RuntimeError(secret_detail)

    class FailingRegistryFactory:
        def create(self, conversation, *, actor_id):
            return FailingRegistry(conversation)

    generation = Mock(return_value=make_generation())

    orchestrator = make_orchestrator(
        generation=generation,
        registry_factory=FailingRegistryFactory(),
    )

    response = orchestrator.run(
        (_test_request := make_request()),
        conversation=_test_conversation_for_request(_test_request),
    )

    assert response.generation is None
    assert response.errors

    generation.assert_not_called()

    # Internal exception details must not leak into the response.
    response_text = str(response.model_dump(mode="json"))
    assert secret_detail not in response_text

    # The registry failure must be represented as a pipeline error.
    assert any(
        getattr(error, "stage", None) == "evidence_registry"
        for error in response.errors
    )


def test_final_evidence_context_builder_failure_blocks_generation():
    generation = Mock(return_value=make_generation())
    orchestrator = make_orchestrator(generation=generation)

    builder = Mock()
    builder.build.side_effect = RuntimeError(
        "mock final evidence context builder failure"
    )
    orchestrator.final_evidence_context_builder = builder

    response = orchestrator.run(
        (_test_request := make_request()),
        conversation=_test_conversation_for_request(_test_request),
    )

    builder.build.assert_called_once()
    generation.assert_not_called()

    assert response.final_evidence_context is None
    assert response.generation is None
    assert response.vision is not None
    assert response.retrieval is not None
    assert any(error.stage == "final_evidence_context" for error in response.errors)


# OPTIONAL_IMAGE_REGRESSION_TESTS


def test_question_retrieval_continues_without_current_image():
    vision = Mock(return_value=make_vision())
    retrieval = Mock(return_value=make_retrieval())
    generation = Mock(return_value=make_generation())

    request = make_request(include_image=False)

    response = make_orchestrator(
        vision=vision,
        retrieval=retrieval,
        generation=generation,
    ).run(
        (_test_request := request),
        conversation=_test_conversation_for_request(_test_request),
    )

    assert response.status == "success"
    assert response.image_provided is False
    assert response.vision is None
    assert response.retrieval is not None
    assert response.generation is not None
    assert response.errors == []
    assert response.warnings == []

    vision.assert_not_called()
    retrieval.assert_called_once()
    generation.assert_called_once()


def test_detection_retrieval_without_image_fails_as_missing_prerequisite():
    vision = Mock(return_value=make_vision())
    retrieval = Mock(return_value=make_retrieval())
    generation = Mock(return_value=make_generation())

    request = make_request(
        include_image=False,
        options=RequestOptions(
            run_vision=True,
            run_retrieval=True,
            generate_answer=True,
            retrieval_query_source="vision_detections",
        ),
    )

    response = make_orchestrator(
        vision=vision,
        retrieval=retrieval,
        generation=generation,
    ).run(
        (_test_request := request),
        conversation=_test_conversation_for_request(_test_request),
    )

    assert response.status == "failed"
    assert response.image_provided is False
    assert response.vision is None
    assert response.retrieval is None
    assert any(error.stage == "retrieval" for error in response.errors)

    vision.assert_not_called()
    retrieval.assert_not_called()
    generation.assert_not_called()


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

def test_orchestrator_forwards_registry_actor_and_current_turn_metadata():
    from datetime import datetime, timezone

    from smart_waste_multimodal.contracts.context import (
        ContextAnalysisOutput,
        ContextAssessment,
        ContextStatus,
    )
    from smart_waste_multimodal.orchestration.context_recovery import (
        ContextRecoveryResult,
    )

    timestamp = datetime(2026, 11, 21, 10, 30, tzinfo=timezone.utc)
    assessment = ContextAssessment(
        status=ContextStatus.AMBIGUOUS,
        reason="Controlled unresolved assessment for orchestration testing.",
    )

    class Analyzer:
        def analyze(self, query, conversation_history):
            return ContextAnalysisOutput(
                status=ContextStatus.AMBIGUOUS,
                reason="Controlled ambiguous context.",
            )

    class Validator:
        def __init__(self):
            self.call = None

        def validate(self, output, history, registry, **kwargs):
            self.call = (output, history, registry, kwargs)
            return assessment

    class Recovery:
        def __init__(self):
            self.call = None

        def resolve(self, **kwargs):
            self.call = kwargs
            return ContextRecoveryResult(
                assessment=assessment,
                resolved=False,
                attempts=(),
                reason="Controlled unresolved recovery.",
            )

    request = make_request().model_copy(
        update={
            "turn_id": "current-turn-wiring-test",
            "turn_timestamp": timestamp,
            "conversation_history": [],
        }
    )
    conversation = _test_conversation_for_request(request)
    factory = _RecordingEvidenceRegistryFactory()
    validator = Validator()
    recovery = Recovery()

    orchestrator = PipelineOrchestrator(
        vision_stage=Mock(return_value=make_vision()),
        retrieval_stage=Mock(return_value=make_retrieval()),
        generation_stage=Mock(return_value=make_generation()),
        evidence_registry_factory=factory,
        context_analyzer=Analyzer(),
        context_validator=validator,
        context_recovery_orchestrator=recovery,
    )

    orchestrator.run(request, conversation=conversation)

    assert validator.call is not None
    _, validator_history, validator_registry, validator_kwargs = validator.call

    assert validator_history == []
    assert len(factory.registries) == 1
    assert validator_registry is factory.registries[0]
    assert validator_kwargs["actor_id"] == conversation.owner_id
    assert validator_kwargs["current_turn_id"] == request.turn_id
    assert validator_kwargs["current_turn_timestamp"] == request.turn_timestamp

    assert recovery.call is not None
    assert recovery.call["registry"] is validator_registry
    assert recovery.call["actor_id"] == conversation.owner_id
    assert recovery.call["current_turn_id"] == request.turn_id
    assert recovery.call["current_turn_timestamp"] == request.turn_timestamp


def test_orchestrator_registers_current_turn_vision_evidence_before_recovery():
    from datetime import datetime, timezone

    from smart_waste_multimodal.contracts.context import (
        ContextAnalysisOutput,
        ContextAssessment,
        ContextStatus,
    )
    from smart_waste_multimodal.contracts.evidence import EvidenceSourceType
    from smart_waste_multimodal.orchestration.context_recovery import (
        ContextRecoveryResult,
    )

    timestamp = datetime(2026, 11, 21, 10, 45, tzinfo=timezone.utc)
    assessment = ContextAssessment(
        status=ContextStatus.AMBIGUOUS,
        reason="Controlled unresolved assessment for evidence registration test.",
    )

    class Analyzer:
        def analyze(self, query, conversation_history):
            return ContextAnalysisOutput(
                status=ContextStatus.AMBIGUOUS,
                reason="Controlled ambiguous context.",
            )

    class Validator:
        def validate(self, output, history, registry, **kwargs):
            assert history == []
            assert kwargs["current_turn_id"] == request.turn_id
            assert kwargs["current_turn_timestamp"] == request.turn_timestamp
            return assessment

    class Recovery:
        def __init__(self):
            self.call = None
            self.current_turn_evidence = []

        def resolve(self, **kwargs):
            self.call = kwargs
            registry = kwargs["registry"]
            actor_id = kwargs["actor_id"]

            records = registry.list_all(actor_id=actor_id)
            self.current_turn_evidence = [
                item
                for item in records
                if item.origin_turn_id == request.turn_id
            ]

            return ContextRecoveryResult(
                assessment=assessment,
                resolved=False,
                attempts=(),
                reason="Controlled unresolved recovery.",
            )

    request = make_request().model_copy(
        update={
            "turn_id": "current-turn-evidence-test",
            "turn_timestamp": timestamp,
            "conversation_history": [],
        }
    )
    conversation = _test_conversation_for_request(request)
    factory = _RecordingEvidenceRegistryFactory()
    recovery = Recovery()

    orchestrator = PipelineOrchestrator(
        vision_stage=Mock(return_value=make_vision()),
        retrieval_stage=Mock(return_value=make_retrieval()),
        generation_stage=Mock(return_value=make_generation()),
        evidence_registry_factory=factory,
        context_analyzer=Analyzer(),
        context_validator=Validator(),
        context_recovery_orchestrator=recovery,
    )

    orchestrator.run(request, conversation=conversation)

    assert recovery.call is not None
    assert len(factory.registries) == 1
    assert recovery.call["registry"] is factory.registries[0]
    assert recovery.call["actor_id"] == conversation.owner_id
    assert recovery.call["current_turn_id"] == request.turn_id
    assert recovery.call["current_turn_timestamp"] == request.turn_timestamp

    assert recovery.current_turn_evidence
    assert any(
        item.source_type == EvidenceSourceType.VISION_SNAPSHOT
        for item in recovery.current_turn_evidence
    )
    assert any(
        item.source_type == EvidenceSourceType.VISION_DETECTION
        for item in recovery.current_turn_evidence
    )
    assert all(
        item.origin_turn_id == request.turn_id
        for item in recovery.current_turn_evidence
    )

def test_human_review_returns_needs_clarification_without_generation():
    options = RequestOptions(
        run_vision=False,
        run_retrieval=True,
        generate_answer=True,
        retrieval_query_source="question",
    )

    decision = FinalDecisionResult(
        execution_status="DECIDED",
        action="HUMAN_REVIEW",
        reason_code="HUMAN_ACTION_REQUIRED",
        reason="The request requires user clarification.",
        clarification_required=True,
        clarification_reason="The referenced item is ambiguous.",
    )

    final_decision_engine = Mock()
    final_decision_engine.decide.return_value = decision

    retrieval = Mock(return_value=make_retrieval())
    generation = Mock(return_value=make_generation())
    vision = Mock(return_value=make_vision())

    request = make_request(options=options, include_image=False)

    response = make_orchestrator(
        vision=vision,
        retrieval=retrieval,
        generation=generation,
        final_decision_engine=final_decision_engine,
    ).run(
        request,
        conversation=_test_conversation_for_request(request),
    )

    assert response.status == "needs_clarification"
    assert response.final_decision is not None
    assert response.final_decision.action == "HUMAN_REVIEW"
    assert response.clarification_message == "I need a little more information to answer safely. Please clarify your request."
    assert response.warnings.count(response.clarification_message) == 1
    assert response.generation is None
    assert response.final_evidence_context is None

    retrieval.assert_called_once()
    generation.assert_not_called()
    vision.assert_not_called()
    final_decision_engine.decide.assert_called_once()
