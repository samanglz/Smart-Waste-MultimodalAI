"""Unit tests for inter-module contracts."""

import pytest
from pydantic import ValidationError

from smart_waste_multimodal.contracts import (
    AnalyzeRequest,
    AnalyzeResponse,
    Detection,
    ErrorInfo,
    EvidenceItem,
    FinalDecisionResult,
    GenerationResult,
    ImageArtifact,
    RetrievalResult,
    VisionResult,
)


def test_request_accepts_image_artifact():
    request = AnalyzeRequest(
        request_id="req-001",
        conversation_id="test-conversation-test_contracts-21",
        turn_id="test-turn-test_contracts-21",
        image=ImageArtifact(
            artifact_id="image-001",
            media_type="image/jpeg",
            width=640,
            height=480,
        ),
        question="What material is visible?",
    )
    assert request.contract_version == "1.0"
    assert request.options.run_vision is True


def test_request_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        AnalyzeRequest(
            request_id="req-002",
            conversation_id="test-conversation-test_contracts-37",
            turn_id="test-turn-test_contracts-37",
            image={"artifact_id": "img", "media_type": "image/png"},
            unexpected="not allowed",
        )


def test_detection_validates_confidence_and_box():
    with pytest.raises(ValidationError):
        Detection(
            class_name="plastic",
            confidence=1.2,
            bbox_xyxy=(1, 2, 10, 20),
        )
    with pytest.raises(ValidationError):
        Detection(
            class_name="plastic",
            confidence=0.9,
            bbox_xyxy=(10, 2, 1, 20),
        )


def test_vision_status_matches_detection_list():
    with pytest.raises(ValidationError):
        VisionResult(
            status="no_detections",
            detections=[
                Detection(
                    class_name="paper",
                    confidence=0.8,
                    bbox_xyxy=(1, 2, 10, 20),
                )
            ],
            image_width=100,
            image_height=100,
        )
    result = VisionResult(status="no_detections", image_width=100, image_height=100)
    assert result.detections == []


def test_vision_rejects_box_outside_image():
    with pytest.raises(ValidationError):
        VisionResult(
            status="success",
            detections=[
                Detection(
                    class_name="metal",
                    confidence=0.9,
                    bbox_xyxy=(1, 2, 110, 20),
                )
            ],
            image_width=100,
            image_height=100,
        )


def test_retrieval_distinguishes_no_evidence():
    result = RetrievalResult(status="no_evidence", query="recycling metal", results=[])
    assert result.results == []
    with pytest.raises(ValidationError):
        RetrievalResult(status="success", query="recycling metal", results=[])


def test_retrieval_rejects_legacy_human_review_contract():
    with pytest.raises(ValidationError):
        RetrievalResult(
            status="human_review_required",
            query="What about the other one?",
            results=[],
            review_reason="context_ambiguous",
        )


def _human_review_decision():
    return FinalDecisionResult(
        execution_status="DECIDED",
        action="HUMAN_REVIEW",
        reason_code="HUMAN_ACTION_REQUIRED",
        reason="The request requires user clarification.",
        clarification_required=True,
        clarification_reason="The referenced item is ambiguous.",
    )


def test_needs_clarification_requires_final_human_review_decision():
    with pytest.raises(ValidationError):
        AnalyzeResponse(
            request_id="req-no-final-decision",
            status="needs_clarification",
            clarification_message="Please clarify the referenced item.",
            warnings=["Please clarify the referenced item."],
        )

    response = AnalyzeResponse(
        request_id="req-final-human-review",
        status="needs_clarification",
        final_decision=_human_review_decision(),
        clarification_message="Please clarify the referenced item.",
        warnings=["Please clarify the referenced item."],
    )
    assert response.final_decision.action == "HUMAN_REVIEW"


def test_insufficient_evidence_cannot_request_clarification():
    decision = FinalDecisionResult(
        execution_status="DECIDED",
        action="INSUFFICIENT_EVIDENCE",
        reason_code="INSUFFICIENT_EVIDENCE",
        reason="No sufficient evidence is available.",
    )

    with pytest.raises(ValidationError):
        AnalyzeResponse(
            request_id="req-insufficient-not-review",
            status="needs_clarification",
            final_decision=decision,
            clarification_message="Please clarify.",
            warnings=["Please clarify."],
        )


def test_final_human_review_cannot_use_normal_response_status():
    with pytest.raises(ValidationError):
        AnalyzeResponse(
            request_id="req-review-status-mismatch",
            status="partial_success",
            final_decision=_human_review_decision(),
            warnings=["A stage returned a warning."],
        )


@pytest.mark.parametrize(
    "score",
    [float("nan"), float("inf"), float("-inf")],
    ids=["nan", "positive-infinity", "negative-infinity"],
)
def test_evidence_item_rejects_non_finite_scores(score):
    with pytest.raises(ValidationError, match="score must be finite"):
        EvidenceItem(
            chunk_id="chunk-finite-check",
            source="recycling-guide.pdf",
            text="Recycling guidance.",
            score=score,
            rank=1,
        )


@pytest.mark.parametrize("score", [-2.5, 0.0, 1.0, 4.25])
def test_evidence_item_accepts_finite_scores_outside_probability_range(score):
    item = EvidenceItem(
        chunk_id="chunk-ranking-score",
        source="recycling-guide.pdf",
        text="Retrieval ranking scores are not probabilities.",
        score=score,
        rank=1,
    )
    assert item.score == score


def test_generation_requires_an_answer():
    GenerationResult(
        status="insufficient_evidence",
        answer="I do not have enough evidence to answer reliably.",
        reason="The retrieved evidence does not support a reliable answer.",
    )
    with pytest.raises(ValidationError):
        GenerationResult(status="success", answer="  ")


def test_response_rejects_unknown_citation_ids():
    generation = GenerationResult(
        status="success",
        answer="This appears to be recyclable paper.",
        citation_chunk_ids=["missing-chunk"],
    )
    with pytest.raises(ValidationError):
        AnalyzeResponse(
            request_id="req-003",
            status="success",
            vision=VisionResult(
                status="no_detections",
                image_width=100,
                image_height=100,
            ),
            generation=generation,
            retrieval=RetrievalResult(
                status="success",
                query="paper recycling",
                results=[
                    EvidenceItem(
                        chunk_id="actual-chunk",
                        source="guide.pdf",
                        text="Paper recycling guidance.",
                        score=0.8,
                        rank=1,
                    )
                ],
            ),
        )


def test_failed_response_requires_error():
    with pytest.raises(ValidationError):
        AnalyzeResponse(request_id="req-004", status="failed")
    response = AnalyzeResponse(
        request_id="req-005",
        status="failed",
        errors=[
            ErrorInfo(
                code="VISION_MODEL_UNAVAILABLE",
                stage="vision",
                message="Vision stage could not run.",
                retryable=True,
            )
        ],
    )
    assert response.errors[0].retryable is True


def test_json_schema_is_generated_from_model():
    schema = AnalyzeRequest.model_json_schema()
    assert schema["title"] == "AnalyzeRequest"
    assert "image" in schema["properties"]
    assert "request_id" in schema["properties"]


def test_success_requires_all_requested_stage_outputs():
    with pytest.raises(ValidationError):
        AnalyzeResponse(
            request_id="req-success-missing",
            status="success",
        )


def test_success_accepts_normal_empty_stage_outcomes():
    response = AnalyzeResponse(
        request_id="req-empty-normal",
        status="success",
        vision=VisionResult(
            status="no_detections",
            image_width=100,
            image_height=100,
        ),
        retrieval=RetrievalResult(
            status="no_evidence",
            query="recycling information",
            results=[],
        ),
        generation=GenerationResult(
            status="insufficient_evidence",
            answer="I do not have enough evidence to answer reliably.",
            reason="No relevant evidence was retrieved.",
        ),
    )
    assert response.status == "success"


def test_disabled_stages_can_be_omitted_from_success_response():
    from smart_waste_multimodal.contracts import RequestOptions

    response = AnalyzeResponse(
        request_id="req-disabled-stages",
        status="success",
        requested_options=RequestOptions(
            run_vision=False,
            run_retrieval=True,
            generate_answer=False,
        ),
        retrieval=RetrievalResult(
            status="no_evidence",
            query="recycling information",
            results=[],
        ),
        generation=GenerationResult(
            status="skipped",
            reason="Answer generation was not requested.",
        ),
    )
    assert response.vision is None
    assert response.generation.status == "skipped"


def test_disabled_vision_cannot_return_vision_result():
    from smart_waste_multimodal.contracts import RequestOptions

    with pytest.raises(ValidationError):
        AnalyzeResponse(
            request_id="req-disabled-vision",
            status="success",
            requested_options=RequestOptions(
                run_vision=False,
                run_retrieval=True,
                generate_answer=False,
            ),
            vision=VisionResult(
                status="no_detections",
                image_width=100,
                image_height=100,
            ),
            retrieval=RetrievalResult(
                status="no_evidence",
                query="recycling information",
                results=[],
            ),
            generation=GenerationResult(
                status="skipped",
                reason="Answer generation was not requested.",
            ),
        )


def test_partial_success_requires_a_useful_result_and_issue():
    with pytest.raises(ValidationError):
        AnalyzeResponse(
            request_id="req-partial-empty",
            status="partial_success",
            warnings=["A stage did not complete."],
        )

    response = AnalyzeResponse(
        request_id="req-partial",
        status="partial_success",
        vision=VisionResult(
            status="no_detections",
            image_width=100,
            image_height=100,
        ),
        warnings=["Retrieval was not completed."],
    )
    assert response.status == "partial_success"


def test_failed_response_must_not_contain_useful_stage_results():
    with pytest.raises(ValidationError):
        AnalyzeResponse(
            request_id="req-failed-with-output",
            status="failed",
            vision=VisionResult(
                status="no_detections",
                image_width=100,
                image_height=100,
            ),
            errors=[
                ErrorInfo(
                    code="RETRIEVAL_FAILED",
                    stage="retrieval",
                    message="Retrieval failed.",
                )
            ],
        )


def test_success_cannot_contain_warnings():
    with pytest.raises(ValidationError):
        AnalyzeResponse(
            request_id="req-success-warning",
            status="success",
            vision=VisionResult(
                status="no_detections",
                image_width=100,
                image_height=100,
            ),
            retrieval=RetrievalResult(
                status="no_evidence",
                query="recycling information",
                results=[],
            ),
            generation=GenerationResult(
                status="insufficient_evidence",
                answer="I do not have enough evidence to answer reliably.",
                reason="No relevant evidence was retrieved.",
            ),
            warnings=["A non-critical issue occurred."],
        )


def test_generation_rejects_duplicate_citation_ids():
    with pytest.raises(ValidationError):
        GenerationResult(
            status="success",
            answer="See the retrieved guidance.",
            citation_chunk_ids=["chunk-1", "chunk-1"],
        )


def test_insufficient_evidence_requires_reason():
    with pytest.raises(ValidationError):
        GenerationResult(
            status="insufficient_evidence",
            answer="I do not have enough evidence to answer reliably.",
        )


def test_request_rejects_all_stages_disabled():
    from smart_waste_multimodal.contracts import RequestOptions

    with pytest.raises(ValidationError, match="At least one pipeline stage"):
        AnalyzeRequest(
            request_id="req-no-stages",
            conversation_id="test-conversation-test_contracts-430",
            turn_id="test-turn-test_contracts-430",
            image=ImageArtifact(
                artifact_id="image-no-stages",
                media_type="image/jpeg",
            ),
            options=RequestOptions(
                run_vision=False,
                run_retrieval=False,
                generate_answer=False,
            ),
        )


def test_request_rejects_question_retrieval_without_question():
    with pytest.raises(ValidationError, match="question is required"):
        AnalyzeRequest(
            request_id="req-no-question",
            conversation_id="test-conversation-test_contracts-446",
            turn_id="test-turn-test_contracts-446",
            image=ImageArtifact(
                artifact_id="image-no-question",
                media_type="image/jpeg",
            ),
        )


def test_request_allows_independent_retrieval_without_vision():
    from smart_waste_multimodal.contracts import RequestOptions

    request = AnalyzeRequest(
        request_id="req-independent-retrieval",
        conversation_id="test-conversation-test_contracts-458",
        turn_id="test-turn-test_contracts-458",
        image=ImageArtifact(
            artifact_id="image-independent",
            media_type="image/jpeg",
        ),
        question="How should plastic bottles be recycled?",
        options=RequestOptions(
            run_vision=False,
            run_retrieval=True,
            generate_answer=False,
            retrieval_query_source="question",
        ),
    )

    assert request.options.run_vision is False
    assert request.options.run_retrieval is True
    assert request.options.retrieval_query_source == "question"


def test_request_rejects_detection_based_retrieval_without_vision():
    from smart_waste_multimodal.contracts import RequestOptions

    with pytest.raises(
        ValidationError,
        match="Vision must be enabled",
    ):
        AnalyzeRequest(
            request_id="req-dependent-retrieval",
            conversation_id="test-conversation-test_contracts-485",
            turn_id="test-turn-test_contracts-485",
            image=ImageArtifact(
                artifact_id="image-dependent",
                media_type="image/jpeg",
            ),
            options=RequestOptions(
                run_vision=False,
                run_retrieval=True,
                generate_answer=False,
                retrieval_query_source="vision_detections",
            ),
        )


def test_request_allows_detection_based_retrieval_with_vision():
    from smart_waste_multimodal.contracts import RequestOptions

    request = AnalyzeRequest(
        request_id="req-vision-retrieval",
        task_type="detection",
        conversation_id="test-conversation-test_contracts-503",
        turn_id="test-turn-test_contracts-503",
        image=ImageArtifact(
            artifact_id="image-vision-retrieval",
            media_type="image/jpeg",
        ),
        options=RequestOptions(
            run_vision=True,
            run_retrieval=True,
            generate_answer=False,
            retrieval_query_source="vision_detections",
        ),
    )

    assert request.options.run_vision is True
    assert request.options.retrieval_query_source == "vision_detections"


@pytest.mark.parametrize(
    "coordinate",
    [float("inf"), float("-inf"), float("nan")],
    ids=["positive-infinity", "negative-infinity", "nan"],
)
def test_detection_rejects_non_finite_bbox_coordinates(coordinate):
    with pytest.raises(ValidationError):
        Detection(
            class_name="plastic",
            confidence=0.9,
            bbox_xyxy=(1, 2, coordinate, 20),
        )


def test_detection_rejects_non_finite_first_bbox_coordinate():
    with pytest.raises(ValidationError):
        Detection(
            class_name="plastic",
            confidence=0.9,
            bbox_xyxy=(float("nan"), 2, 10, 20),
        )


def test_final_evidence_context_contract_accepts_aligned_evidence():
    from datetime import UTC, datetime

    from smart_waste_multimodal.contracts.evidence import Evidence
    from smart_waste_multimodal.contracts.final_evidence_context import (
        FinalEvidenceContext,
    )

    item = Evidence.model_construct(
        evidence_id="ev-1",
        request_id="req-1",
        schema_version="1",
        source_type="RETRIEVAL_CHUNK",
        provenance=None,
        content={"text": "approved evidence"},
        created_at=datetime.now(UTC),
        metadata={},
        conversation_id="conversation-1",
        created_by_request_id="req-1",
        origin_turn_id="turn-1",
    )

    context = FinalEvidenceContext(
        request_id="req-1",
        conversation_id="conversation-1",
        evidence_ids=("ev-1",),
        evidence=(item,),
    )

    assert context.evidence_ids == ("ev-1",)
    assert context.evidence[0].evidence_id == "ev-1"


def test_final_evidence_context_rejects_empty_evidence():
    import pytest
    from pydantic import ValidationError

    from smart_waste_multimodal.contracts.final_evidence_context import (
        FinalEvidenceContext,
    )

    with pytest.raises(ValidationError):
        FinalEvidenceContext(
            request_id="req-1",
            evidence_ids=(),
            evidence=(),
        )


def test_final_evidence_context_rejects_misaligned_ids():
    from datetime import UTC, datetime

    import pytest
    from pydantic import ValidationError

    from smart_waste_multimodal.contracts.evidence import Evidence
    from smart_waste_multimodal.contracts.final_evidence_context import (
        FinalEvidenceContext,
    )

    item = Evidence.model_construct(
        evidence_id="ev-actual",
        request_id="req-1",
        schema_version="1",
        source_type="RETRIEVAL_CHUNK",
        provenance=None,
        content={"text": "approved evidence"},
        created_at=datetime.now(UTC),
        metadata={},
        conversation_id="conversation-1",
        created_by_request_id="req-1",
        origin_turn_id="turn-1",
    )

    with pytest.raises(ValidationError):
        FinalEvidenceContext(
            request_id="req-1",
            evidence_ids=("ev-requested",),
            evidence=(item,),
        )


def test_final_evidence_context_rejects_evidence_from_another_request():
    from datetime import UTC, datetime

    import pytest
    from pydantic import ValidationError

    from smart_waste_multimodal.contracts.evidence import Evidence
    from smart_waste_multimodal.contracts.final_evidence_context import (
        FinalEvidenceContext,
    )

    item = Evidence.model_construct(
        evidence_id="ev-1",
        request_id="different-request",
        schema_version="1",
        source_type="RETRIEVAL_CHUNK",
        provenance=None,
        content={"text": "evidence"},
        created_at=datetime.now(UTC),
        metadata={},
        conversation_id="conversation-1",
        created_by_request_id="different-request",
        origin_turn_id="turn-1",
    )

    with pytest.raises(ValidationError):
        FinalEvidenceContext(
            request_id="req-1",
            evidence_ids=("ev-1",),
            evidence=(item,),
        )


def test_error_info_accepts_final_evidence_context_stage():
    error = ErrorInfo(
        code="FINAL_EVIDENCE_CONTEXT_FAILED",
        stage="final_evidence_context",
        message="Final evidence context could not be built.",
    )

    assert error.stage == "final_evidence_context"


def test_error_info_rejects_unknown_stage():
    with pytest.raises(ValidationError):
        ErrorInfo(
            code="UNKNOWN_STAGE",
            stage="unknown_stage",
            message="This stage is not allowed.",
        )


def test_request_accepts_text_only_turn_without_image():
    request = AnalyzeRequest(
        request_id="req-text-only",
        conversation_id="conversation-text-only",
        turn_id="turn-text-only",
        task_type="knowledge_question",
        image=None,
        question="How should glass be recycled?",
    )

    assert request.image is None
    assert request.task_type == "knowledge_question"


def test_request_accepts_detection_task_without_question():
    request = AnalyzeRequest(
        request_id="req-detection-no-question",
        conversation_id="conversation-detection",
        turn_id="turn-detection",
        task_type="detection",
        image=None,
        question=None,
        options={
            "run_vision": False,
            "run_retrieval": False,
            "generate_answer": True,
        },
    )

    assert request.image is None
    assert request.question is None
    assert request.task_type == "detection"


def test_request_requires_conversation_and_turn_ids():
    with pytest.raises(ValidationError):
        AnalyzeRequest(
            request_id="req-missing-identifiers",
            question="How should glass be recycled?",
        )


def test_request_rejects_naive_turn_timestamp():
    from datetime import datetime

    with pytest.raises(
        ValidationError,
        match="turn_timestamp must include a timezone",
    ):
        AnalyzeRequest(
            request_id="req-naive-timestamp",
            conversation_id="conversation-naive",
            turn_id="turn-naive",
            turn_timestamp=datetime(2030, 1, 1, 12, 0, 0),
            question="How should glass be recycled?",
        )


def test_history_timestamp_is_compared_with_request_turn_timestamp():
    from datetime import UTC, datetime, timedelta

    from smart_waste_multimodal.contracts.conversation import ConversationTurn

    request_time = datetime(2030, 5, 10, 12, 0, 0, tzinfo=UTC)

    future_turn = ConversationTurn(
        turn_id="future-history-turn",
        role="user",
        content="A future turn",
        timestamp=request_time + timedelta(seconds=1),
    )

    with pytest.raises(
        ValidationError,
        match="relative to turn_timestamp",
    ):
        AnalyzeRequest(
            request_id="req-future-history",
            conversation_id="conversation-future",
            turn_id="turn-current",
            turn_timestamp=request_time,
            question="How should glass be recycled?",
            conversation_history=[future_turn],
        )


# OPTIONAL_IMAGE_RESPONSE_CONTRACT_TESTS


def test_response_allows_vision_to_be_skipped_when_no_image_was_provided():
    from smart_waste_multimodal.contracts import RequestOptions

    response = AnalyzeResponse(
        request_id="req-no-image",
        status="success",
        requested_options=RequestOptions(
            run_vision=True,
            run_retrieval=False,
            generate_answer=False,
        ),
        image_provided=False,
    )

    assert response.image_provided is False
    assert response.vision is None


def test_response_rejects_vision_output_when_no_image_was_provided():
    from smart_waste_multimodal.contracts import RequestOptions

    with pytest.raises(
        ValidationError,
        match="no image was provided",
    ):
        AnalyzeResponse(
            request_id="req-no-image-with-vision",
            status="success",
            requested_options=RequestOptions(
                run_vision=True,
                run_retrieval=False,
                generate_answer=False,
            ),
            image_provided=False,
            vision=VisionResult(
                status="no_detections",
                image_width=100,
                image_height=100,
            ),
        )


# P1_CONVERSATION_EVIDENCE_CONTRACT_TESTS


def test_conversation_contract_validates_owner_and_lifecycle():
    from datetime import UTC, datetime, timedelta

    import pytest
    from pydantic import ValidationError

    from smart_waste_multimodal.contracts.conversation import Conversation

    now = datetime.now(UTC)
    active = Conversation(
        conversation_id="conversation-p1",
        owner_id="owner-p1",
        created_at=now,
        updated_at=now,
    )
    assert active.status == "active"

    closed = Conversation(
        conversation_id="conversation-p1-closed",
        owner_id="owner-p1",
        status="closed",
        created_at=now,
        updated_at=now + timedelta(minutes=1),
        closed_at=now + timedelta(minutes=1),
    )
    assert closed.status == "closed"

    with pytest.raises(ValidationError):
        Conversation(
            conversation_id="conversation-invalid",
            owner_id="owner-p1",
            status="closed",
            created_at=now,
            updated_at=now,
        )


def test_conversation_turn_supports_optional_request_and_evidence_links():
    from datetime import UTC, datetime

    from smart_waste_multimodal.contracts.conversation import ConversationTurn

    turn = ConversationTurn(
        turn_id="turn-p1",
        role="user",
        content="What material is visible?",
        timestamp=datetime.now(UTC),
        conversation_id="conversation-p1",
        request_id="request-p1",
        evidence_ids=("evidence-1", "evidence-2"),
    )
    assert turn.conversation_id == "conversation-p1"
    assert turn.request_id == "request-p1"
    assert turn.evidence_ids == ("evidence-1", "evidence-2")


def test_conversation_turn_image_artifact_id_is_optional():
    from datetime import UTC, datetime

    from smart_waste_multimodal.contracts.conversation import ConversationTurn

    turn = ConversationTurn(
        turn_id="turn-image-optional",
        role="user",
        content="Text-only message",
        timestamp=datetime.now(UTC),
    )

    assert turn.image_artifact_id is None


def test_conversation_turn_accepts_image_artifact_id():
    from datetime import UTC, datetime

    from smart_waste_multimodal.contracts.conversation import ConversationTurn

    turn = ConversationTurn(
        turn_id="turn-image-present",
        role="user",
        content="What is in this image?",
        timestamp=datetime.now(UTC),
        image_artifact_id="image-artifact-001",
    )

    assert turn.image_artifact_id == "image-artifact-001"


@pytest.mark.parametrize(
    "image_artifact_id",
    ["", "   ", "a" * 201],
)
def test_conversation_turn_rejects_invalid_image_artifact_id(
    image_artifact_id,
):
    from datetime import UTC, datetime

    from smart_waste_multimodal.contracts.conversation import ConversationTurn

    with pytest.raises(ValidationError):
        ConversationTurn(
            turn_id="turn-invalid-image-id",
            role="user",
            content="Message",
            timestamp=datetime.now(UTC),
            image_artifact_id=image_artifact_id,
        )


def test_evidence_allows_request_id_to_differ_from_creator_request_id():
    from datetime import UTC, datetime

    from smart_waste_multimodal.contracts.evidence import (
        Evidence,
        EvidenceProvenance,
        EvidenceSourceType,
    )

    evidence = Evidence(
        evidence_id="evidence-p1",
        request_id="request-current",
        conversation_id="conversation-p1",
        created_by_request_id="request-original",
        origin_turn_id="turn-original",
        source_type=EvidenceSourceType.RETRIEVAL_CHUNK,
        provenance=EvidenceProvenance(source_id="source-p1"),
        content={"text": "reusable evidence"},
        created_at=datetime.now(UTC),
    )
    assert evidence.request_id == "request-current"
    assert evidence.created_by_request_id == "request-original"


# STAGE_1_CONVERSATION_EVIDENCE_FINALIZATION_TESTS


def test_closed_conversation_timestamp_must_not_follow_updated_at():
    from datetime import UTC, datetime, timedelta

    from pydantic import ValidationError

    from smart_waste_multimodal.contracts import Conversation

    now = datetime.now(UTC)

    with pytest.raises(ValidationError, match="closed_at must not follow updated_at"):
        Conversation(
            conversation_id="conversation-close-order",
            owner_id="owner-1",
            status="closed",
            created_at=now,
            updated_at=now + timedelta(minutes=1),
            closed_at=now + timedelta(minutes=2),
        )


def test_evidence_request_id_is_marked_as_temporary_compatibility_field():
    from smart_waste_multimodal.contracts import Evidence

    schema = Evidence.model_json_schema()
    request_id_schema = schema["properties"]["request_id"]

    assert request_id_schema["deprecated"] is True
    assert "Temporary compatibility field" in request_id_schema["description"]


def test_evidence_preserves_creator_and_origin_independently_of_request_id():
    from datetime import UTC, datetime

    from smart_waste_multimodal.contracts import Evidence
    from smart_waste_multimodal.contracts.evidence import (
        EvidenceProvenance,
        EvidenceSourceType,
    )

    evidence = Evidence(
        evidence_id="evidence-independent-provenance",
        request_id="request-current",
        conversation_id="conversation-shared",
        created_by_request_id="request-original",
        origin_turn_id="turn-original",
        source_type=EvidenceSourceType.CONVERSATION_TEXT,
        provenance=EvidenceProvenance(source_id="turn-original"),
        content={"text": "Reusable conversation evidence."},
        created_at=datetime.now(UTC),
    )

    assert evidence.request_id == "request-current"
    assert evidence.created_by_request_id == "request-original"
    assert evidence.origin_turn_id == "turn-original"
    assert evidence.conversation_id == "conversation-shared"
