"""Unit tests for inter-module contracts."""

import pytest
from pydantic import ValidationError

from smart_waste_multimodal.contracts import (
    AnalyzeRequest,
    AnalyzeResponse,
    Detection,
    ErrorInfo,
    EvidenceItem,
    GenerationResult,
    ImageArtifact,
    RetrievalResult,
    VisionResult,
)


def test_request_accepts_image_artifact():
    request = AnalyzeRequest(
        request_id="req-001",
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
            image={"artifact_id": "img", "media_type": "image/png"},
            unexpected="not allowed",
        )


def test_detection_validates_confidence_and_box():
    with pytest.raises(ValidationError):
        Detection(
            class_name="plastic", confidence=1.2,
            bbox_xyxy=(1, 2, 10, 20),
        )
    with pytest.raises(ValidationError):
        Detection(
            class_name="plastic", confidence=0.9,
            bbox_xyxy=(10, 2, 1, 20),
        )


def test_vision_status_matches_detection_list():
    with pytest.raises(ValidationError):
        VisionResult(
            status="no_detections",
            detections=[
                Detection(
                    class_name="paper", confidence=0.8,
                    bbox_xyxy=(1, 2, 10, 20),
                )
            ],
            image_width=100, image_height=100,
        )
    result = VisionResult(
        status="no_detections", image_width=100, image_height=100
    )
    assert result.detections == []


def test_vision_rejects_box_outside_image():
    with pytest.raises(ValidationError):
        VisionResult(
            status="success",
            detections=[
                Detection(
                    class_name="metal", confidence=0.9,
                    bbox_xyxy=(1, 2, 110, 20),
                )
            ],
            image_width=100, image_height=100,
        )


def test_retrieval_distinguishes_no_evidence():
    result = RetrievalResult(
        status="no_evidence", query="recycling metal", results=[]
    )
    assert result.results == []
    with pytest.raises(ValidationError):
        RetrievalResult(
            status="success", query="recycling metal", results=[]
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
                        chunk_id="actual-chunk", source="guide.pdf",
                        text="Paper recycling guidance.", score=0.8, rank=1,
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
                code="VISION_MODEL_UNAVAILABLE", stage="vision",
                message="Vision stage could not run.", retryable=True,
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
            image=ImageArtifact(
                artifact_id="image-no-question",
                media_type="image/jpeg",
            ),
        )


def test_request_allows_independent_retrieval_without_vision():
    from smart_waste_multimodal.contracts import RequestOptions

    request = AnalyzeRequest(
        request_id="req-independent-retrieval",
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
