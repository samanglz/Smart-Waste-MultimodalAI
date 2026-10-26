"""Mock-based tests for pipeline orchestration."""

import inspect
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
from smart_waste_multimodal.orchestration.orchestrator import (
    PipelineOrchestrator,
)


def make_request(
    *,
    options=None,
    question="How should I recycle this item?",
):
    return AnalyzeRequest(
        request_id="mock-request-001",
        image=ImageArtifact(
            artifact_id="image-001",
            media_type="image/jpeg",
            width=100,
            height=100,
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
        citation_chunk_ids=["chunk-001"],
        model_version="mock-generator-v1",
    )


def make_orchestrator(
    vision=None,
    retrieval=None,
    generation=None,
):
    return PipelineOrchestrator(
        vision_stage=vision or Mock(return_value=make_vision()),
        retrieval_stage=retrieval or Mock(return_value=make_retrieval()),
        generation_stage=generation or Mock(return_value=make_generation()),
    )


def test_all_stages_succeed():
    vision = Mock(return_value=make_vision())
    retrieval = Mock(return_value=make_retrieval())
    generation = Mock(return_value=make_generation())

    response = make_orchestrator(
        vision, retrieval, generation
    ).run(make_request())

    assert response.status == "success"
    assert response.vision is not None
    assert response.retrieval is not None
    assert response.generation is not None
    assert response.errors == []
    assert response.warnings == []
    vision.assert_called_once()
    retrieval.assert_called_once()
    generation.assert_called_once()


def test_independent_retrieval_runs_even_if_vision_fails():
    vision = Mock(side_effect=RuntimeError("mock vision failure"))
    retrieval = Mock(return_value=make_retrieval())
    generation = Mock(return_value=make_generation())

    response = make_orchestrator(
        vision, retrieval, generation
    ).run(make_request())

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
    ).run(request)

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
    ).run(make_request(options=options))

    assert response.status == "partial_success"
    assert response.vision is None
    assert response.retrieval is None
    assert response.generation is not None
    assert response.errors[0].stage == "vision"
    assert response.warnings
    retrieval.assert_not_called()
    generation.assert_called_once()


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
    ).run(make_request(options=options))

    assert response.status == "partial_success"
    assert response.vision is not None
    assert response.vision.status == "no_detections"
    assert response.retrieval is None
    assert response.warnings
    retrieval.assert_not_called()


def test_disabled_vision_is_never_called():
    options = RequestOptions(run_vision=False)
    vision = Mock(return_value=make_vision())

    response = make_orchestrator(vision=vision).run(
        make_request(options=options)
    )

    assert response.status == "success"
    assert response.vision is None
    vision.assert_not_called()


def test_generation_failure_preserves_upstream_results():
    generation = Mock(side_effect=RuntimeError("mock generation failure"))

    response = make_orchestrator(generation=generation).run(
        make_request()
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
    ).run(make_request(options=options))

    assert response.status == "failed"
    assert response.vision is None
    assert response.retrieval is None
    assert response.generation is None
    assert {error.stage for error in response.errors} == {
        "vision", "retrieval", "generation"
    }


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
    ).run(make_request())

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

    response = make_orchestrator(
        vision, retrieval, generation
    ).run(make_request(options=options))

    assert response.status == "success"
    assert response.vision is None
    assert response.retrieval is None
    assert response.warnings == []
    vision.assert_not_called()
    retrieval.assert_not_called()
    generation.assert_called_once()


def test_adapter_protocol_signatures_are_stable():
    assert list(inspect.signature(VisionStage.__call__).parameters) == [
        "self", "request"
    ]
    assert list(inspect.signature(RetrievalStage.__call__).parameters) == [
        "self", "request", "vision"
    ]
    assert list(inspect.signature(GenerationStage.__call__).parameters) == [
        "self", "request", "vision", "retrieval"
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
        make_request(
            options=RequestOptions(
                run_vision=True,
                run_retrieval=True,
                generate_answer=True,
                retrieval_query_source="question",
            )
        )
    )

    assert response.status == "partial_success"
    assert response.vision is None
    assert response.retrieval is not None
    assert response.generation is not None
    assert any(error.stage == "vision" for error in response.errors)
    assert all(error.message == "The vision stage could not complete."
               for error in response.errors if error.stage == "vision")
    retrieval.assert_called_once()
    generation.assert_called_once()


def test_wrong_retrieval_result_type_is_handled_as_stage_failure():
    retrieval = Mock(return_value={"unexpected": "dictionary"})

    response = make_orchestrator(
        retrieval=retrieval,
    ).run(
        make_request(
            options=RequestOptions(
                run_vision=False,
                run_retrieval=True,
                generate_answer=False,
                retrieval_query_source="question",
            )
        )
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
    ).run(make_request())

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
        make_request(
            options=RequestOptions(
                run_vision=True,
                run_retrieval=False,
                generate_answer=False,
            )
        )
    )

    assert response.status == "failed"
    assert response.vision is None
    assert len(response.errors) == 1
    assert response.errors[0].stage == "vision"
    assert response.errors[0].code == "vision_stage_failed"
    assert response.errors[0].message == (
        "The vision stage could not complete."
    )
    vision.assert_called_once()
