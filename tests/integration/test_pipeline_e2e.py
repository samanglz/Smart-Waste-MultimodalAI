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
from smart_waste_multimodal.orchestration.orchestrator import (
    PipelineOrchestrator,
)


def make_e2e_request() -> AnalyzeRequest:
    return AnalyzeRequest(
        request_id="e2e-mock-request-001",
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
    )

    response = orchestrator.run(make_e2e_request())

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
    assert response.retrieval.results[0].chunk_id == (
        "mock-recycling-chunk-001"
    )

    assert response.generation is not None
    assert response.generation.status == "success"
    assert "plastic" in response.generation.answer.lower()
    assert response.generation.citation_chunk_ids == [
        "mock-recycling-chunk-001"
    ]

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
    )

    response = orchestrator.run(request)

    assert response.status == "success"
    assert response.vision is not None
    assert response.retrieval is not None
    assert "plastic" in response.retrieval.query.lower()
    assert response.generation is not None
    assert response.generation.citation_chunk_ids == [
        response.retrieval.results[0].chunk_id
    ]
