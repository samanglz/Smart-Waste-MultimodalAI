"""End-to-end failure-path tests using deterministic mock adapters."""

from unittest.mock import Mock

from smart_waste_multimodal.adapters.mock_adapters import (
    MockGenerationAdapter,
    MockRetrievalAdapter,
    MockVisionAdapter,
)
from smart_waste_multimodal.contracts import (
    RequestOptions,
    RetrievalResult,
)
from smart_waste_multimodal.orchestration.orchestrator import (
    PipelineOrchestrator,
)
from tests.integration.test_pipeline_e2e import make_e2e_request


def test_vision_failure_preserves_independent_retrieval_output():
    vision = Mock(side_effect=RuntimeError("mock vision unavailable"))
    retrieval = MockRetrievalAdapter()
    generation = MockGenerationAdapter()

    orchestrator = PipelineOrchestrator(
        vision_stage=vision,
        retrieval_stage=retrieval,
        generation_stage=generation,
    )

    response = orchestrator.run(make_e2e_request())

    assert response.status == "partial_success"
    assert response.vision is None
    assert response.retrieval is not None
    assert response.retrieval.status == "success"
    assert response.generation is not None
    assert response.generation.status == "success"
    assert any(error.stage == "vision" for error in response.errors)



def test_invalid_vision_contract_preserves_independent_pipeline_stages():
    vision = Mock(return_value={"invalid": "vision payload"})
    retrieval = MockRetrievalAdapter()
    generation = MockGenerationAdapter()

    orchestrator = PipelineOrchestrator(
        vision_stage=vision,
        retrieval_stage=retrieval,
        generation_stage=generation,
    )

    response = orchestrator.run(make_e2e_request())

    assert response.status == "partial_success"
    assert response.vision is None
    assert response.retrieval is not None
    assert response.generation is not None
    assert any(error.stage == "vision" for error in response.errors)
    assert response.errors[0].message == (
        "The vision stage could not complete."
    )



def test_no_retrieval_evidence_produces_insufficient_evidence_answer():
    request = make_e2e_request()
    retrieval = Mock(
        return_value=RetrievalResult(
            status="no_evidence",
            query=request.question or "recycling guidance",
            results=[],
            index_version="mock-empty-index-v1",
        )
    )

    orchestrator = PipelineOrchestrator(
        vision_stage=MockVisionAdapter(),
        retrieval_stage=retrieval,
        generation_stage=MockGenerationAdapter(),
    )

    response = orchestrator.run(request)

    assert response.status == "success"
    assert response.errors == []
    assert response.warnings == []
    assert response.vision is not None
    assert response.retrieval is not None
    assert response.retrieval.status == "no_evidence"
    assert response.retrieval.results == []
    assert response.generation is not None
    assert response.generation.status == "insufficient_evidence"
    assert response.generation.citation_chunk_ids == []


def test_generation_failure_preserves_vision_and_retrieval():
    generation = Mock(
        side_effect=RuntimeError("mock generation unavailable")
    )

    orchestrator = PipelineOrchestrator(
        vision_stage=MockVisionAdapter(),
        retrieval_stage=MockRetrievalAdapter(),
        generation_stage=generation,
    )

    response = orchestrator.run(make_e2e_request())

    assert response.status == "partial_success"
    assert response.vision is not None
    assert response.vision.status == "success"
    assert response.retrieval is not None
    assert response.retrieval.status == "success"
    assert response.generation is None
    assert any(error.stage == "generation" for error in response.errors)


def test_vision_failure_skips_detection_dependent_retrieval():
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
    vision = Mock(side_effect=RuntimeError("mock vision unavailable"))
    retrieval = Mock(wraps=MockRetrievalAdapter())

    orchestrator = PipelineOrchestrator(
        vision_stage=vision,
        retrieval_stage=retrieval,
        generation_stage=MockGenerationAdapter(),
    )

    response = orchestrator.run(request)

    assert response.status == "partial_success"
    assert response.vision is None
    assert response.retrieval is None
    assert response.generation is not None
    assert response.generation.status == "insufficient_evidence"
    assert any(error.stage == "vision" for error in response.errors)
    assert any("Retrieval was not run" in warning for warning in response.warnings)
    retrieval.assert_not_called()
