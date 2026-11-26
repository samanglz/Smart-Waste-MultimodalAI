"""Opt-in integration test for the real legacy RAG adapter."""

import os
from pathlib import Path

import pytest

if os.getenv("RUN_REAL_RAG_TESTS") != "1":
    pytest.skip(
        "Set RUN_REAL_RAG_TESTS=1 to run the real RAG integration test.",
        allow_module_level=True,
    )

from smart_waste_multimodal.adapters.mock_adapters import (
    MockGenerationAdapter,
    MockVisionAdapter,
)
from smart_waste_multimodal.contracts import (
    AnalyzeRequest,
    ImageArtifact,
    RequestOptions,
)
from smart_waste_multimodal.evidence import (
    InMemoryEvidenceRegistryFactory,
)
from smart_waste_multimodal.orchestration.orchestrator import (
    PipelineOrchestrator,
)
from smart_waste_multimodal.retrieval.bootstrap import build_rag_adapter


def test_real_rag_retrieval_through_orchestrator():
    rag_root = Path(
        os.environ.get(
            "WASTE_DETECTION_RAG_ROOT",
            "/content/drive/MyDrive/waste-detection-rag",
        )
    ).expanduser().resolve()

    index_dir = Path(
        os.environ.get(
            "WASTE_DETECTION_RAG_INDEX_DIR",
            str(
                rag_root
                / "data"
                / "processed"
                / "retrieval_index_v4_expanded"
            ),
        )
    ).expanduser().resolve()

    adapter = build_rag_adapter(
        rag_root=rag_root,
        index_dir=index_dir,
        model_name=os.getenv(
            "RAG_EMBEDDING_MODEL",
            "intfloat/multilingual-e5-small",
        ),
        top_k=5,
        index_version="v4_expanded",
        rewrite_mode="natural_only",
    )

    request = AnalyzeRequest(
        request_id="real-rag-integration-test",
        conversation_id="test-conversation-test_real_rag_adapter-64",
        turn_id="test-turn-test_real_rag_adapter-64",
        image=ImageArtifact(
            artifact_id="integration-test-image",
            media_type="image/jpeg",
            width=640,
            height=480,
        ),
        question="How should glass waste be sorted and recycled?",
        locale="en",
        options=RequestOptions(
            run_vision=False,
            run_retrieval=True,
            generate_answer=False,
            retrieval_query_source="question",
        ),
    )

    orchestrator = PipelineOrchestrator(
        vision_stage=MockVisionAdapter(),
        retrieval_stage=adapter,
        generation_stage=MockGenerationAdapter(),

        evidence_registry_factory=InMemoryEvidenceRegistryFactory(),)

    response = orchestrator.run(request)

    assert response.status == "success"
    assert response.retrieval is not None
    assert response.retrieval.status == "success"
    assert response.retrieval.index_version == "v4_expanded"
    assert len(response.retrieval.results) == 5
    assert response.errors == []
    assert response.warnings == []
