from unittest.mock import Mock

from smart_waste_multimodal.contracts import (
    AnalyzeRequest,
    Detection,
    ImageArtifact,
    RequestOptions,
    VisionResult,
)
from smart_waste_multimodal.retrieval.hybrid_rag_adapter import (
    HybridRAGAdapter,
)


def make_request(source="question", question="How do I recycle plastic?"):
    return AnalyzeRequest(
        request_id="rewrite-adapter-test",
        conversation_id="test-conversation-test_hybrid_rag_query_rewriting-16",
        turn_id="test-turn-test_hybrid_rag_query_rewriting-16",
        image=ImageArtifact(
            artifact_id="rewrite-image",
            media_type="image/jpeg",
            width=100,
            height=100,
        ),
        question=question,
        options=RequestOptions(
            run_vision=True,
            run_retrieval=True,
            generate_answer=False,
            retrieval_query_source=source,
        ),
    )


def make_raw_result():
    return {
        "chunk_id": "chunk-1",
        "source": "guide.txt",
        "text": "Recycling guidance.",
        "score": 0.2,
        "rank": 1,
    }


def test_adapter_retrieves_with_rewritten_query():
    retriever = Mock()
    retriever.search.return_value = [make_raw_result()]
    rewrite_service = Mock()
    rewrite_service.rewrite_for_retrieval.return_value = Mock(
        action="rewritten",
        original_query="How do I recycle plastic?",
        retrieval_query="plastic container recycling guidance",
    )

    adapter = HybridRAGAdapter(
        retriever=retriever,
        rewrite_service=rewrite_service,
    )
    result = adapter(make_request(), vision=None)

    rewrite_service.rewrite_for_retrieval.assert_called_once_with(
        "How do I recycle plastic?",
        context_assessment=None,
    )
    retriever.search.assert_called_once_with(
        "plastic container recycling guidance",
        top_k=5,
    )
    assert result.query == "plastic container recycling guidance"


def test_detection_query_bypasses_rewrite_service():
    retriever = Mock()
    retriever.search.return_value = [make_raw_result()]
    rewrite_service = Mock()

    vision = VisionResult(
        status="success",
        detections=[
            Detection(
                class_name="plastic",
                confidence=0.9,
                bbox_xyxy=(1, 1, 20, 20),
            )
        ],
        image_width=100,
        image_height=100,
    )

    adapter = HybridRAGAdapter(
        retriever=retriever,
        rewrite_service=rewrite_service,
    )
    result = adapter(
        make_request(source="vision_detections", question=None),
        vision=vision,
    )

    rewrite_service.rewrite_for_retrieval.assert_not_called()
    retriever.search.assert_called_once_with("plastic", top_k=5)
    assert result.query == "plastic"

def test_adapter_rejects_legacy_human_review_action():
    retriever = Mock()
    rewrite_service = Mock()

    rewrite_service.rewrite_for_retrieval.return_value = Mock(
        action="human_review",
        original_query="What about the other one?",
        retrieval_query="",
        reason="context_ambiguous",
    )

    adapter = HybridRAGAdapter(
        retriever=retriever,
        rewrite_service=rewrite_service,
    )

    original_query = "What about the other one?"

    import pytest

    with pytest.raises(ValueError, match="Invalid query rewrite action"):
        adapter(
            make_request(question=original_query),
            vision=None,
        )

    rewrite_service.rewrite_for_retrieval.assert_called_once_with(
        original_query,
        context_assessment=None,
    )
    retriever.search.assert_not_called()
