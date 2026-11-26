"""Unit tests for the legacy hybrid RAG adapter."""

from unittest.mock import Mock

import pytest

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


def make_request(*, source="question", question="How do I recycle plastic?"):
    return AnalyzeRequest(
        request_id="rag-adapter-test-001",
        conversation_id="test-conversation-test_hybrid_rag_adapter-20",
        turn_id="test-turn-test_hybrid_rag_adapter-20",
        image=ImageArtifact(
            artifact_id="test-image-001",
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


def make_vision():
    return VisionResult(
        status="success",
        detections=[
            Detection(
                class_name="plastic",
                confidence=0.95,
                bbox_xyxy=(10, 10, 40, 40),
            ),
            Detection(
                class_name="plastic",
                confidence=0.80,
                bbox_xyxy=(45, 10, 70, 40),
            ),
            Detection(
                class_name="glass",
                confidence=0.91,
                bbox_xyxy=(10, 45, 40, 80),
            ),
        ],
        image_width=100,
        image_height=100,
        model_version="test-model-v1",
    )


def make_raw_result():
    return {
        "chunk_id": "chunk-001",
        "source": "recycling-guide",
        "text": "Clean plastic containers before recycling.",
        "metadata": {"page": 3},
        "score": 0.025,
        "rank": 1,
    }


def test_question_query_maps_results_to_contract():
    retriever = Mock(return_value=None)
    retriever.search.return_value = [make_raw_result()]

    adapter = HybridRAGAdapter(
        retriever=retriever,
        top_k=3,
        index_version="v4_expanded",
    )

    result = adapter(make_request(), vision=None)

    assert result.status == "success"
    assert result.query == "How do I recycle plastic?"
    assert result.index_version == "v4_expanded"
    assert len(result.results) == 1
    assert result.results[0].chunk_id == "chunk-001"
    assert result.results[0].source == "recycling-guide"
    assert result.results[0].score == 0.025
    assert result.results[0].rank == 1

    retriever.search.assert_called_once_with(
        "How do I recycle plastic?",
        top_k=3,
    )


def test_detection_query_uses_unique_class_names():
    retriever = Mock()
    retriever.search.return_value = [make_raw_result()]

    adapter = HybridRAGAdapter(retriever=retriever)

    result = adapter(
        make_request(source="vision_detections", question=None),
        vision=make_vision(),
    )

    assert result.query == "plastic glass"
    retriever.search.assert_called_once_with(
        "plastic glass",
        top_k=5,
    )


def test_empty_search_results_return_no_evidence():
    retriever = Mock()
    retriever.search.return_value = []

    result = HybridRAGAdapter(
        retriever=retriever,
        index_version="v4_expanded",
    )(make_request(), vision=None)

    assert result.status == "no_evidence"
    assert result.results == []
    assert result.index_version == "v4_expanded"


def test_retriever_failure_propagates_to_orchestrator():
    retriever = Mock()
    retriever.search.side_effect = RuntimeError("retrieval unavailable")
    adapter = HybridRAGAdapter(retriever=retriever)

    with pytest.raises(RuntimeError, match="retrieval unavailable"):
        adapter(make_request(), vision=None)


def test_detection_query_requires_vision_results():
    adapter = HybridRAGAdapter(retriever=Mock())

    with pytest.raises(ValueError, match="Vision results are required"):
        adapter(
            make_request(source="vision_detections", question=None),
            vision=None,
        )


def test_detection_query_rejects_empty_detections():
    adapter = HybridRAGAdapter(retriever=Mock())
    vision = VisionResult(
        status="no_detections",
        detections=[],
        image_width=100,
        image_height=100,
    )

    with pytest.raises(ValueError, match="at least one detection"):
        adapter(
            make_request(source="vision_detections", question=None),
            vision=vision,
        )


def test_top_k_must_be_positive():
    with pytest.raises(ValueError, match="top_k must be positive"):
        HybridRAGAdapter(retriever=Mock(), top_k=0)


def test_invalid_retriever_output_is_not_silently_accepted():
    retriever = Mock()
    retriever.search.return_value = [
        {
            "chunk_id": "chunk-001",
            "source": "recycling-guide",
            "text": "Some evidence.",
            "score": float("nan"),
            "rank": 1,
        }
    ]

    adapter = HybridRAGAdapter(retriever=retriever)

    with pytest.raises(ValueError):
        adapter(make_request(), vision=None)

@pytest.mark.parametrize("raw_results", [None, {}, "invalid"])
def test_retriever_output_must_be_a_list(raw_results):
    retriever = Mock()
    retriever.search.return_value = raw_results
    adapter = HybridRAGAdapter(retriever=retriever)

    with pytest.raises(TypeError, match="must be a list"):
        adapter(make_request(), vision=None)


@pytest.mark.parametrize("record", [None, "invalid", 42])
def test_retriever_records_must_be_dictionaries(record):
    retriever = Mock()
    retriever.search.return_value = [record]
    adapter = HybridRAGAdapter(retriever=retriever)

    with pytest.raises(TypeError, match="must be a dictionary"):
        adapter(make_request(), vision=None)


def test_retriever_record_must_contain_all_required_fields():
    retriever = Mock()
    record = make_raw_result()
    record.pop("source")
    retriever.search.return_value = [record]
    adapter = HybridRAGAdapter(retriever=retriever)

    with pytest.raises(ValueError, match="missing required fields: source"):
        adapter(make_request(), vision=None)


def test_retrieval_contract_rejects_duplicate_chunk_ids():
    retriever = Mock()
    first = make_raw_result()
    second = make_raw_result()
    second["rank"] = 2
    retriever.search.return_value = [first, second]
    adapter = HybridRAGAdapter(retriever=retriever, top_k=2)

    with pytest.raises(ValueError, match="duplicate chunk_id"):
        adapter(make_request(), vision=None)


@pytest.mark.parametrize("ranks", [[1, 1], [1, 3], [2, 1]])
def test_retrieval_contract_rejects_invalid_rank_sequences(ranks):
    retriever = Mock()
    records = []
    for index, rank in enumerate(ranks):
        record = make_raw_result()
        record["chunk_id"] = f"chunk-{index + 1:03d}"
        record["rank"] = rank
        records.append(record)
    retriever.search.return_value = records
    adapter = HybridRAGAdapter(retriever=retriever, top_k=len(records))

    with pytest.raises(ValueError, match="ranks must be unique, ordered, and consecutive"):
        adapter(make_request(), vision=None)


def test_retriever_cannot_return_more_than_top_k():
    retriever = Mock()
    records = []
    for index in range(2):
        record = make_raw_result()
        record["chunk_id"] = f"chunk-{index + 1:03d}"
        record["rank"] = index + 1
        records.append(record)
    retriever.search.return_value = records
    adapter = HybridRAGAdapter(retriever=retriever, top_k=1)

    with pytest.raises(ValueError, match="exceeding top_k=1"):
        adapter(make_request(), vision=None)
