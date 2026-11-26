"""Unit tests for Gemini generation from authorized evidence."""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from smart_waste_multimodal.contracts import (
    AnalyzeRequest,
    Detection,
    EvidenceItem,
    FinalEvidenceContext,
    ImageArtifact,
    RequestOptions,
    RetrievalResult,
)
from smart_waste_multimodal.contracts.evidence import (
    Evidence,
    EvidenceProvenance,
    EvidenceSourceType,
)
from smart_waste_multimodal.generation.gemini_adapter import (
    GeminiGenerationAdapter,
)

REQUEST_ID = "generation-test-001"


def make_request(*, question="How should I recycle plastic?", locale="en"):
    return AnalyzeRequest(
        request_id=REQUEST_ID,
        conversation_id="test-conversation-test_generation_adapter-31",
        turn_id="test-turn-test_generation_adapter-31",
        image=ImageArtifact(
            artifact_id="generation-image-001",
            media_type="image/jpeg",
            width=100,
            height=100,
        ),
        question=question,
        locale=locale,
        options=RequestOptions(
            run_vision=False,
            run_retrieval=True,
            generate_answer=True,
            retrieval_query_source="question",
        ),
    )


def make_retrieval():
    return RetrievalResult(
        status="success",
        query="How should I recycle plastic?",
        index_version="v4_expanded",
        results=[
            EvidenceItem(
                chunk_id="chunk-plastic-001",
                source="plastic-guide.pdf",
                text="Clean plastic containers before recycling.",
                score=0.82,
                rank=1,
            ),
            EvidenceItem(
                chunk_id="chunk-plastic-002",
                source="plastic-guide.pdf",
                text="Follow the recycling rules of your local facility.",
                score=0.76,
                rank=2,
            ),
            EvidenceItem(
                chunk_id="chunk-local-001",
                source="local-recycling-guide.pdf",
                text="Some facilities require caps to be removed.",
                score=0.71,
                rank=3,
            ),
        ],
    )


def make_context(*, retrieval=None, detections=()):
    records = []
    now = datetime.now(UTC)

    if retrieval is not None:
        for item in retrieval.results:
            records.append(
                Evidence(
                    evidence_id=f"retrieval:{item.chunk_id}",
                    request_id=REQUEST_ID,
                    source_type=EvidenceSourceType.RETRIEVAL_CHUNK,
                    provenance=EvidenceProvenance(
                        source_id=item.chunk_id,
                        source_version=retrieval.index_version,
                    ),
                    content=item.model_dump(mode="json"),
                    created_at=now,
                    conversation_id="conversation-generation-001",
                    created_by_request_id=REQUEST_ID,
                    origin_turn_id="turn-generation-001",
                )
            )

    for index, detection in enumerate(detections):
        records.append(
            Evidence(
                evidence_id=f"vision:detection:{index}",
                request_id=REQUEST_ID,
                source_type=EvidenceSourceType.VISION_DETECTION,
                provenance=EvidenceProvenance(
                    source_id="generation-image-001",
                    source_locator={
                        "artifact_id": "generation-image-001",
                        "detection_index": str(index),
                    },
                ),
                content=detection.model_dump(mode="json"),
                created_at=now,
                conversation_id="conversation-generation-001",
                created_by_request_id=REQUEST_ID,
                origin_turn_id="turn-generation-001",
            )
        )

    if not records:
        records.append(
            Evidence(
                evidence_id="vision:snapshot-only",
                request_id=REQUEST_ID,
                source_type=EvidenceSourceType.VISION_SNAPSHOT,
                provenance=EvidenceProvenance(
                    source_id="generation-image-001",
                ),
                content={"status": "success"},
                created_at=now,
                conversation_id="conversation-generation-001",
                created_by_request_id=REQUEST_ID,
                origin_turn_id="turn-generation-001",
            )
        )

    return FinalEvidenceContext(
        request_id=REQUEST_ID,
        conversation_id="conversation-generation-001",
        evidence_ids=tuple(item.evidence_id for item in records),
        evidence=tuple(records),
    )


def make_client(answer):
    client = Mock()
    client.models.generate_content.return_value = SimpleNamespace(text=answer)
    return client


def test_context_builder_uses_only_retrieval_chunk_records():
    context = make_context(
        retrieval=make_retrieval(),
        detections=[
            Detection(
                class_name="plastic",
                confidence=0.95,
                bbox_xyxy=(10, 10, 90, 90),
            )
        ],
    )

    text, citation_map = GeminiGenerationAdapter._build_context(context)

    assert "Clean plastic containers before recycling." in text
    assert "plastic" not in citation_map.values()
    assert citation_map == {
        "C1": "chunk-plastic-001",
        "C2": "chunk-plastic-002",
        "C3": "chunk-local-001",
    }


def test_no_retrieval_evidence_does_not_call_gemini():
    client = make_client("This should never be returned.")
    adapter = GeminiGenerationAdapter(client=client, model="test-model")

    result = adapter(
        make_request(),
        make_context(detections=[
            Detection(
                class_name="plastic",
                confidence=0.95,
                bbox_xyxy=(10, 10, 90, 90),
            )
        ]),
    )

    assert result.status == "insufficient_evidence"
    assert result.answer
    assert result.citation_chunk_ids == []
    assert result.reason
    client.models.generate_content.assert_not_called()


def test_actual_model_version_is_recorded():
    client = make_client("Clean the container [C1].")
    client.models.generate_content.return_value = SimpleNamespace(
        text="Clean the container [C1].",
        model_version="gemini-3.5-flash-lite",
    )
    adapter = GeminiGenerationAdapter(
        client=client,
        model="gemini-flash-lite-latest",
    )

    result = adapter(make_request(), make_context(retrieval=make_retrieval()))

    assert result.status == "success"
    assert result.model_version == "gemini-3.5-flash-lite"


def test_requested_model_is_fallback_when_actual_version_is_missing():
    client = make_client("Clean the container [C1].")
    adapter = GeminiGenerationAdapter(client=client, model="test-model")

    result = adapter(make_request(), make_context(retrieval=make_retrieval()))

    assert result.status == "success"
    assert result.model_version == "test-model"


def test_valid_citations_map_to_actual_chunk_ids():
    client = make_client(
        "Clean the container before recycling [C1]. "
        "Check local facility rules [C2]."
    )
    adapter = GeminiGenerationAdapter(client=client, model="test-model")

    result = adapter(make_request(), make_context(retrieval=make_retrieval()))

    assert result.status == "success"
    assert result.citation_chunk_ids == [
        "chunk-plastic-001",
        "chunk-plastic-002",
    ]
    prompt = client.models.generate_content.call_args.kwargs["contents"]
    assert "[C1]" in prompt
    assert "[C2]" in prompt


def test_citation_ids_are_deduplicated_in_first_seen_order():
    client = make_client(
        "Clean the container [C1]. The same guidance applies [C1]. "
        "Check local rules [C3]."
    )
    adapter = GeminiGenerationAdapter(client=client, model="test-model")

    result = adapter(make_request(), make_context(retrieval=make_retrieval()))

    assert result.citation_chunk_ids == [
        "chunk-plastic-001",
        "chunk-local-001",
    ]


def test_unknown_citation_is_rejected():
    client = make_client("Recycle it according to this guide [C99].")
    adapter = GeminiGenerationAdapter(client=client, model="test-model")

    with pytest.raises(ValueError, match="unknown citation"):
        adapter(make_request(), make_context(retrieval=make_retrieval()))


def test_answer_without_citations_does_not_invent_citations():
    client = make_client("Clean plastic containers before recycling.")
    adapter = GeminiGenerationAdapter(client=client, model="test-model")

    result = adapter(make_request(), make_context(retrieval=make_retrieval()))

    assert result.status == "success"
    assert result.citation_chunk_ids == []


def test_empty_model_response_is_rejected():
    client = make_client("   ")
    adapter = GeminiGenerationAdapter(client=client, model="test-model")

    with pytest.raises(RuntimeError, match="empty response"):
        adapter(make_request(), make_context(retrieval=make_retrieval()))


def test_model_call_failure_propagates_to_orchestrator():
    client = Mock()
    client.models.generate_content.side_effect = RuntimeError(
        "provider unavailable"
    )
    adapter = GeminiGenerationAdapter(client=client, model="test-model")

    with pytest.raises(RuntimeError, match="provider unavailable"):
        adapter(make_request(), make_context(retrieval=make_retrieval()))


def test_prompt_contains_question_evidence_and_grounding_rules():
    client = make_client("Clean the container [C1].")
    adapter = GeminiGenerationAdapter(client=client, model="test-model")

    adapter(
        make_request(question="How should I recycle plastic?"),
        make_context(retrieval=make_retrieval()),
    )

    prompt = client.models.generate_content.call_args.kwargs["contents"]
    assert "How should I recycle plastic?" in prompt
    assert "Clean plastic containers before recycling." in prompt
    assert "Follow the recycling rules of your local facility." in prompt
    assert "Do not invent facts" in prompt
    assert "cite" in prompt.lower()


def test_disabled_generation_returns_skipped_without_calling_gemini():
    client = make_client("This should never be returned.")
    adapter = GeminiGenerationAdapter(client=client, model="test-model")

    request = make_request()
    request.options.generate_answer = False

    result = adapter(request, make_context(retrieval=make_retrieval()))

    assert result.status == "skipped"
    assert result.answer is None
    assert result.reason
    client.models.generate_content.assert_not_called()


def test_prompt_includes_visual_observations_from_authorized_evidence():
    client = make_client("Cardboard boxes can be flattened [C1].")
    adapter = GeminiGenerationAdapter(client=client, model="test-model")

    detection = Detection(
        class_name="cardboard",
        confidence=0.872,
        bbox_xyxy=(10, 10, 90, 90),
    )
    adapter(
        make_request(question="How should I recycle this item?"),
        make_context(
            retrieval=make_retrieval(),
            detections=[detection],
        ),
    )

    prompt = client.models.generate_content.call_args.kwargs["contents"]
    assert "- cardboard (confidence: 0.872)" in prompt
    assert "Treat detections as model predictions" in prompt
    assert "How should I recycle this item?" in prompt


def test_vision_summary_separates_multiple_authorized_detections():
    detections = [
        Detection(
            class_name="cardboard",
            confidence=0.872,
            bbox_xyxy=(0, 0, 40, 40),
        ),
        Detection(
            class_name="plastic",
            confidence=0.931,
            bbox_xyxy=(40, 40, 90, 90),
        ),
    ]

    summary = GeminiGenerationAdapter._build_vision_summary(
        make_context(detections=detections)
    )

    assert summary.splitlines() == [
        "- cardboard (confidence: 0.872)",
        "- plastic (confidence: 0.931)",
    ]


def test_context_from_another_request_is_rejected():
    context = make_context(retrieval=make_retrieval()).model_copy(
        update={"request_id": "different-request"}
    )
    adapter = GeminiGenerationAdapter(client=make_client("Answer [C1]."))

    with pytest.raises(ValueError, match="request_id"):
        adapter(make_request(), context)
