"""Deterministic mock adapters for integration tests."""

from smart_waste_multimodal.contracts import (
    AnalyzeRequest,
    Detection,
    EvidenceItem,
    FinalEvidenceContext,
    GenerationResult,
    RetrievalResult,
    VisionResult,
)
from smart_waste_multimodal.contracts.context import ContextAssessment
from smart_waste_multimodal.contracts.evidence import EvidenceSourceType


class MockVisionAdapter:
    """Return a deterministic plastic detection."""

    def __call__(self, request: AnalyzeRequest) -> VisionResult:
        width = request.image.width or 100
        height = request.image.height or 100

        # Use a valid box within the dimensions declared by the request.
        x2 = min(40.0, float(width))
        y2 = min(40.0, float(height))
        x1 = min(10.0, x2 / 4)
        y1 = min(10.0, y2 / 4)

        return VisionResult(
            status="success",
            detections=[
                Detection(
                    class_name="plastic",
                    confidence=0.95,
                    bbox_xyxy=(x1, y1, x2, y2),
                )
            ],
            image_width=width,
            image_height=height,
            model_version="mock-vision-v1",
            artifact_id=request.image.artifact_id,
        )


class MockRetrievalAdapter:
    """Return deterministic recycling evidence."""

    def __call__(
        self,
        request: AnalyzeRequest,
        vision: VisionResult | None,
        context_assessment: ContextAssessment | None = None,
    ) -> RetrievalResult:
        if request.options.retrieval_query_source == "vision_detections":
            if vision is None or not vision.detections:
                return RetrievalResult(
                    status="no_evidence",
                    query="detected waste material",
                    results=[],
                    index_version="mock-index-v1",
                )
            detected_classes = ", ".join(
                detection.class_name for detection in vision.detections
            )
            query = f"How to recycle {detected_classes}"
        else:
            query = (request.question or "").strip()

        if not query:
            query = "recycling guidance"

        return RetrievalResult(
            status="success",
            query=query,
            results=[
                EvidenceItem(
                    chunk_id="mock-recycling-chunk-001",
                    source="mock-recycling-guide",
                    text="Clean plastic containers before recycling.",
                    score=0.82,
                    rank=1,
                )
            ],
            index_version="mock-index-v1",
        )


class MockGenerationAdapter:
    """Generate an answer from authorized Final Evidence Context records."""

    def __call__(
        self,
        request: AnalyzeRequest,
        final_evidence_context: FinalEvidenceContext,
    ) -> GenerationResult:
        if final_evidence_context.request_id != request.request_id:
            raise ValueError(
                "Final Evidence Context request_id does not match the request."
            )

        retrieval_evidence = [
            item
            for item in final_evidence_context.evidence
            if item.source_type == EvidenceSourceType.RETRIEVAL_CHUNK
            and isinstance(item.content, dict)
            and item.content.get("chunk_id") == item.provenance.source_id
        ]

        if not retrieval_evidence:
            return GenerationResult(
                status="insufficient_evidence",
                answer=(
                    "I cannot provide a grounded recycling recommendation "
                    "because no supporting retrieval evidence is available."
                ),
                reason="No authorized retrieval evidence was available.",
                model_version="mock-generator-v1",
            )

        material = "the item"
        for item in final_evidence_context.evidence:
            if item.source_type != EvidenceSourceType.VISION_DETECTION:
                continue
            if isinstance(item.content, dict):
                class_name = item.content.get("class_name")
                if isinstance(class_name, str) and class_name.strip():
                    material = class_name.strip()
                    break

        evidence = retrieval_evidence[0]
        text = evidence.content.get("text", "")
        chunk_id = evidence.provenance.source_id

        return GenerationResult(
            status="success",
            answer=(
                f"For {material}, follow the retrieved guidance: {text}"
            ),
            citation_chunk_ids=[chunk_id],
            model_version="mock-generator-v1",
        )
