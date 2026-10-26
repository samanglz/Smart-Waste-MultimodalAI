"""Deterministic mock adapters for integration tests."""

from smart_waste_multimodal.contracts import (
    AnalyzeRequest,
    Detection,
    EvidenceItem,
    GenerationResult,
    RetrievalResult,
    VisionResult,
)


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
    """Generate an answer grounded in retrieved evidence."""

    def __call__(
        self,
        request: AnalyzeRequest,
        vision: VisionResult | None,
        retrieval: RetrievalResult | None,
    ) -> GenerationResult:
        if retrieval is None or not retrieval.results:
            return GenerationResult(
                status="insufficient_evidence",
                answer=(
                    "I cannot provide a grounded recycling recommendation "
                    "because no supporting evidence was retrieved."
                ),
                reason="No retrieval evidence was available.",
                model_version="mock-generator-v1",
            )

        material = "the item"
        if vision is not None and vision.detections:
            material = vision.detections[0].class_name

        evidence = retrieval.results[0]
        return GenerationResult(
            status="success",
            answer=(
                f"For {material}, follow the retrieved guidance: "
                f"{evidence.text}"
            ),
            citation_chunk_ids=[evidence.chunk_id],
            model_version="mock-generator-v1",
        )
