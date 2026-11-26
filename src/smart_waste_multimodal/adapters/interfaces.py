"""Stable callable interfaces for pipeline stage adapters."""

from typing import Protocol

from smart_waste_multimodal.contracts import (
    AnalyzeRequest,
    FinalEvidenceContext,
    GenerationResult,
    RetrievalResult,
    VisionResult,
)
from smart_waste_multimodal.contracts.context import ContextAssessment


class VisionStage(Protocol):
    """Callable interface for the vision adapter."""

    def __call__(self, request: AnalyzeRequest) -> VisionResult: ...


class RetrievalStage(Protocol):
    """Callable interface for the retrieval adapter."""

    def __call__(
        self,
        request: AnalyzeRequest,
        vision: VisionResult | None,
        context_assessment: ContextAssessment | None = None,
    ) -> RetrievalResult:
        ...


class GenerationStage(Protocol):
    """Callable interface for generation using authorized evidence only."""

    def __call__(
        self,
        request: AnalyzeRequest,
        final_evidence_context: FinalEvidenceContext,
    ) -> GenerationResult: ...
