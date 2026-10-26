"""Stable callable interfaces for pipeline stage adapters."""

from typing import Protocol

from smart_waste_multimodal.contracts import (
    AnalyzeRequest,
    GenerationResult,
    RetrievalResult,
    VisionResult,
)


class VisionStage(Protocol):
    """Callable interface for the vision adapter."""

    def __call__(self, request: AnalyzeRequest) -> VisionResult: ...


class RetrievalStage(Protocol):
    """Callable interface for the retrieval adapter."""

    def __call__(
        self,
        request: AnalyzeRequest,
        vision: VisionResult | None,
    ) -> RetrievalResult: ...


class GenerationStage(Protocol):
    """Callable interface for the generation adapter."""

    def __call__(
        self,
        request: AnalyzeRequest,
        vision: VisionResult | None,
        retrieval: RetrievalResult | None,
    ) -> GenerationResult: ...
