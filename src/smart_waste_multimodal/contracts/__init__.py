"""Versioned data contracts for the multimodal pipeline."""

from .common import ContractModel, ContractVersion, ErrorInfo, ErrorStage
from .generation import GenerationResult, GenerationStatus
from .request import AnalyzeRequest, ImageArtifact, RequestOptions
from .response import AnalyzeResponse, ResponseStatus
from .retrieval import EvidenceItem, RetrievalResult, RetrievalStatus
from .vision import Detection, VisionResult, VisionStatus

__all__ = [
    "AnalyzeRequest", "AnalyzeResponse", "ContractModel",
    "ContractVersion", "Detection", "ErrorInfo", "ErrorStage",
    "EvidenceItem", "GenerationResult", "GenerationStatus",
    "ImageArtifact", "RequestOptions", "ResponseStatus",
    "RetrievalResult", "RetrievalStatus", "VisionResult",
    "VisionStatus",
]
