"""Application composition root for local and container deployment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import onnxruntime as ort

from smart_waste_multimodal.api.application_service import (
    AnalyzeApplicationService,
)
from smart_waste_multimodal.conversation.in_memory_conversation_store import (
    InMemoryConversationStore,
)
from smart_waste_multimodal.evidence.in_memory_registry import (
    ConversationScopedInMemoryEvidenceRegistryFactory,
)
from smart_waste_multimodal.generation.gemini_adapter import (
    GeminiGenerationAdapter,
)
from smart_waste_multimodal.orchestration.orchestrator import (
    PipelineOrchestrator,
)
from smart_waste_multimodal.orchestration.context_recovery import (
    GeminiContextRecoveryStrategy,
)
from smart_waste_multimodal.query_rewriting.context_analyzer import (
    ContextAnalyzer,
)
from smart_waste_multimodal.query_rewriting.service import (
    QueryRewriteService,
)
from smart_waste_multimodal.request_execution.in_memory_request_execution_store import (
    InMemoryRequestExecutionStore,
)
from smart_waste_multimodal.retrieval.bootstrap import (
    build_rag_adapter,
)
from smart_waste_multimodal.vision.in_memory_image_loader import (
    InMemoryImageLoader,
)
from smart_waste_multimodal.vision.in_memory_image_store import (
    InMemoryImageStore,
)
from smart_waste_multimodal.vision.onnx_adapter import (
    ONNXVisionAdapter,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ApplicationServices:
    actor_id: str
    conversation_store: InMemoryConversationStore
    image_store: InMemoryImageStore
    request_execution_store: InMemoryRequestExecutionStore
    evidence_registry_factory: ConversationScopedInMemoryEvidenceRegistryFactory
    orchestrator: PipelineOrchestrator
    application_service: AnalyzeApplicationService
    onnx_session: object


def _environment_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    if not value:
        return default.resolve()
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _has_gemini_key() -> bool:
    return bool(
        os.getenv("GEMINI_API_KEY", "").strip()
        or os.getenv("GOOGLE_API_KEY", "").strip()
    )


def build_application_services() -> ApplicationServices:
    """Build one process-scoped graph of stores and pipeline adapters."""

    actor_id = os.getenv("SMART_WASTE_ACTOR_ID", "local-user").strip()
    if not actor_id or len(actor_id) > 200:
        raise RuntimeError("SMART_WASTE_ACTOR_ID is invalid.")

    model_path = _environment_path(
        "SMART_WASTE_ONNX_MODEL",
        PROJECT_ROOT / "exports" / "best.onnx",
    )
    index_dir = _environment_path(
        "SMART_WASTE_RAG_INDEX_DIR",
        PROJECT_ROOT / "data" / "retrieval_index" / "v4_expanded",
    )

    if not model_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    if not (index_dir / "index.faiss").is_file():
        raise FileNotFoundError(f"FAISS index not found in: {index_dir}")

    if not (index_dir / "chunks.json").is_file():
        raise FileNotFoundError(f"Retrieval chunks not found in: {index_dir}")

    # Shared process-scoped stores. Never instantiate these per request.
    conversation_store = InMemoryConversationStore()
    image_store = InMemoryImageStore()
    request_execution_store = InMemoryRequestExecutionStore()
    evidence_registry_factory = (
        ConversationScopedInMemoryEvidenceRegistryFactory()
    )

    image_loader = InMemoryImageLoader(image_store)

    onnx_session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )

    vision_stage = ONNXVisionAdapter(
        image_loader=image_loader,
        session=onnx_session,
    )

    # Selective rewriting remains enabled. Without a Gemini key, the
    # QueryRewriteService safely falls back to the original user query.
    gemini_rewriter = None
    if _has_gemini_key():
        from smart_waste_multimodal.query_rewriting.gemini_rewriter import (
            build_gemini_rewriter,
        )
        gemini_rewriter = build_gemini_rewriter()

    rewrite_service = QueryRewriteService(
        rewriter=gemini_rewriter,
        mode="selective",
    )

    retrieval_stage = build_rag_adapter(
        index_dir=index_dir,
        rewrite_mode="selective",
        rewrite_service=rewrite_service,
    )

    # The adapter initializes its Gemini client lazily. The HTTP layer
    # prevents generation requests when no Gemini key is configured.
    generation_stage = GeminiGenerationAdapter()

    context_analyzer = ContextAnalyzer()
    context_recovery_strategy = (
        GeminiContextRecoveryStrategy(context_analyzer)
        if _has_gemini_key()
        else None
    )

    orchestrator = PipelineOrchestrator(
        vision_stage=vision_stage,
        retrieval_stage=retrieval_stage,
        generation_stage=generation_stage,
        evidence_registry_factory=evidence_registry_factory,
        context_analyzer=context_analyzer,
        context_recovery_strategy=context_recovery_strategy,
    )

    application_service = AnalyzeApplicationService(
        conversation_store=conversation_store,
        image_store=image_store,
        request_execution_store=request_execution_store,
        evidence_registry_factory=evidence_registry_factory,
        orchestrator=orchestrator,
    )

    return ApplicationServices(
        actor_id=actor_id,
        conversation_store=conversation_store,
        image_store=image_store,
        request_execution_store=request_execution_store,
        evidence_registry_factory=evidence_registry_factory,
        orchestrator=orchestrator,
        application_service=application_service,
        onnx_session=onnx_session,
    )
