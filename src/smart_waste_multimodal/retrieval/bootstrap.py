"""Application bootstrap for the self-contained hybrid RAG adapter.

The index directory is supplied explicitly or through
SMART_WASTE_RAG_INDEX_DIR. No Legacy project path or sys.path mutation
is used here.
"""

import os
from pathlib import Path

from smart_waste_multimodal.embeddings.encoder import TextEncoder
from smart_waste_multimodal.retrieval.bm25_retriever import BM25Retriever
from smart_waste_multimodal.retrieval.hybrid_rag_adapter import (
    HybridRAGAdapter,
)
from smart_waste_multimodal.retrieval.hybrid_retriever import HybridRetriever
from smart_waste_multimodal.indexing.index_store import IndexStore
from smart_waste_multimodal.query_rewriting.service import RewriteMode


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INDEX_VERSION = "v4_expanded"
DEFAULT_MODEL_NAME = "intfloat/multilingual-e5-small"


def resolve_index_dir(
    index_dir: str | Path | None = None,
    index_version: str = DEFAULT_INDEX_VERSION,
) -> Path:
    """Resolve the configured, deployment-independent index directory."""
    configured = (
        index_dir
        or os.getenv("SMART_WASTE_RAG_INDEX_DIR")
        or PROJECT_ROOT / "data" / "retrieval_index" / index_version
    )
    return Path(configured).expanduser().resolve()


def load_hybrid_retriever(
    index_dir: str | Path,
    encoder: TextEncoder,
) -> HybridRetriever:
    """Load the existing FAISS index and build its BM25 companion."""
    vector_index, chunks = IndexStore(index_dir).load()
    bm25_retriever = BM25Retriever(chunks)

    return HybridRetriever(
        chunks=chunks,
        encoder=encoder,
        vector_index=vector_index,
        bm25_retriever=bm25_retriever,
    )


def build_rag_adapter(
    *,
    index_dir: str | Path | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    top_k: int = 5,
    index_version: str = DEFAULT_INDEX_VERSION,
    rewrite_mode: RewriteMode = "selective",
    rewrite_service=None,
) -> HybridRAGAdapter:
    """Build the internal RAG adapter around the existing persisted index."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    if rewrite_mode not in {"natural_only", "rewrite_always", "selective"}:
        raise ValueError(
            "rewrite_mode must be natural_only, rewrite_always, or selective"
        )

    resolved_index_dir = resolve_index_dir(
        index_dir=index_dir,
        index_version=index_version,
    )

    required_index_files = (
        resolved_index_dir / "index.faiss",
        resolved_index_dir / "chunks.json",
    )
    missing_index_files = [
        path for path in required_index_files
        if not path.is_file()
    ]

    if missing_index_files:
        missing_paths = ", ".join(
            str(path) for path in missing_index_files
        )
        raise FileNotFoundError(
            "Retrieval index is incomplete or unavailable. "
            f"Index directory: {resolved_index_dir}. "
            f"Missing files: {missing_paths}"
        )

    encoder = TextEncoder(model_name=model_name)
    retriever = load_hybrid_retriever(
        index_dir=resolved_index_dir,
        encoder=encoder,
    )

    if rewrite_service is None:
        from smart_waste_multimodal.query_rewriting.service import (
            QueryRewriteService,
        )

        if rewrite_mode == "natural_only":
            rewrite_service = QueryRewriteService(
                rewriter=None,
                mode=rewrite_mode,
            )
        else:
            from smart_waste_multimodal.query_rewriting.gemini_rewriter import (
                build_gemini_rewriter,
            )

            rewrite_service = QueryRewriteService(
                rewriter=build_gemini_rewriter(),
                mode=rewrite_mode,
            )

    return HybridRAGAdapter(
        retriever=retriever,
        top_k=top_k,
        index_version=index_version,
        rewrite_service=rewrite_service,
    )
