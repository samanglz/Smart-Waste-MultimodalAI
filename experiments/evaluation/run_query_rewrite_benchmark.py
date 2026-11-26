"""Offline retrieval benchmark for natural, always-rewrite, and selective modes.

The frozen dataset's evidence_chunk_ids are used only as provisional retrieval
targets. The pending Ground Truth file is deliberately never loaded.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from smart_waste_multimodal.query_rewriting.service import QueryRewriteService
from smart_waste_multimodal.retrieval.bootstrap import build_rag_adapter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = (
    PROJECT_ROOT / "experiments/evaluation/natural_queries_eval_v2_frozen.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments/evaluation/benchmark_runs"
MODES = ("natural_only", "rewrite_always", "selective")


def load_frozen_dataset(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("status") != "frozen":
        raise ValueError("Expected a frozen evaluation dataset.")
    queries = data.get("queries")
    if not isinstance(queries, list) or not queries:
        raise ValueError("Dataset must contain a non-empty queries list.")

    for row in queries:
        if not isinstance(row, dict):
            raise TypeError("Every query entry must be a JSON object.")
        if not row.get("query_id") or not row.get("original_query"):
            raise ValueError("Every query needs query_id and original_query.")
        if not isinstance(row.get("evidence_chunk_ids"), list):
            raise TypeError(
                f"evidence_chunk_ids must be a list: {row.get('query_id')}"
            )
    return queries


def score_retrieval(
    expected_ids: list[str], results: list[dict[str, Any]], ks=(5, 10, 20)
) -> dict[str, Any]:
    ranked_ids = [item["chunk_id"] for item in results]
    expected = set(expected_ids)
    first_rank = next(
        (rank for rank, chunk_id in enumerate(ranked_ids, start=1)
         if chunk_id in expected),
        None,
    )

    scores: dict[str, Any] = {
        "provisional_hit": first_rank is not None,
        "provisional_reciprocal_rank": 1.0 / first_rank if first_rank else 0.0,
        "first_matching_rank": first_rank,
    }
    for k in ks:
        scores[f"provisional_hit_at_{k}"] = any(
            chunk_id in expected for chunk_id in ranked_ids[:k]
        )
    return scores


def run_benchmark(
    *,
    dataset_path: Path = DEFAULT_DATASET,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    rag_root: Path = Path("/content/drive/MyDrive/waste-detection-rag"),
    top_k: int = 20,
    rewriter: Callable[[list[dict[str, str]]], str] | None = None,
) -> Path:
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    queries = load_frozen_dataset(dataset_path)

    # Loads the existing persisted retriever. No index rebuild or legacy writes.
    base_adapter = build_rag_adapter(
        rag_root=rag_root,
        top_k=top_k,
        index_version="v4_expanded",
    )
    retriever = base_adapter.retriever

    mode_reports = {}
    for mode in MODES:
        service = QueryRewriteService(rewriter=rewriter, mode=mode)
        rows = []
        for item in queries:
            query_id = item["query_id"]
            original_query = item["original_query"]

            started = time.perf_counter()
            rewrite_outcome = service.rewrite_for_retrieval(original_query)
            retrieval_query = rewrite_outcome.retrieval_query

            raw_results = retriever.search(retrieval_query, top_k=top_k)
            latency_ms = (time.perf_counter() - started) * 1000

            retrieved = [
                {
                    "chunk_id": item["chunk_id"],
                    "source": item["source"],
                    "score": item["score"],
                    "rank": item["rank"],
                }
                for item in raw_results
            ]
            metrics = score_retrieval(item["evidence_chunk_ids"], retrieved)

            rows.append({
                "query_id": query_id,
                "difficulty": item.get("difficulty"),
                "original_query": original_query,
                "retrieval_query": retrieval_query,
                "rewrite_action": rewrite_outcome.action,
                "rewrite_reason": rewrite_outcome.reason,
                "latency_ms": round(latency_ms, 3),
                "retrieved": retrieved,
                "metrics": metrics,
            })

        count = len(rows)
        mode_reports[mode] = {
            "query_count": count,
            "llm_rewriter_configured": rewriter is not None,
            "mean_latency_ms": round(
                sum(row["latency_ms"] for row in rows) / count, 3
            ),
            "provisional_hit_rate_at_top_k": round(
                sum(row["metrics"]["provisional_hit"] for row in rows) / count,
                4,
            ),
            "provisional_mrr": round(
                sum(row["metrics"]["provisional_reciprocal_rank"] for row in rows)
                / count,
                4,
            ),
            "provisional_hit_rate_at_5": round(
                sum(row["metrics"]["provisional_hit_at_5"] for row in rows)
                / count,
                4,
            ),
            "queries": rows,
        }

    report = {
        "benchmark_id": "query-rewrite-retrieval-benchmark-v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset_path": str(dataset_path),
        "dataset_query_count": len(queries),
        "index_version": "v4_expanded",
        "top_k": top_k,
        "ground_truth_used": False,
        "metric_warning": (
            "Metrics are provisional: they match retrieved chunk IDs against "
            "evidence_chunk_ids in the frozen dataset, not approved human labels."
        ),
        "llm_warning": (
            "No LLM rewrite quality is measured when no rewriter is configured. "
            "rewrite_always and selective may fall back to the original query."
        ),
        "modes": mode_reports,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"query_rewrite_benchmark_{timestamp}.json"
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--rag-root",
        type=Path,
        default=Path("/content/drive/MyDrive/waste-detection-rag"),
    )
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    output = run_benchmark(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        rag_root=args.rag_root,
        top_k=args.top_k,
    )
    print(f"Benchmark report written: {output}")


if __name__ == "__main__":
    main()
