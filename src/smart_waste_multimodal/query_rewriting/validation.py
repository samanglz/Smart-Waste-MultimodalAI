"""Structural validation for LLM-generated retrieval query rewrites."""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class RewriteResult:
    rewritten_query: str
    preserved_aspects: tuple[str, ...]


def validate_rewrite_response(raw_response: str) -> RewriteResult:
    """Validate the response structure; this does not prove semantic fidelity."""
    if not isinstance(raw_response, str):
        raise TypeError("raw_response must be a string")

    raw_response = raw_response.strip()
    if not raw_response:
        raise ValueError("LLM response must not be empty")

    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise ValueError("LLM response must be valid JSON") from exc

    if not isinstance(payload, dict):
        # The input is valid JSON text, but its structure is invalid.
        raise ValueError(  # noqa: TRY004
            "LLM response must be a JSON object"
        )

    expected_keys = {"rewritten_query", "preserved_aspects"}
    if set(payload) != expected_keys:
        raise ValueError(
            f"LLM response must contain exactly these fields: "
            f"{sorted(expected_keys)}"
        )

    query = payload["rewritten_query"]
    aspects = payload["preserved_aspects"]

    if not isinstance(query, str) or not query.strip():
        raise ValueError("rewritten_query must be a non-empty string")

    if not isinstance(aspects, list):
        # Preserve the validator's established ValueError contract.
        raise ValueError(  # noqa: TRY004
            "preserved_aspects must be a list"
        )

    if any(not isinstance(item, str) or not item.strip() for item in aspects):
        raise ValueError(
            "Each preserved_aspects item must be a non-empty string"
        )

    return RewriteResult(
        rewritten_query=query.strip(),
        preserved_aspects=tuple(item.strip() for item in aspects),
    )
