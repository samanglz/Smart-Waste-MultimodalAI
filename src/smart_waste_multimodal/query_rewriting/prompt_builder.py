"""Versioned prompt construction for retrieval-oriented query rewriting."""

from __future__ import annotations

import json

from smart_waste_multimodal.contracts.context import (
    ContextAssessment,
    ContextStatus,
)

PROMPT_VERSION = "query-rewriting-v2"

SYSTEM_PROMPT = """
You are a query rewriter for a hybrid information retrieval system.

Your task is to rewrite the user's original query so that relevant evidence
is easier to retrieve. You are NOT answering the user's question.

CORE RULES

1. Preserve the original intent.
   - Preserve every distinct sub-question and information need.
   - Preserve entities, material names, product types, constraints, conditions,
     comparisons, quantities, and safety concerns.
   - For multi-part questions, ensure that no meaningful part is lost.

2. Improve retrieval clarity.
   - Use concise, precise terms that express the original information needs.
   - Add useful synonyms only when they preserve the original meaning.
   - Keep important lexical terms that may help hybrid lexical and semantic
     retrieval.
   - Do not merely concatenate keywords or produce a keyword dump.

3. Do not invent information.
   - Do not add assumptions, facts, answers, entities, locations, or constraints
     that are not supported by the original query or the supplied resolved
     conversation context.
   - Do not silently narrow a broad question to one subtopic.
   - Do not change the requested jurisdiction or local context.

4. Preserve the evidence boundary.
   - The original query is the source of truth and must remain fully represented.
   - Resolved conversation context may only be used to resolve references
     already present in the original query.
   - Do not use context to remove, narrow, or replace any original information
     need.
   - Do not assume access to target documents, chunk IDs, relevance labels,
     ground-truth answers, or evaluation results.
   - Never tailor the rewrite to a known target chunk.

5. Do not answer the question.
   - Return a rewritten query, not advice, an explanation, or a response.
   - Keep the rewrite concise while preserving all meaningful information.

OUTPUT CONTRACT

Return valid JSON only, with exactly these fields:
{
  "rewritten_query": "the rewritten query",
  "preserved_aspects": [
    "a brief description of each distinct information need preserved"
  ]
}

The rewritten_query must be usable directly as a retrieval query.
The preserved_aspects field is for auditing coverage of the original query.
Do not include markdown fences or additional fields.
""".strip()


def build_rewrite_messages(
    original_query: str,
    context_assessment: ContextAssessment | None = None,
) -> list[dict[str, str]]:
    """Build rewrite messages using only the original query and validated context."""
    if not isinstance(original_query, str):
        raise TypeError("original_query must be a string")

    query = original_query.strip()
    if not query:
        raise ValueError("original_query must not be empty")

    user_payload: dict[str, object] = {
        "task": "Rewrite this query for information retrieval.",
        "original_query": query,
    }

    if context_assessment is not None:
        if not isinstance(context_assessment, ContextAssessment):
            raise TypeError(
                "context_assessment must be a ContextAssessment instance or None"
            )

        if context_assessment.status == ContextStatus.RESOLVED:
            resolved_context = context_assessment.resolved_context
            evidence_turn_ids = context_assessment.evidence_turn_ids

            if (
                not isinstance(resolved_context, str)
                or not resolved_context.strip()
            ):
                raise ValueError(
                    "Resolved context must contain non-empty resolved_context."
                )

            if (
                not isinstance(evidence_turn_ids, (tuple, list))
                or not evidence_turn_ids
                or any(
                    not isinstance(turn_id, str) or not turn_id.strip()
                    for turn_id in evidence_turn_ids
                )
                or len(set(evidence_turn_ids)) != len(evidence_turn_ids)
            ):
                raise ValueError(
                    "Resolved context must contain unique evidence_turn_ids."
                )

            user_payload["resolved_context"] = resolved_context.strip()
            user_payload["evidence_turn_ids"] = list(evidence_turn_ids)

        elif context_assessment.status != ContextStatus.INDEPENDENT:
            raise ValueError(
                "Only independent or resolved context may be used "
                "to build rewrite messages."
            )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(user_payload, ensure_ascii=False),
        },
    ]
