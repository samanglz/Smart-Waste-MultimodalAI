import json

import pytest

from smart_waste_multimodal.query_rewriting.prompt_builder import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    build_rewrite_messages,
)


def test_prompt_builder_returns_system_and_user_messages():
    messages = build_rewrite_messages(
        "How should I sort different kinds of glass, and what about broken glass?"
    )

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_original_query_is_preserved_in_user_payload():
    query = "How should I sort different kinds of glass, and what about broken glass?"
    messages = build_rewrite_messages(query)

    payload = json.loads(messages[1]["content"])
    assert payload["original_query"] == query


def test_prompt_contract_contains_multi_part_preservation_rules():
    prompt = SYSTEM_PROMPT.lower()

    assert "every distinct sub-question" in prompt
    assert "safety concerns" in prompt
    assert "do not invent information" in prompt
    assert "ground-truth answers" in prompt
    assert "do not answer the question" in prompt
    assert "source of truth" in prompt
    assert "resolved conversation context" in prompt


def test_prompt_version_is_explicit():
    assert PROMPT_VERSION == "query-rewriting-v2"


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_empty_query_is_rejected(query):
    with pytest.raises(ValueError):
        build_rewrite_messages(query)


def test_non_string_query_is_rejected():
    with pytest.raises(TypeError):
        build_rewrite_messages(None)



def test_resolved_context_and_provenance_are_included_in_prompt():
    from smart_waste_multimodal.contracts.context import (
        ContextAssessment,
        ContextStatus,
    )

    query = "What about greasy ones?"
    assessment = ContextAssessment(
        status=ContextStatus.RESOLVED,
        evidence_turn_ids=("turn-17",),
        resolved_context="The previous question discussed cardboard pizza boxes.",
    )

    messages = build_rewrite_messages(
        query,
        context_assessment=assessment,
    )

    payload = json.loads(messages[1]["content"])

    assert payload["original_query"] == query
    assert payload["resolved_context"] == (
        "The previous question discussed cardboard pizza boxes."
    )
    assert payload["evidence_turn_ids"] == ["turn-17"]


def test_independent_context_does_not_add_resolved_context_to_prompt():
    from smart_waste_multimodal.contracts.context import (
        ContextAssessment,
        ContextStatus,
    )

    messages = build_rewrite_messages(
        "Recycle plastic bottles",
        context_assessment=ContextAssessment(
            status=ContextStatus.INDEPENDENT,
        ),
    )

    payload = json.loads(messages[1]["content"])

    assert "resolved_context" not in payload
    assert "evidence_turn_ids" not in payload
