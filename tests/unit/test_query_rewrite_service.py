import json

import pytest

from smart_waste_multimodal.query_rewriting.selective_fallback import (
    ContextAssessment,
    ContextStatus,
)
from smart_waste_multimodal.query_rewriting.service import (
    QueryRewriteService,
)


def model_response(query):
    return json.dumps({
        "rewritten_query": query,
        "preserved_aspects": ["original information need"],
    })


def test_simple_query_goes_direct_without_calling_model():
    called = []

    service = QueryRewriteService(
        rewriter=lambda messages: called.append(messages)
        or model_response("unused")
    )

    result = service.rewrite_for_retrieval("Recycle plastic bottles")

    assert result.action == "direct"
    assert result.retrieval_query == "Recycle plastic bottles"
    assert called == []


def test_independent_multi_part_query_goes_direct():
    called = []

    service = QueryRewriteService(
        rewriter=lambda messages: called.append(messages)
        or model_response("unused")
    )

    query = (
        "How should I sort glass, "
        "and what should I do with a metal bottle cap?"
    )

    result = service.rewrite_for_retrieval(
        query,
        context_assessment=ContextAssessment(
            status=ContextStatus.INDEPENDENT
        ),
    )

    assert result.action == "direct"
    assert result.retrieval_query == query
    assert called == []


def test_resolved_context_dependent_query_is_rewritten():
    service = QueryRewriteService(
        rewriter=lambda messages: model_response(
            "Can greasy cardboard pizza boxes be recycled?"
        )
    )

    result = service.rewrite_for_retrieval(
        "What about greasy ones?",
        context_assessment=ContextAssessment(
            status=ContextStatus.RESOLVED,
            signals=("resolved_reference",),
            evidence_turn_ids=("turn-1",),
            resolved_context="Cardboard pizza boxes",
        ),
    )

    assert result.action == "rewritten"
    assert result.retrieval_query == (
        "Can greasy cardboard pizza boxes be recycled?"
    )
    assert result.original_query == "What about greasy ones?"


def test_unresolved_context_is_rejected_without_human_review_decision():
    called = []

    service = QueryRewriteService(
        rewriter=lambda messages: called.append(messages)
        or model_response("unused")
    )

    with pytest.raises(ValueError, match="Context must be resolved"):
        service.rewrite_for_retrieval(
            "What about the other one?",
            context_assessment=ContextAssessment(
                status=ContextStatus.AMBIGUOUS,
                signals=("multiple_possible_references",),
            ),
        )

    assert called == []


def test_missing_context_assessment_with_marker_fails_closed():
    called = []

    service = QueryRewriteService(
        rewriter=lambda messages: called.append(messages)
        or model_response("unused")
    )

    with pytest.raises(ValueError, match="Context assessment is required"):
        service.rewrite_for_retrieval("What about broken glass?")

    assert called == []


def test_candidate_query_falls_back_when_model_is_unavailable():
    service = QueryRewriteService(rewriter=None)

    result = service.rewrite_for_retrieval(
        "What about greasy ones?",
        context_assessment=ContextAssessment(
            status=ContextStatus.RESOLVED,
            evidence_turn_ids=("turn-1",),
            resolved_context="Greasy cardboard pizza boxes",
        ),
    )

    assert result.action == "fallback_original"
    assert result.retrieval_query == result.original_query
    assert result.reason == "rewriter_unavailable"


@pytest.mark.parametrize(
    "raw",
    ["not-json", '{"rewritten_query": "glass"}', " "],
)
def test_invalid_model_output_falls_back_to_original(raw):
    service = QueryRewriteService(rewriter=lambda messages: raw)

    result = service.rewrite_for_retrieval(
        "What about greasy ones?",
        context_assessment=ContextAssessment(
            status=ContextStatus.RESOLVED,
            evidence_turn_ids=("turn-1",),
            resolved_context="Greasy cardboard pizza boxes",
        ),
    )

    assert result.action == "fallback_original"
    assert result.retrieval_query == result.original_query


def test_provider_exception_falls_back_to_original():
    def fail(messages):
        raise RuntimeError("provider unavailable")

    service = QueryRewriteService(rewriter=fail)
    query = "What about greasy ones?"

    result = service.rewrite_for_retrieval(
        query,
        context_assessment=ContextAssessment(
            status=ContextStatus.RESOLVED,
            evidence_turn_ids=("turn-1",),
            resolved_context="Greasy cardboard pizza boxes",
        ),
    )

    assert result.action == "fallback_original"
    assert result.retrieval_query == query


def test_policy_failure_propagates_without_human_review_decision(monkeypatch):
    service = QueryRewriteService()

    def fail(*args, **kwargs):
        raise RuntimeError("policy failure")

    monkeypatch.setattr(service.policy, "decide", fail)

    with pytest.raises(RuntimeError, match="policy failure"):
        service.rewrite_for_retrieval("Recycle plastic bottles")


def test_natural_only_mode_does_not_call_rewriter():
    called = []

    service = QueryRewriteService(
        rewriter=lambda messages: called.append(messages)
        or model_response("unused"),
        mode="natural_only",
    )

    result = service.rewrite_for_retrieval(
        "Recycle broken glass safely"
    )

    assert result.action == "natural_only"
    assert result.retrieval_query == result.original_query
    assert called == []


def test_empty_query_is_rejected():
    service = QueryRewriteService()

    with pytest.raises(ValueError, match="must not be empty"):
        service.rewrite_for_retrieval("   ")
