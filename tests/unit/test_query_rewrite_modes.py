import json

from smart_waste_multimodal.query_rewriting.service import QueryRewriteService


def valid_response(query):
    return json.dumps({
        "rewritten_query": query,
        "preserved_aspects": ["material", "task"],
    })


def test_natural_only_never_calls_rewriter():
    calls = []
    service = QueryRewriteService(
        rewriter=lambda messages: calls.append(messages) or valid_response("rewritten"),
        mode="natural_only",
    )

    outcome = service.rewrite_for_retrieval("Recycle plastic bottles")

    assert outcome.retrieval_query == "Recycle plastic bottles"
    assert outcome.action == "natural_only"
    assert calls == []


def test_rewrite_always_calls_rewriter_for_simple_query():
    calls = []
    service = QueryRewriteService(
        rewriter=lambda messages: calls.append(messages)
        or valid_response("How to recycle plastic bottles"),
        mode="rewrite_always",
    )

    outcome = service.rewrite_for_retrieval("Recycle plastic bottles")

    assert len(calls) == 1
    assert outcome.retrieval_query == "How to recycle plastic bottles"
    assert outcome.action == "rewritten"


def test_rewrite_always_falls_back_when_rewriter_is_missing():
    service = QueryRewriteService(rewriter=None, mode="rewrite_always")

    outcome = service.rewrite_for_retrieval("Recycle plastic bottles")

    assert outcome.retrieval_query == "Recycle plastic bottles"
    assert outcome.action == "fallback_original"
    assert outcome.reason == "rewriter_unavailable"


def test_selective_mode_keeps_simple_query_direct():
    calls = []
    service = QueryRewriteService(
        rewriter=lambda messages: calls.append(messages) or valid_response("rewritten"),
        mode="selective",
    )

    outcome = service.rewrite_for_retrieval("Recycle plastic bottles")

    assert outcome.retrieval_query == "Recycle plastic bottles"
    assert calls == []


def test_invalid_mode_is_rejected():
    try:
        QueryRewriteService(rewriter=None, mode="invalid")
    except ValueError:
        pass
    else:
        raise AssertionError("An invalid rewrite mode should raise ValueError")


def test_natural_only_still_rejects_unresolved_context():
    import pytest

    from smart_waste_multimodal.contracts.context import (
        ContextAssessment,
        ContextStatus,
    )

    service = QueryRewriteService(rewriter=None, mode="natural_only")

    with pytest.raises(ValueError, match="Context must be resolved"):
        service.rewrite_for_retrieval(
            "What about that one?",
            context_assessment=ContextAssessment(
                status=ContextStatus.AMBIGUOUS,
            ),
        )
