import pytest

from smart_waste_multimodal.query_rewriting.selective_fallback import (
    ContextAssessment,
    ContextStatus,
    RewriteAction,
    RewritePolicyDecision,
    SelectiveRewritePolicy,
)


@pytest.fixture
def policy():
    return SelectiveRewritePolicy()


@pytest.mark.parametrize(
    "query",
    [
        "Can greasy pizza boxes be recycled?",
        "What should I do to a cardboard box before recycling it?",
        "Should broken glass go into the recycling bin?",
        "How clean should a plastic food container be before recycling?",
        "Can aluminum foil be recycled if it has food residue?",
        "How does recycling PET into textile fibres compare with using it in new packaging?",
        (
            "My old phone and laptop are no longer needed. "
            "What collection options does the guidance describe, "
            "and can they go in curbside recycling?"
        ),
        (
            "I have household lithium-ion batteries and a battery "
            "from a large electric vehicle. Why should I avoid "
            "ordinary household bins, and does the guidance "
            "recommend the same return route for both?"
        ),
    ],
)
def test_complete_independent_queries_go_direct(policy, query):
    decision = policy.decide(
        query,
        ContextAssessment(status=ContextStatus.INDEPENDENT),
    )

    assert decision == RewritePolicyDecision(
        action=RewriteAction.DIRECT,
        reason="independent_query",
        signals=(),
    )


@pytest.mark.parametrize(
    "status",
    [
        ContextStatus.INSUFFICIENT,
        ContextStatus.AMBIGUOUS,
        ContextStatus.CONFLICT,
        ContextStatus.STALE,
    ],
)
def test_unresolved_context_fails_closed(policy, status):
    with pytest.raises(ValueError, match="Context must be resolved"):
        policy.decide(
            "What about that one?",
            ContextAssessment(
                status=status,
                signals=("context_dependency",),
            ),
        )


def test_resolved_context_dependency_is_rewritten(policy):
    decision = policy.decide(
        "What about greasy ones?",
        ContextAssessment(
            status=ContextStatus.RESOLVED,
            signals=("resolved_reference", "prior_topic_available"),
        ),
    )

    assert decision.action == RewriteAction.REWRITE
    assert decision.reason == "context_dependency_resolved"
    assert decision.signals == (
        "resolved_reference",
        "prior_topic_available",
    )


def test_possible_context_dependency_without_assessment_fails_closed(
    policy,
):
    with pytest.raises(ValueError, match="Context assessment is required"):
        policy.decide("What about broken glass?")


def test_complete_query_with_context_marker_can_still_go_direct(policy):
    decision = policy.decide(
        "How should I sort glass, and what about broken glass?",
        ContextAssessment(status=ContextStatus.INDEPENDENT),
    )

    assert decision.action == RewriteAction.DIRECT


def test_independent_multi_part_query_goes_direct(policy):
    decision = policy.decide(
        "How should I sort glass, and what should I do with a metal bottle cap?",
        ContextAssessment(status=ContextStatus.INDEPENDENT),
    )

    assert decision.action == RewriteAction.DIRECT


def test_non_string_query_is_rejected(policy):
    with pytest.raises(TypeError, match="query must be a string"):
        policy.decide(None)


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_empty_query_is_rejected(policy, query):
    with pytest.raises(ValueError, match="must not be empty"):
        policy.decide(query)


def test_invalid_context_assessment_is_rejected(policy):
    with pytest.raises(TypeError, match="ContextAssessment"):
        policy.decide(
            "What about that one?",
            context_assessment="resolved",
        )
