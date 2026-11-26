from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from smart_waste_multimodal.query_rewriting import service as service_module
from smart_waste_multimodal.query_rewriting.selective_fallback import (
    ContextAssessment,
    ContextStatus,
    RewriteAction,
    RewritePolicyDecision,
)
from smart_waste_multimodal.query_rewriting.service import (
    QueryRewriteService,
)


def make_resolved_assessment():
    return ContextAssessment(
        status=ContextStatus.RESOLVED,
        signals=("resolved_reference",),
    )


def make_rewrite_decision():
    return RewritePolicyDecision(
        action=RewriteAction.REWRITE,
        reason="context_dependency_resolved",
        signals=("resolved_reference",),
    )


def install_valid_rewrite_pipeline(
    monkeypatch,
    *,
    rewritten_query="plastic container recycling guidance",
    preserved_aspects=("material", "recycling_topic"),
):
    messages = [{"role": "user", "content": "test prompt"}]

    build_messages = Mock(return_value=messages)
    validate_response = Mock(
        return_value=SimpleNamespace(
            rewritten_query=rewritten_query,
            preserved_aspects=preserved_aspects,
        )
    )

    monkeypatch.setattr(
        service_module,
        "build_rewrite_messages",
        build_messages,
    )
    monkeypatch.setattr(
        service_module,
        "validate_rewrite_response",
        validate_response,
    )

    return build_messages, validate_response, messages


def test_direct_action_preserves_original_query_without_rewriter():
    service = QueryRewriteService(
        rewriter=Mock(),
        mode="selective",
    )

    result = service.rewrite_for_retrieval(
        "How do I recycle plastic?",
        context_assessment=ContextAssessment(
            status=ContextStatus.INDEPENDENT,
        ),
    )

    assert result.original_query == "How do I recycle plastic?"
    assert result.retrieval_query == "How do I recycle plastic?"
    assert result.action == "direct"
    assert result.reason == "independent_query"
    assert result.preserved_aspects == ()


def test_direct_action_does_not_call_rewriter():
    rewriter = Mock()
    service = QueryRewriteService(
        rewriter=rewriter,
        mode="selective",
    )

    service.rewrite_for_retrieval(
        "How do I recycle glass?",
        context_assessment=ContextAssessment(
            status=ContextStatus.INDEPENDENT,
        ),
    )

    rewriter.assert_not_called()


def test_unresolved_context_fails_closed_without_human_review_decision():
    rewriter = Mock()
    service = QueryRewriteService(
        rewriter=rewriter,
        mode="selective",
    )

    original_query = "What about the other one?"

    with pytest.raises(ValueError, match="Context must be resolved"):
        service.rewrite_for_retrieval(
            original_query,
            context_assessment=ContextAssessment(
                status=ContextStatus.AMBIGUOUS,
                signals=("ambiguous_reference",),
            ),
        )

    rewriter.assert_not_called()


def test_rewrite_action_uses_validated_rewritten_query(monkeypatch):
    build_messages, validate_response, messages = (
        install_valid_rewrite_pipeline(monkeypatch)
    )

    rewriter = Mock(return_value="raw model response")

    service = QueryRewriteService(
        rewriter=rewriter,
        mode="selective",
    )

    result = service.rewrite_for_retrieval(
        "What about greasy ones?",
        context_assessment=make_resolved_assessment(),
    )

    build_messages.assert_called_once_with(
        "What about greasy ones?",
        context_assessment=make_resolved_assessment(),
    )
    rewriter.assert_called_once_with(messages)
    validate_response.assert_called_once_with("raw model response")

    assert result.original_query == "What about greasy ones?"
    assert result.retrieval_query == (
        "plastic container recycling guidance"
    )
    assert result.action == "rewritten"
    assert result.reason == "context_dependency_resolved"
    assert result.preserved_aspects == (
        "material",
        "recycling_topic",
    )


def test_rewrite_unchanged_returns_original_query(monkeypatch):
    install_valid_rewrite_pipeline(
        monkeypatch,
        rewritten_query="What about greasy ones?",
    )

    service = QueryRewriteService(
        rewriter=Mock(return_value="raw response"),
        mode="selective",
    )

    result = service.rewrite_for_retrieval(
        "What about greasy ones?",
        context_assessment=make_resolved_assessment(),
    )

    assert result.retrieval_query == "What about greasy ones?"
    assert result.action == "rewrite_unchanged"
    assert result.reason == "context_dependency_resolved"


def test_missing_rewriter_falls_back_to_original_query():
    service = QueryRewriteService(
        rewriter=None,
        mode="selective",
    )

    result = service.rewrite_for_retrieval(
        "What about greasy ones?",
        context_assessment=make_resolved_assessment(),
    )

    assert result.original_query == "What about greasy ones?"
    assert result.retrieval_query == "What about greasy ones?"
    assert result.action == "fallback_original"
    assert result.reason == "rewriter_unavailable"


def test_rewriter_exception_falls_back_to_original_query(monkeypatch):
    install_valid_rewrite_pipeline(monkeypatch)

    rewriter = Mock(side_effect=RuntimeError("provider unavailable"))

    service = QueryRewriteService(
        rewriter=rewriter,
        mode="selective",
    )

    result = service.rewrite_for_retrieval(
        "What about greasy ones?",
        context_assessment=make_resolved_assessment(),
    )

    assert result.retrieval_query == "What about greasy ones?"
    assert result.action == "fallback_original"
    assert result.reason == "rewrite_failed_or_invalid"


def test_invalid_rewrite_response_falls_back_to_original_query(
    monkeypatch,
):
    build_messages = Mock(return_value=[{"role": "user", "content": "test"}])
    validate_response = Mock(
        side_effect=ValueError("invalid model output")
    )

    monkeypatch.setattr(
        service_module,
        "build_rewrite_messages",
        build_messages,
    )
    monkeypatch.setattr(
        service_module,
        "validate_rewrite_response",
        validate_response,
    )

    service = QueryRewriteService(
        rewriter=Mock(return_value="invalid response"),
        mode="selective",
    )

    result = service.rewrite_for_retrieval(
        "What about greasy ones?",
        context_assessment=make_resolved_assessment(),
    )

    assert result.retrieval_query == "What about greasy ones?"
    assert result.action == "fallback_original"
    assert result.reason == "rewrite_failed_or_invalid"


def test_policy_exception_propagates_without_human_review_decision():
    policy = Mock()
    policy.decide.side_effect = RuntimeError("policy failure")

    rewriter = Mock()

    service = QueryRewriteService(
        rewriter=rewriter,
        policy=policy,
        mode="selective",
    )

    original_query = "What about the other one?"

    with pytest.raises(RuntimeError, match="policy failure"):
        service.rewrite_for_retrieval(original_query)

    policy.decide.assert_called_once()
    rewriter.assert_not_called()


def test_natural_only_mode_does_not_call_rewriter():
    rewriter = Mock()

    service = QueryRewriteService(
        rewriter=rewriter,
        mode="natural_only",
    )

    result = service.rewrite_for_retrieval(
        "  How do I recycle plastic?  "
    )

    rewriter.assert_not_called()

    assert result.original_query == "How do I recycle plastic?"
    assert result.retrieval_query == "How do I recycle plastic?"
    assert result.action == "natural_only"
    assert result.reason == "rewriting_disabled_by_mode"


def test_rewrite_always_mode_runs_rewriter_without_context_assessment(
    monkeypatch,
):
    build_messages, validate_response, messages = (
        install_valid_rewrite_pipeline(monkeypatch)
    )

    rewriter = Mock(return_value="raw model response")

    service = QueryRewriteService(
        rewriter=rewriter,
        mode="rewrite_always",
    )

    result = service.rewrite_for_retrieval(
        "How do I recycle plastic?"
    )

    build_messages.assert_called_once_with(
        "How do I recycle plastic?",
        context_assessment=None,
    )
    rewriter.assert_called_once_with(messages)
    validate_response.assert_called_once_with("raw model response")

    assert result.retrieval_query == (
        "plastic container recycling guidance"
    )
    assert result.action == "rewritten"
    assert result.reason == "rewrite_requested_by_mode"


def test_empty_original_query_is_rejected():
    service = QueryRewriteService(mode="selective")

    with pytest.raises(
        ValueError,
        match="original_query must not be empty",
    ):
        service.rewrite_for_retrieval("   ")


def test_unsupported_mode_is_rejected():
    with pytest.raises(ValueError, match="Unsupported mode"):
        QueryRewriteService(mode="unsupported")


def test_provider_failure_is_classified_and_preserves_original_query():
    rewriter = Mock(side_effect=RuntimeError("private provider message"))
    service = QueryRewriteService(rewriter=rewriter, mode="rewrite_always")

    result = service.rewrite_for_retrieval("How do I recycle plastic?")

    assert result.action == "fallback_original"
    assert result.retrieval_query == result.original_query
    assert result.failure_category == "provider_failure"
    assert result.failure_type == "RuntimeError"
    assert result.duration_ms >= 0
    assert "private provider message" not in repr(result)


def test_invalid_model_response_is_classified_separately(monkeypatch):
    _build_messages, validate_response, _ = install_valid_rewrite_pipeline(monkeypatch)
    validate_response.side_effect = ValueError("invalid model output")
    service = QueryRewriteService(
        rewriter=Mock(return_value="invalid"),
        mode="rewrite_always",
    )

    result = service.rewrite_for_retrieval("How do I recycle plastic?")

    assert result.action == "fallback_original"
    assert result.retrieval_query == result.original_query
    assert result.failure_category == "invalid_response"
    assert result.failure_type == "ValueError"


def test_unexpected_validation_programming_error_propagates(monkeypatch):
    install_valid_rewrite_pipeline(monkeypatch)
    service = QueryRewriteService(
        rewriter=Mock(return_value="raw model response"),
        mode="rewrite_always",
    )
    from smart_waste_multimodal.query_rewriting import service as service_module

    def broken_validator(_):
        raise RuntimeError("internal validator bug")

    monkeypatch.setattr(
        service_module, "validate_rewrite_response", broken_validator
    )

    with pytest.raises(RuntimeError, match="internal validator bug"):
        service.rewrite_for_retrieval("How do I recycle plastic?")


def test_rewriter_unavailable_has_explicit_failure_category():
    service = QueryRewriteService(rewriter=None, mode="rewrite_always")

    result = service.rewrite_for_retrieval("How do I recycle plastic?")

    assert result.action == "fallback_original"
    assert result.retrieval_query == result.original_query
    assert result.failure_category == "rewriter_unavailable"
    assert result.failure_type == "RewriterUnavailable"
