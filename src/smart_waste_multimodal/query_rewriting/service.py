from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import ClassVar, Literal

from pydantic import ValidationError

from smart_waste_multimodal.contracts.context import ContextAssessment
from smart_waste_multimodal.query_rewriting.prompt_builder import (
    build_rewrite_messages,
)
from smart_waste_multimodal.query_rewriting.selective_fallback import (
    RewriteAction,
    SelectiveRewritePolicy,
)
from smart_waste_multimodal.query_rewriting.validation import (
    validate_rewrite_response,
)

RewriteMode = Literal["natural_only", "rewrite_always", "selective"]


@dataclass(frozen=True)
class QueryRewriteOutcome:
    original_query: str
    retrieval_query: str
    action: str
    reason: str
    preserved_aspects: tuple[str, ...] = ()
    duration_ms: float = 0.0
    failure_category: str | None = None
    failure_type: str | None = None


class QueryRewriteService:
    """Apply a rewriting strategy with safe, observable fallback behavior."""

    VALID_MODES: ClassVar[frozenset[str]] = frozenset(
        {"natural_only", "rewrite_always", "selective"}
    )

    def __init__(
        self,
        rewriter: Callable[[list[dict[str, str]]], str] | None = None,
        policy: SelectiveRewritePolicy | None = None,
        mode: RewriteMode = "selective",
    ) -> None:
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Unsupported mode {mode!r}; expected one of "
                f"{sorted(self.VALID_MODES)}"
            )

        self.rewriter = rewriter
        self.policy = policy or SelectiveRewritePolicy()
        self.mode: RewriteMode = mode

    def rewrite_for_retrieval(
        self,
        original_query: str,
        context_assessment: ContextAssessment | None = None,
    ) -> QueryRewriteOutcome:
        started = perf_counter()

        if not isinstance(original_query, str) or not original_query.strip():
            raise ValueError("original_query must not be empty")

        original_query = original_query.strip()

        def outcome(
            *,
            retrieval_query: str,
            action: str,
            reason: str,
            preserved_aspects: tuple[str, ...] = (),
            failure_category: str | None = None,
            failure_type: str | None = None,
        ) -> QueryRewriteOutcome:
            return QueryRewriteOutcome(
                original_query=original_query,
                retrieval_query=retrieval_query,
                action=action,
                reason=reason,
                preserved_aspects=preserved_aspects,
                duration_ms=max(0.0, (perf_counter() - started) * 1000),
                failure_category=failure_category,
                failure_type=failure_type,
            )

        # Context safety applies to every mode, including experiment modes.
        # The policy never makes HUMAN_REVIEW decisions.
        decision = self.policy.decide(
            original_query,
            context_assessment=context_assessment,
        )

        if self.mode == "natural_only":
            return outcome(
                retrieval_query=original_query,
                action="natural_only",
                reason="rewriting_disabled_by_mode",
            )

        if self.mode == "selective":
            if decision.action == RewriteAction.DIRECT:
                return outcome(
                    retrieval_query=original_query,
                    action="direct",
                    reason=decision.reason,
                )

            if decision.action != RewriteAction.REWRITE:
                raise ValueError(
                    f"Unsupported selective rewrite action: {decision.action!r}"
                )

            policy_reason = decision.reason
        else:
            # rewrite_always is experimental and still passes the context gate.
            policy_reason = "rewrite_requested_by_mode"

        if self.rewriter is None:
            return outcome(
                retrieval_query=original_query,
                action="fallback_original",
                reason="rewriter_unavailable",
                failure_category="rewriter_unavailable",
                failure_type="RewriterUnavailable",
            )

        # Prompt construction errors are internal errors and must propagate.
        messages = build_rewrite_messages(
            original_query,
            context_assessment=context_assessment,
        )

        # Only exceptions raised at the provider boundary are provider failures.
        try:
            raw_response = self.rewriter(messages)
        except Exception as exc:  # noqa: BLE001
            return outcome(
                retrieval_query=original_query,
                action="fallback_original",
                reason="rewrite_failed_or_invalid",
                failure_category="provider_failure",
                failure_type=type(exc).__name__,
            )

        # Validation failures are expected model-output failures. Unexpected
        # programming errors are deliberately not swallowed.
        try:
            validated = validate_rewrite_response(raw_response)
            rewritten_query = validated.rewritten_query.strip()

            if not rewritten_query:
                raise ValueError("Validated rewritten query is empty")
        except (ValidationError, ValueError) as exc:
            return outcome(
                retrieval_query=original_query,
                action="fallback_original",
                reason="rewrite_failed_or_invalid",
                failure_category="invalid_response",
                failure_type=type(exc).__name__,
            )

        if rewritten_query == original_query:
            return outcome(
                retrieval_query=original_query,
                action="rewrite_unchanged",
                reason=policy_reason,
                preserved_aspects=tuple(validated.preserved_aspects),
            )

        return outcome(
            retrieval_query=rewritten_query,
            action="rewritten",
            reason=policy_reason,
            preserved_aspects=tuple(validated.preserved_aspects),
        )
