"""Bounded, evidence-validated conversation-context recovery."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any
from dataclasses import dataclass
from typing import Protocol

from smart_waste_multimodal.contracts.context import (
    ContextAnalysisOutput,
    ContextAssessment,
    ContextStatus,
)
from smart_waste_multimodal.contracts.conversation import ConversationTurn
from smart_waste_multimodal.query_rewriting.context_validator import (
    ContextAnalysisValidator,
)
from smart_waste_multimodal.evidence.interfaces import (
    ConversationEvidenceRegistry,
)
from smart_waste_multimodal.query_rewriting.context_analyzer import (
    ContextAnalyzer,
)


class ContextRecoveryStrategy(Protocol):
    """Propose context interpretations using conversation and attached vision."""

    def propose(
        self,
        question: str,
        conversation_history: Sequence[ConversationTurn],
        current_assessment: ContextAssessment | None,
        attempt: int,
        previous_proposal_fingerprints: frozenset[str],
        evidence_catalog: Sequence[Mapping[str, str]],
    ) -> ContextAnalysisOutput | None:
        """Return an untrusted proposal for deterministic validation."""
        ...


@dataclass(frozen=True)
class ContextRecoveryAttempt:
    attempt: int
    outcome: str
    reason: str
    evidence_ids: tuple[str, ...] = ()
    proposal_fingerprint: str | None = None


@dataclass(frozen=True)
class ContextRecoveryResult:
    assessment: ContextAssessment | None
    resolved: bool
    attempts: tuple[ContextRecoveryAttempt, ...]
    reason: str


class GeminiContextRecoveryStrategy:
    """Use Gemini to propose a recovery grounded in the actual evidence catalog.

    The strategy is deliberately untrusted. ContextAnalysisValidator remains
    the only authority that can accept a proposal.
    """

    def __init__(self, analyzer: ContextAnalyzer) -> None:
        self.analyzer = analyzer

    def propose(
        self,
        question: str,
        conversation_history: Sequence[ConversationTurn],
        current_assessment: ContextAssessment | None,
        attempt: int,
        previous_proposal_fingerprints: frozenset[str],
        evidence_catalog: Sequence[Mapping[str, str]],
    ) -> ContextAnalysisOutput | None:
        recovery_context: dict[str, Any] = {
            "attempt": attempt,
            "previous_proposal_fingerprints": sorted(
                previous_proposal_fingerprints
            ),
        }

        if current_assessment is not None:
            recovery_context["current_assessment"] = (
                current_assessment.model_dump(mode="json")
            )

        proposal = self.analyzer.analyze(
            question,
            conversation_history,
            evidence_catalog=evidence_catalog,
            recovery_context=recovery_context,
        )

        # Recovery must not turn uncertainty into an unsupported independent
        # decision. Only a resolved proposal proceeds to validation.
        if proposal.status != "resolved":
            return None

        return proposal


class ContextRecoveryService:
    """Run one strategy proposal and validate it."""

    def __init__(
        self,
        *,
        strategy: ContextRecoveryStrategy | None,
        validator: ContextAnalysisValidator,
    ) -> None:
        self.strategy = strategy
        self.validator = validator

    @staticmethod
    def _fingerprint(output: ContextAnalysisOutput) -> str:
        payload = json.dumps(
            output.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def recover_once(
        self,
        *,
        question: str,
        conversation_history: Sequence[ConversationTurn],
        current_assessment: ContextAssessment | None,
        attempt: int,
        previous_proposal_fingerprints: frozenset[str],
        evidence_catalog: Sequence[Mapping[str, str]],
        registry: ConversationEvidenceRegistry,
        actor_id: str,
        current_turn_id: str | None = None,
        current_turn_timestamp: datetime | None = None,
    ) -> tuple[ContextAssessment | None, ContextRecoveryAttempt]:
        if self.strategy is None:
            return None, ContextRecoveryAttempt(
                attempt=attempt,
                outcome="strategy_unavailable",
                reason="No context recovery strategy was injected.",
            )

        try:
            proposal = self.strategy.propose(
                question=question,
                conversation_history=conversation_history,
                current_assessment=current_assessment,
                attempt=attempt,
                previous_proposal_fingerprints=previous_proposal_fingerprints,
                evidence_catalog=evidence_catalog,
            )
        except Exception as exc:  # noqa: BLE001
            return None, ContextRecoveryAttempt(
                attempt=attempt,
                outcome="strategy_error",
                reason=f"Strategy raised {type(exc).__name__}.",
            )

        if proposal is None:
            return None, ContextRecoveryAttempt(
                attempt=attempt,
                outcome="no_proposal",
                reason="Strategy returned no proposal.",
            )

        if not isinstance(proposal, ContextAnalysisOutput):
            return None, ContextRecoveryAttempt(
                attempt=attempt,
                outcome="invalid_proposal_type",
                reason="Strategy returned an unsupported proposal type.",
            )

        fingerprint = self._fingerprint(proposal)

        if fingerprint in previous_proposal_fingerprints:
            return None, ContextRecoveryAttempt(
                attempt=attempt,
                outcome="repeated_proposal",
                reason="An identical proposal was already evaluated.",
                evidence_ids=tuple(proposal.evidence_ids),
                proposal_fingerprint=fingerprint,
            )

        # This validator is the sole authority for accepting a proposal.
        assessment = self.validator.validate(
            proposal,
            conversation_history,
            registry,
            actor_id=actor_id,
            current_turn_id=current_turn_id,
            current_turn_timestamp=current_turn_timestamp,
        )

        if assessment.analysis_error:
            return assessment, ContextRecoveryAttempt(
                attempt=attempt,
                outcome="validation_failed",
                reason=assessment.reason,
                evidence_ids=tuple(proposal.evidence_ids),
                proposal_fingerprint=fingerprint,
            )

        if assessment.status in (
            ContextStatus.INDEPENDENT,
            ContextStatus.RESOLVED,
        ):
            return assessment, ContextRecoveryAttempt(
                attempt=attempt,
                outcome="resolved",
                reason=assessment.reason,
                evidence_ids=assessment.evidence_ids,
                proposal_fingerprint=fingerprint,
            )

        return assessment, ContextRecoveryAttempt(
            attempt=attempt,
            outcome="still_unresolved",
            reason=assessment.reason,
            evidence_ids=assessment.evidence_ids,
            proposal_fingerprint=fingerprint,
        )


class ContextRecoveryOrchestrator:
    """Control a finite recovery cycle and prevent repeated proposals."""

    def __init__(
        self,
        *,
        strategy: ContextRecoveryStrategy | None,
        validator: ContextAnalysisValidator,
        max_attempts: int = 2,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1.")

        self.service = ContextRecoveryService(
            strategy=strategy,
            validator=validator,
        )
        self.max_attempts = max_attempts

    def resolve(
        self,
        *,
        question: str,
        conversation_history: Sequence[ConversationTurn],
        initial_assessment: ContextAssessment | None,
        registry: ConversationEvidenceRegistry,
        actor_id: str,
        current_turn_id: str | None = None,
        current_turn_timestamp: datetime | None = None,
        evidence_catalog: Sequence[Mapping[str, str]] = (),
    ) -> ContextRecoveryResult:
        current = initial_assessment
        records: list[ContextRecoveryAttempt] = []
        fingerprints: set[str] = set()

        if (
            current is not None
            and not current.analysis_error
            and current.status in (
                ContextStatus.INDEPENDENT,
                ContextStatus.RESOLVED,
            )
        ):
            return ContextRecoveryResult(
                assessment=current,
                resolved=True,
                attempts=(),
                reason="Initial assessment was already valid.",
            )

        if self.service.strategy is None:
            return ContextRecoveryResult(
                assessment=current,
                resolved=False,
                attempts=(),
                reason="No context recovery strategy is configured.",
            )

        for number in range(1, self.max_attempts + 1):
            assessment, record = self.service.recover_once(
                question=question,
                conversation_history=conversation_history,
                current_assessment=current,
                attempt=number,
                previous_proposal_fingerprints=frozenset(fingerprints),
                evidence_catalog=evidence_catalog,
                registry=registry,
                actor_id=actor_id,
                current_turn_id=current_turn_id,
                current_turn_timestamp=current_turn_timestamp,
            )
            records.append(record)

            if record.proposal_fingerprint:
                if record.outcome == "repeated_proposal":
                    break
                fingerprints.add(record.proposal_fingerprint)

            if assessment is not None:
                current = assessment

            if record.outcome == "resolved":
                return ContextRecoveryResult(
                    assessment=current,
                    resolved=True,
                    attempts=tuple(records),
                    reason="Context was accepted by the validator.",
                )

            if record.outcome in (
                "no_proposal",
                "repeated_proposal",
                "strategy_unavailable",
                "invalid_proposal_type",
            ):
                break

        reason = (
            records[-1].reason
            if records
            else "No recovery attempt was performed."
        )
        return ContextRecoveryResult(
            assessment=current,
            resolved=False,
            attempts=tuple(records),
            reason=reason,
        )
