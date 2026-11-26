"""Deterministic validation of semantic context-analysis output."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from smart_waste_multimodal.config.context import ContextConfig
from smart_waste_multimodal.contracts.context import (
    ContextAnalysisOutput,
    ContextAssessment,
    ContextStatus,
)
from smart_waste_multimodal.contracts.conversation import ConversationTurn
from smart_waste_multimodal.contracts.evidence import EvidenceSourceType
from smart_waste_multimodal.evidence.interfaces import (
    ConversationEvidenceRegistry,
)


class ContextValidationError(ValueError):
    """Raised when context-analysis evidence is structurally invalid."""


class ContextAnalysisValidator:
    """Validate context references against conversation evidence."""

    def __init__(self, config: ContextConfig | None = None):
        self.config = config or ContextConfig()

    @staticmethod
    def _utc(value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ContextValidationError(
                "Evidence and conversation timestamps must be timezone-aware."
            )
        return value.astimezone(UTC)

    def validate(
        self,
        output: ContextAnalysisOutput,
        conversation_history: Sequence[ConversationTurn],
        registry: ConversationEvidenceRegistry,
        *,
        actor_id: str,
        now: datetime | None = None,
        current_turn_id: str | None = None,
        current_turn_timestamp: datetime | None = None,
    ) -> ContextAssessment:
        try:
            return self._validate(
                output,
                conversation_history,
                registry,
                actor_id=actor_id,
                now=now,
                current_turn_id=current_turn_id,
                current_turn_timestamp=current_turn_timestamp,
            )
        except ContextValidationError as exc:
            return ContextAssessment(
                status=ContextStatus.INSUFFICIENT,
                signals=("context_validation_failed",),
                reason=f"context_validation_error:{type(exc).__name__}",
                analysis_error=True,
            )

    def _validate(
        self,
        output: ContextAnalysisOutput,
        conversation_history: Sequence[ConversationTurn],
        registry: ConversationEvidenceRegistry,
        *,
        actor_id: str,
        now: datetime | None,
        current_turn_id: str | None,
        current_turn_timestamp: datetime | None,
    ) -> ContextAssessment:
        if not isinstance(output, ContextAnalysisOutput):
            raise ContextValidationError(
                "output must be a ContextAnalysisOutput instance."
            )

        now_utc = self._utc(now or datetime.now(UTC))

        if (current_turn_id is None) != (
            current_turn_timestamp is None
        ):
            raise ContextValidationError(
                "Current turn ID and timestamp must be provided together."
            )

        current_turn_id_normalized = None

        if current_turn_id is not None:
            if not current_turn_id.strip():
                raise ContextValidationError(
                    "Current turn ID must not be empty."
                )

            current_timestamp_utc = self._utc(
                current_turn_timestamp
            )

            if current_timestamp_utc > now_utc:
                raise ContextValidationError(
                    "Current turn timestamp is in the future."
                )

            current_turn_id_normalized = current_turn_id

        turns_by_id = {}
        for turn in conversation_history:
            if turn.turn_id in turns_by_id:
                raise ContextValidationError(
                    f"Duplicate conversation turn ID: {turn.turn_id}"
                )

            if self._utc(turn.timestamp) > now_utc:
                raise ContextValidationError(
                    f"Future conversation timestamp: {turn.turn_id}"
                )

            turns_by_id[turn.turn_id] = turn

        if current_turn_id_normalized is not None:
            if current_turn_id_normalized in turns_by_id:
                raise ContextValidationError(
                    "Current turn ID duplicates a historical turn ID."
                )

            turns_by_id[current_turn_id_normalized] = None

        if not set(output.referenced_turn_ids).issubset(turns_by_id):
            raise ContextValidationError("Unknown referenced turn IDs.")

        if output.status == "independent":
            return ContextAssessment(
                status=ContextStatus.INDEPENDENT,
                signals=tuple(output.signals),
                referenced_turn_ids=tuple(output.referenced_turn_ids),
                reason=output.reason,
            )

        if output.status != "resolved":
            status_map = {
                "insufficient": ContextStatus.INSUFFICIENT,
                "ambiguous": ContextStatus.AMBIGUOUS,
                "conflict": ContextStatus.CONFLICT,
            }
            return ContextAssessment(
                status=status_map[output.status],
                signals=tuple(output.signals),
                referenced_turn_ids=tuple(output.referenced_turn_ids),
                reason=output.reason,
            )

        evidence_items = registry.get_many(
            output.evidence_ids,
            actor_id=actor_id,
        )

        evidence_by_id = {
            item.evidence_id: item for item in evidence_items
        }

        if len(evidence_by_id) != len(evidence_items):
            raise ContextValidationError(
                "Registry returned duplicate evidence identifiers."
            )

        valid_evidence_ids = tuple(
            evidence_id
            for evidence_id in output.evidence_ids
            if evidence_id in evidence_by_id
        )

        valid_evidence = [
            evidence_by_id[evidence_id]
            for evidence_id in valid_evidence_ids
        ]

        evidence_turn_ids = tuple(
            dict.fromkeys(
                item.origin_turn_id for item in valid_evidence
            )
        )

        for item in valid_evidence:
            if item.conversation_id != registry.conversation_id:
                raise ContextValidationError(
                    "Evidence belongs to another conversation."
                )

            if item.origin_turn_id not in turns_by_id:
                raise ContextValidationError(
                    "Evidence references an unknown origin turn."
                )

        assessment_fields = {
            "signals": tuple(output.signals),
            "referenced_turn_ids": tuple(output.referenced_turn_ids),
            "evidence_ids": valid_evidence_ids,
            "evidence_turn_ids": evidence_turn_ids,
        }

        if len(valid_evidence_ids) != len(output.evidence_ids):
            return ContextAssessment(
                status=ContextStatus.INSUFFICIENT,
                **assessment_fields,
                reason="one_or_more_required_evidence_items_not_found",
            )

        actual_types = {item.source_type for item in valid_evidence}
        required_types = set(output.required_evidence_types)

        vision_types = {
            EvidenceSourceType.VISION_DETECTION,
            EvidenceSourceType.VISION_SNAPSHOT,
        }

        non_vision_required = required_types - vision_types
        missing_non_vision_types = non_vision_required - actual_types

        vision_required = bool(required_types & vision_types)
        vision_available = bool(actual_types & vision_types)

        if missing_non_vision_types or (
            vision_required and not vision_available
        ):
            return ContextAssessment(
                status=ContextStatus.INSUFFICIENT,
                **assessment_fields,
                reason="required_evidence_types_not_available",
            )

        ttl = timedelta(seconds=self.config.max_age_seconds)
        expired = False

        for item in valid_evidence:
            created_at = self._utc(item.created_at)

            if created_at > now_utc:
                raise ContextValidationError(
                    f"Future evidence creation timestamp: {item.evidence_id}"
                )

            if now_utc - created_at > ttl:
                expired = True

        if expired:
            return ContextAssessment(
                status=ContextStatus.STALE,
                signals=tuple(output.signals) + ("evidence_expired",),
                referenced_turn_ids=tuple(output.referenced_turn_ids),
                evidence_ids=valid_evidence_ids,
                evidence_turn_ids=evidence_turn_ids,
                resolved_context=None,
                reason="required_context_evidence_exceeded_ttl",
            )

        return ContextAssessment(
            status=ContextStatus.RESOLVED,
            **assessment_fields,
            resolved_context=output.resolved_context,
            reason=output.reason,
        )
