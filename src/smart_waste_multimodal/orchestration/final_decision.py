"""Evidence-based final decision engine."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, Protocol

from smart_waste_multimodal.contracts.common import ErrorInfo
from smart_waste_multimodal.contracts.context import (
    ContextAssessment,
    ContextStatus,
)
from smart_waste_multimodal.contracts.evidence import (
    Evidence,
    EvidenceSourceType,
)
from smart_waste_multimodal.contracts.final_decision import (
    DecisionTraceEntry,
    FinalDecisionResult,
)
from smart_waste_multimodal.contracts.request import AnalyzeRequest
from smart_waste_multimodal.contracts.retrieval import RetrievalResult
from smart_waste_multimodal.contracts.vision import VisionResult
from smart_waste_multimodal.evidence.interfaces import ConversationEvidenceRegistry
from smart_waste_multimodal.orchestration.context_recovery import (
    ContextRecoveryResult,
)

RetrievalExecutionStatus = Literal[
    "completed",
    "disabled",
    "skipped_context",
    "skipped_dependency",
    "failed",
]


@dataclass(frozen=True)
class DecisionPolicyInput:
    """Validated evidence and execution facts supplied to a decision policy."""

    request: AnalyzeRequest
    context_assessment: ContextAssessment | None
    retrieval: RetrievalResult | None
    vision: VisionResult | None
    retrieval_execution_status: RetrievalExecutionStatus
    retrieval_evidence_ids: tuple[str, ...]
    vision_evidence_ids: tuple[str, ...]
    context_evidence_ids: tuple[str, ...]
    errors: tuple[ErrorInfo, ...]


@dataclass(frozen=True)
class DecisionAssessment:
    """A policy outcome, not a user-facing response."""

    action: Literal[
        "DIRECT",
        "INSUFFICIENT_EVIDENCE",
        "HUMAN_REVIEW",
    ]
    reason_code: Literal[
        "SUFFICIENT_EVIDENCE",
        "INSUFFICIENT_EVIDENCE",
        "HUMAN_ACTION_REQUIRED",
    ]
    reason: str
    evidence_ids: tuple[str, ...] = ()
    clarification_required: bool = False
    clarification_reason: str | None = None


class DecisionPolicy(Protocol):
    """Injectable policy for deciding whether evidence supports the task."""

    def evaluate(
        self,
        decision_input: DecisionPolicyInput,
    ) -> DecisionAssessment:
        """Assess the complete validated evidence set."""
        ...


class ConservativeDecisionPolicy:
    """Conservative deterministic baseline policy.

    A successful retrieval can support a question-based answer.
    Vision-only sufficiency is accepted for the explicit detection-based
    task mode. Other vision-only cases require a more specific injected
    policy; mere evidence existence is not treated as answer sufficiency.
    """

    def evaluate(
        self,
        decision_input: DecisionPolicyInput,
    ) -> DecisionAssessment:
        request = decision_input.request
        if (
            decision_input.retrieval is not None
            and decision_input.retrieval.status == "success"
            and decision_input.retrieval_evidence_ids
        ):
            relevant_ids = list(decision_input.retrieval_evidence_ids)
            relevant_ids.extend(decision_input.context_evidence_ids)

            return DecisionAssessment(
                action="DIRECT",
                reason_code="SUFFICIENT_EVIDENCE",
                reason=(
                    "Validated retrieval evidence is available for "
                    "the requested question."
                ),
                evidence_ids=tuple(dict.fromkeys(relevant_ids)),
            )

        if (
            request.options.retrieval_query_source == "vision_detections"
            and decision_input.vision is not None
            and decision_input.vision.status == "success"
            and decision_input.vision_evidence_ids
        ):
            relevant_ids = list(decision_input.vision_evidence_ids)
            relevant_ids.extend(decision_input.context_evidence_ids)

            return DecisionAssessment(
                action="DIRECT",
                reason_code="SUFFICIENT_EVIDENCE",
                reason=(
                    "Current-image detection evidence is available "
                    "for the detection-based task."
                ),
                evidence_ids=tuple(dict.fromkeys(relevant_ids)),
            )

        return DecisionAssessment(
            action="INSUFFICIENT_EVIDENCE",
            reason_code="INSUFFICIENT_EVIDENCE",
            reason=(
                "The validated evidence set does not establish "
                "sufficient support for a direct answer."
            ),
        )


class FinalDecisionEngine:
    """Validate request-scoped evidence and produce the final decision."""

    def __init__(
        self,
        policy: DecisionPolicy | None = None,
    ) -> None:
        self.policy = policy or ConservativeDecisionPolicy()

    @staticmethod
    def _unavailable(
        reason: str,
        trace: list[DecisionTraceEntry] | None = None,
    ) -> FinalDecisionResult:
        return FinalDecisionResult(
            execution_status="UNAVAILABLE",
            action=None,
            reason_code="DECISION_DEPENDENCY_UNAVAILABLE",
            reason=reason,
            evidence_ids=[],
            clarification_required=False,
            clarification_reason=None,
            decision_trace=trace or [],
        )

    @staticmethod
    def _trace(
        stage: str,
        outcome: str,
        reason_code: str | None = None,
        evidence_ids: Iterable[str] = (),
    ) -> DecisionTraceEntry:
        return DecisionTraceEntry(
            stage=stage,
            outcome=outcome,
            reason_code=reason_code,
            evidence_ids=list(dict.fromkeys(evidence_ids)),
        )

    @staticmethod
    def _context_assessment(
        context_assessment: ContextAssessment | None,
        context_recovery: ContextRecoveryResult | None,
    ) -> ContextAssessment | None:
        if context_recovery is None:
            return context_assessment

        assessment = context_recovery.assessment
        assessment_is_valid = (
            assessment is not None
            and not assessment.analysis_error
            and assessment.status
            in (ContextStatus.INDEPENDENT, ContextStatus.RESOLVED)
        )

        if context_recovery.resolved and not assessment_is_valid:
            raise ValueError(
                "Context recovery is marked resolved without a valid "
                "independent or resolved assessment."
            )

        if not context_recovery.resolved and assessment_is_valid:
            raise ValueError(
                "Context recovery is marked unresolved but contains "
                "a valid independent or resolved assessment."
            )

        return assessment

    @staticmethod
    def _derive_retrieval_status(
        request: AnalyzeRequest,
        retrieval: RetrievalResult | None,
        context: ContextAssessment | None,
        vision: VisionResult | None,
        errors: tuple[ErrorInfo, ...],
        supplied: RetrievalExecutionStatus | None,
    ) -> RetrievalExecutionStatus:
        if supplied is not None:
            return supplied

        if retrieval is not None:
            return "completed"

        if not request.options.run_retrieval:
            return "disabled"

        if (
            request.options.retrieval_query_source == "question"
            and context is not None
            and (
                context.analysis_error
                or context.status
                not in (ContextStatus.INDEPENDENT, ContextStatus.RESOLVED)
            )
        ):
            return "skipped_context"

        if any(error.stage == "retrieval" for error in errors):
            return "failed"

        if (
            request.options.retrieval_query_source == "vision_detections"
            and (
                vision is None
                or vision.status == "no_detections"
            )
        ):
            return "skipped_dependency"

        return "failed"

    @staticmethod
    def _validate_retrieval_execution_consistency(
        request: AnalyzeRequest,
        retrieval: RetrievalResult | None,
        context: ContextAssessment | None,
        vision: VisionResult | None,
        status: RetrievalExecutionStatus,
    ) -> None:
        retrieval_enabled = request.options.run_retrieval
        query_source = request.options.retrieval_query_source
        context_is_usable = (
            context is not None
            and not context.analysis_error
            and context.status
            in (ContextStatus.INDEPENDENT, ContextStatus.RESOLVED)
        )

        if retrieval is not None and status != "completed":
            raise ValueError(
                "A retrieval result requires completed execution status."
            )

        if retrieval is None and status == "completed":
            raise ValueError(
                "Completed retrieval execution requires a result."
            )

        if not retrieval_enabled:
            if status != "disabled" or retrieval is not None:
                raise ValueError(
                    "Disabled retrieval must have disabled status "
                    "and no result."
                )
            return

        if status == "disabled":
            raise ValueError(
                "Retrieval cannot be disabled when enabled in the request."
            )

        if status == "skipped_context":
            if query_source != "question":
                raise ValueError(
                    "Context-based skipping is only valid for "
                    "question-based retrieval."
                )
            if context_is_usable:
                raise ValueError(
                    "Retrieval cannot be skipped for a usable context."
                )
            return

        if status == "skipped_dependency":
            if query_source != "vision_detections":
                raise ValueError(
                    "Dependency-based skipping is only valid for "
                    "Vision-detection retrieval."
                )
            if vision is not None and vision.status != "no_detections":
                raise ValueError(
                    "Vision-detection retrieval cannot be skipped "
                    "when Vision produced detections."
                )
            return

        if (
            status == "failed"
            and query_source == "question"
            and context is not None
            and not context_is_usable
        ):
            raise ValueError(
                "Unusable question context must be reported as "
                "skipped_context, not failed."
            )

    @staticmethod
    def _validate_registry(
        request: AnalyzeRequest,
        registry: ConversationEvidenceRegistry,
        *,
        actor_id: str,
    ) -> list[Evidence]:
        if registry.conversation_id != request.conversation_id:
            raise ValueError('Evidence registry conversation_id mismatch.')

        records = registry.list_all(actor_id=actor_id)
        if not isinstance(records, list):
            raise TypeError("Evidence registry list_all() must return a list.")

        identifiers: set[str] = set()
        for evidence in records:
            if not isinstance(evidence, Evidence):
                raise TypeError("Evidence registry returned an invalid record.")

            if evidence.conversation_id != request.conversation_id:
                raise ValueError('Evidence belongs to another Conversation.')

            if evidence.evidence_id in identifiers:
                raise ValueError("Evidence registry returned duplicate IDs.")

            identifiers.add(evidence.evidence_id)

        return sorted(records, key=lambda item: item.evidence_id)

    @staticmethod
    def _match_retrieval_evidence(
        retrieval: RetrievalResult | None,
        records: list[Evidence],
    ) -> list[str]:
        if retrieval is None or retrieval.status != "success":
            return []

        matched: list[str] = []
        for item in retrieval.results:
            expected_content = item.model_dump(mode="json")
            candidates = [
                record
                for record in records
                if record.source_type == EvidenceSourceType.RETRIEVAL_CHUNK
                and record.provenance.source_id == item.chunk_id
                and record.provenance.source_version == retrieval.index_version
                and record.content == expected_content
            ]

            if len(candidates) != 1:
                raise ValueError(
                    "A retrieval item does not map to exactly one "
                    "registered evidence record."
                )

            matched.append(candidates[0].evidence_id)

        if len(matched) != len(set(matched)):
            raise ValueError("Retrieval evidence mapping is not one-to-one.")

        return matched

    @staticmethod
    def _match_vision_evidence(
        request: AnalyzeRequest,
        vision: VisionResult | None,
        records: list[Evidence],
    ) -> list[str]:
        if vision is None:
            return []

        request_image_id = getattr(request.image, "artifact_id", None)
        artifact_id = (
            vision.artifact_id
            or request_image_id
            or request.request_id
        )
        expected_snapshot = vision.model_dump(mode="json")

        snapshots = [
            record
            for record in records
            if record.source_type == EvidenceSourceType.VISION_SNAPSHOT
            and record.provenance.source_id == artifact_id
            and record.content == expected_snapshot
        ]

        if len(snapshots) != 1:
            raise ValueError(
                "Current vision result does not map to exactly one "
                "registered snapshot."
            )

        matched = [snapshots[0].evidence_id]

        detections = [
            record
            for record in records
            if record.source_type == EvidenceSourceType.VISION_DETECTION
            and record.provenance.source_id == artifact_id
        ]

        if len(detections) != len(vision.detections):
            raise ValueError(
                "Current vision detections do not match registered evidence."
            )

        by_index: dict[int, Evidence] = {}
        for record in detections:
            raw_index = record.provenance.source_locator.get(
                "detection_index"
            )
            if raw_index is None:
                raise ValueError("Vision detection evidence has no index.")

            index = int(raw_index)
            if index in by_index:
                raise ValueError("Duplicate vision detection evidence index.")

            by_index[index] = record

        if set(by_index) != set(range(len(vision.detections))):
            raise ValueError("Vision detection evidence indices are incomplete.")

        for index, detection in enumerate(vision.detections):
            if by_index[index].content != detection.model_dump(mode="json"):
                raise ValueError(
                    "Vision detection content differs from the stage result."
                )
            matched.append(by_index[index].evidence_id)

        return matched

    @staticmethod
    def _match_context_evidence(
        context: ContextAssessment | None,
        records: list[Evidence],
    ) -> list[str]:
        if context is None or context.status != ContextStatus.RESOLVED:
            return []

        turn_ids = tuple(dict.fromkeys(context.evidence_turn_ids))
        if not turn_ids:
            raise ValueError(
                "Resolved context has no validated evidence turn IDs."
            )

        matched: list[str] = []
        for turn_id in turn_ids:
            turn_records = [
                record
                for record in records
                if record.provenance.source_id == turn_id
                and record.source_type
                in (
                    EvidenceSourceType.CONVERSATION_TEXT,
                    EvidenceSourceType.VISION_SNAPSHOT,
                )
            ]
            if not turn_records:
                raise ValueError(
                    "Resolved context references an unregistered turn."
                )
            matched.extend(record.evidence_id for record in turn_records)

        return sorted(set(matched))

    def decide(
        self,
        *,
        request: AnalyzeRequest,
        evidence_registry: ConversationEvidenceRegistry | None,
        retrieval: RetrievalResult | None = None,
        vision: VisionResult | None = None,
        context_assessment: ContextAssessment | None = None,
        context_recovery: ContextRecoveryResult | None = None,
        errors: Iterable[ErrorInfo] = (),
        retrieval_execution_status: RetrievalExecutionStatus | None = None,
        actor_id: str | None = None,
    ) -> FinalDecisionResult:
        """Produce a final decision using classified execution errors."""

        trace: list[DecisionTraceEntry] = []

        try:
            error_items = tuple(errors)

            # Preserve every reported error, including repeated errors.
            for error in error_items:
                trace.append(
                    self._trace(
                        f"error.{error.stage}",
                        "reported",
                        error.code,
                    )
                )

            # Registry integrity is a decision-critical dependency.
            if evidence_registry is None or any(
                error.stage == "evidence_registry"
                for error in error_items
            ):
                trace.append(
                    self._trace(
                        "evidence_registry",
                        "blocked",
                        "EVIDENCE_REGISTRY_ERROR",
                    )
                )
                trace.append(
                    self._trace(
                        "final_decision",
                        "unavailable",
                        "DECISION_DEPENDENCY_UNAVAILABLE",
                    )
                )
                return self._unavailable(
                    "Evidence registry is unavailable or reported an execution error.",
                    trace,
                )

            context_error_present = any(
                error.stage in ("context_analysis", "context_validation")
                for error in error_items
            )

            context = self._context_assessment(
                context_assessment,
                context_recovery,
            )

            recovery_is_valid = (
                context_recovery is not None
                and context_recovery.resolved
                and context is not None
                and not context.analysis_error
                and context.status
                in (ContextStatus.INDEPENDENT, ContextStatus.RESOLVED)
            )

            # Context errors can only be superseded by a valid recovery.
            context_blocked = (
                context_error_present
                and not recovery_is_valid
                and request.options.retrieval_query_source == "question"
            )

            if context_blocked:
                context = None
                trace.append(
                    self._trace(
                        "context",
                        "blocked",
                        "CONTEXT_ERROR_WITHOUT_VALID_RECOVERY",
                    )
                )

            retrieval_error = any(
                error.stage == "retrieval"
                for error in error_items
            )
            vision_error = any(
                error.stage == "vision"
                for error in error_items
            )

            # Errors invalidate only their own modality.
            effective_retrieval = (
                None if retrieval_error or context_blocked else retrieval
            )
            effective_vision = None if vision_error else vision

            if context_blocked and request.options.run_retrieval:
                status = "skipped_context"
            elif retrieval_error and request.options.run_retrieval:
                status = "failed"
            else:
                status = self._derive_retrieval_status(
                    request,
                    effective_retrieval,
                    context,
                    effective_vision,
                    error_items,
                    retrieval_execution_status,
                )

            self._validate_retrieval_execution_consistency(
                request,
                effective_retrieval,
                context,
                effective_vision,
                status,
            )

            records = self._validate_registry(
                request,
                evidence_registry,
                actor_id=actor_id or evidence_registry.owner_id,
            )
            trace.append(
                self._trace(
                    "evidence_registry",
                    "validated",
                    evidence_ids=(
                        record.evidence_id for record in records
                    ),
                )
            )

            retrieval_ids = self._match_retrieval_evidence(
                effective_retrieval,
                records,
            )
            vision_ids = self._match_vision_evidence(
                request,
                effective_vision,
                records,
            )
            context_ids = self._match_context_evidence(
                context,
                records,
            )

            if retrieval_error:
                trace.append(
                    self._trace(
                        "retrieval_evidence",
                        "invalidated",
                        "RETRIEVAL_ERROR",
                    )
                )
            elif context_blocked:
                trace.append(
                    self._trace(
                        "retrieval_evidence",
                        "not_used",
                        "CONTEXT_NOT_RECOVERED",
                    )
                )
            elif effective_retrieval is not None:
                if effective_retrieval.status == "success":
                    trace.append(
                        self._trace(
                            "retrieval_evidence",
                            "validated",
                            "RETRIEVAL_EVIDENCE_MAPPED",
                            retrieval_ids,
                        )
                    )
                else:
                    trace.append(
                        self._trace(
                            "retrieval_evidence",
                            effective_retrieval.status,
                            "RETRIEVAL_NO_DIRECT_SUPPORT",
                        )
                    )
            else:
                trace.append(
                    self._trace(
                        "retrieval_execution",
                        status,
                        "RETRIEVAL_NOT_AVAILABLE",
                    )
                )

            if vision_error:
                trace.append(
                    self._trace(
                        "vision_evidence",
                        "invalidated",
                        "VISION_ERROR",
                    )
                )
            elif effective_vision is not None:
                trace.append(
                    self._trace(
                        "vision_evidence",
                        "validated",
                        "VISION_EVIDENCE_MAPPED",
                        vision_ids,
                    )
                )

            if context is not None:
                trace.append(
                    self._trace(
                        "context",
                        context.status.value,
                        "CONTEXT_ASSESSMENT_CONSUMED",
                        context_ids,
                    )
                )

            policy_input = DecisionPolicyInput(
                request=request,
                context_assessment=context,
                retrieval=effective_retrieval,
                vision=effective_vision,
                retrieval_execution_status=status,
                retrieval_evidence_ids=tuple(retrieval_ids),
                vision_evidence_ids=tuple(vision_ids),
                context_evidence_ids=tuple(context_ids),
                errors=error_items,
            )

            assessment = self.policy.evaluate(policy_input)
            if not isinstance(assessment, DecisionAssessment):
                raise TypeError(
                    "Decision policy returned an unsupported assessment."
                )

            allowed_ids = set(
                retrieval_ids + vision_ids + context_ids
            )

            if len(assessment.evidence_ids) != len(
                set(assessment.evidence_ids)
            ):
                raise ValueError(
                    "Decision policy returned duplicate evidence IDs."
                )

            if not set(assessment.evidence_ids).issubset(allowed_ids):
                raise ValueError(
                    "Decision policy referenced evidence outside the "
                    "validated relevant evidence set."
                )

            trace.append(
                self._trace(
                    "final_decision",
                    assessment.action,
                    assessment.reason_code,
                    assessment.evidence_ids,
                )
            )

            return FinalDecisionResult(
                execution_status="DECIDED",
                action=assessment.action,
                reason_code=assessment.reason_code,
                reason=assessment.reason,
                evidence_ids=list(assessment.evidence_ids),
                clarification_required=assessment.clarification_required,
                clarification_reason=assessment.clarification_reason,
                decision_trace=trace,
            )

        except Exception as exc:  # noqa: BLE001
            trace.append(
                self._trace(
                    "final_decision",
                    "unavailable",
                    "DECISION_DEPENDENCY_UNAVAILABLE",
                )
            )
            return self._unavailable(
                "Final Decision could not safely validate its dependencies "
                f"({type(exc).__name__}).",
                trace,
            )
