"""Dependency-injected orchestration for the multimodal pipeline."""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from time import perf_counter
from typing import Any

from smart_waste_multimodal.adapters.interfaces import (
    GenerationStage,
    RetrievalStage,
    VisionStage,
)
from smart_waste_multimodal.config import ContextConfig
from smart_waste_multimodal.contracts import (
    AnalyzeRequest,
    AnalyzeResponse,
    Conversation,
    ErrorInfo,
    GenerationResult,
    QueryRewriteTrace,
    RetrievalResult,
    VisionResult,
)
from smart_waste_multimodal.contracts.context import (
    ContextAssessment,
    ContextStatus,
)
from smart_waste_multimodal.contracts.evidence import (
    Evidence,
    EvidenceProvenance,
    EvidenceSourceType,
)
from smart_waste_multimodal.evidence.final_evidence_context_builder import (
    FinalEvidenceContextBuilder,
)
from smart_waste_multimodal.evidence.interfaces import (
    ConversationEvidenceRegistry,
    ConversationEvidenceRegistryFactory,
)
from smart_waste_multimodal.observability.logging_config import get_logger
from smart_waste_multimodal.orchestration.context_recovery import (
    ContextRecoveryOrchestrator,
    ContextRecoveryResult,
    ContextRecoveryStrategy,
)
from smart_waste_multimodal.orchestration.final_decision import (
    FinalDecisionEngine,
)
from smart_waste_multimodal.query_rewriting.context_analyzer import ContextAnalyzer
from smart_waste_multimodal.query_rewriting.context_validator import (
    ContextAnalysisValidator,
)

logger = get_logger("orchestration")


class PipelineOrchestrator:
    """Run enabled stages and translate stage failures into a response.

    Vision and question-based retrieval are independent and may execute
    concurrently. Detection-based retrieval waits for Vision to finish.
    Stage implementations are injected so they can be mocked in unit tests.
    """

    def __init__(
        self,
        *,
        vision_stage: VisionStage,
        retrieval_stage: RetrievalStage,
        generation_stage: GenerationStage,
        evidence_registry_factory: ConversationEvidenceRegistryFactory,
        context_config: ContextConfig | None = None,
        context_analyzer: ContextAnalyzer | None = None,
        context_validator: ContextAnalysisValidator | None = None,
        context_recovery_strategy: ContextRecoveryStrategy | None = None,
        context_recovery_orchestrator: ContextRecoveryOrchestrator | None = None,
        final_decision_engine: FinalDecisionEngine | None = None,
        final_evidence_context_builder: FinalEvidenceContextBuilder | None = None,
    ) -> None:
        self.vision_stage = vision_stage
        self.retrieval_stage = retrieval_stage
        self.generation_stage = generation_stage
        self.evidence_registry_factory = evidence_registry_factory
        self.final_decision_engine = (
            final_decision_engine
            if final_decision_engine is not None
            else FinalDecisionEngine()
        )
        self.final_evidence_context_builder = (
            final_evidence_context_builder
            if final_evidence_context_builder is not None
            else FinalEvidenceContextBuilder()
        )
        self.context_config = (
            context_config if context_config is not None else ContextConfig()
        )
        self.context_analyzer = (
            context_analyzer
            if context_analyzer is not None
            else ContextAnalyzer(config=self.context_config)
        )
        self.context_validator = (
            context_validator
            if context_validator is not None
            else ContextAnalysisValidator(config=self.context_config)
        )
        self.context_recovery_orchestrator = (
            context_recovery_orchestrator
            if context_recovery_orchestrator is not None
            else ContextRecoveryOrchestrator(
                strategy=context_recovery_strategy,
                validator=self.context_validator,
                max_attempts=self.context_config.max_recovery_attempts,
            )
        )

    def _apply_history_limit(
        self,
        request: AnalyzeRequest,
    ) -> AnalyzeRequest:
        """Return a request containing only the newest allowed turns."""
        history = request.conversation_history
        limit = self.context_config.max_history_turns

        if len(history) <= limit:
            return request

        limited_history = history[-limit:]

        logger.info(
            "conversation_history_truncated request_id=%s "
            "original_turns=%d retained_turns=%d",
            request.request_id,
            len(history),
            len(limited_history),
        )

        return request.model_copy(update={"conversation_history": limited_history})

    @staticmethod
    def _error(stage: str, exc: Exception) -> ErrorInfo:
        """Create a safe public error without exposing exception details."""
        return ErrorInfo(
            code=f"{stage}_stage_failed",
            stage=stage,
            message=f"The {stage} stage could not complete.",
            retryable=False,
        )

    def run(
        self,
        request: AnalyzeRequest,
        *,
        conversation: Conversation,
        actor_id: str | None = None,
    ) -> AnalyzeResponse:
        if request.conversation_id != conversation.conversation_id:
            raise ValueError("Request and Conversation IDs do not match.")
        if conversation.status != "active":
            raise ValueError("Conversation is not active.")
        # Keep the caller's request immutable and use a bounded snapshot.
        request = self._apply_history_limit(request)

        request_started = perf_counter()
        options = request.options

        logger.info(
            "request_started request_id=%s run_vision=%s "
            "run_retrieval=%s generate_answer=%s",
            request.request_id,
            options.run_vision,
            options.run_retrieval,
            options.generate_answer,
        )

        vision: VisionResult | None = None
        retrieval: RetrievalResult | None = None
        query_rewrite_trace: QueryRewriteTrace | None = None
        generation: GenerationResult | None = None
        context_assessment: ContextAssessment | None = None
        recovery_result: ContextRecoveryResult | None = None
        final_decision = None
        final_evidence_context = None

        errors: list[ErrorInfo] = []
        warnings: list[str] = []

        context_blocked = False
        context_analysis_error: ErrorInfo | None = None

        # Evidence belongs to the Conversation and is shared across turns.
        evidence_registry: ConversationEvidenceRegistry | None = None
        evidence_registry_blocked = False
        evidence_created_at = datetime.now(UTC)
        # The application boundary should supply the trusted actor identity.
        # The fallback temporarily preserves compatibility with existing callers.
        actor_id = actor_id or conversation.owner_id

        if actor_id != conversation.owner_id:
            raise PermissionError(
                "Actor does not own the requested Conversation."
            )

        try:
            evidence_registry = self.evidence_registry_factory.create(
                conversation,
                actor_id=actor_id,
            )

            if evidence_registry.conversation_id != conversation.conversation_id:
                raise ValueError("Evidence registry conversation_id mismatch.")

        except Exception as exc:  # noqa: BLE001
            evidence_registry_blocked = True
            errors.append(self._error("evidence_registry", exc))

            logger.warning(
                "evidence_registry_creation_failed request_id=%s exception_type=%s",
                request.request_id,
                type(exc).__name__,
            )

        def register_evidence_batch(items: list[Evidence]) -> None:
            nonlocal evidence_registry_blocked

            if evidence_registry is None or evidence_registry_blocked:
                return

            try:
                for item in items:
                    try:
                        existing = evidence_registry.get(
                            item.evidence_id,
                            actor_id=actor_id,
                        )
                    except KeyError:
                        existing = None

                    if existing is None:
                        evidence_registry.register(
                            item,
                            actor_id=actor_id,
                        )

            except Exception as exc:  # noqa: BLE001
                evidence_registry_blocked = True
                errors.append(self._error("evidence_registry", exc))

                logger.warning(
                    "evidence_registration_failed request_id=%s exception_type=%s",
                    request.request_id,
                    type(exc).__name__,
                )

        # Reuse previously registered evidence whenever a turn provides IDs.
        # Legacy turns without IDs are materialized once for compatibility.
        conversation_evidence: list[Evidence] = []

        for turn in request.conversation_history:
            if (
                turn.conversation_id is not None
                and turn.conversation_id != conversation.conversation_id
            ):
                evidence_registry_blocked = True
                errors.append(
                    self._error(
                        "evidence_registry",
                        ValueError("History turn belongs to another Conversation."),
                    )
                )
                break

            if turn.evidence_ids:
                if evidence_registry is None or evidence_registry_blocked:
                    break

                try:
                    historical_records = evidence_registry.get_many(
                        list(turn.evidence_ids),
                        actor_id=actor_id,
                    )

                    if not isinstance(historical_records, list):
                        raise TypeError("Registry get_many() must return a list.")

                    if len(historical_records) != len(turn.evidence_ids):
                        raise ValueError(
                            "One or more historical evidence records are missing."
                        )

                    for expected_id, record in zip(
                        turn.evidence_ids,
                        historical_records,
                    ):
                        if not isinstance(record, Evidence):
                            raise TypeError(
                                "Registry returned an invalid historical record."
                            )
                        if record.evidence_id != expected_id:
                            raise ValueError(
                                "Historical evidence identity or order mismatch."
                            )
                        if record.conversation_id != conversation.conversation_id:
                            raise ValueError(
                                "Historical evidence belongs to another Conversation."
                            )
                        if record.origin_turn_id != turn.turn_id:
                            raise ValueError(
                                "Historical evidence origin turn mismatch."
                            )

                except Exception as exc:  # noqa: BLE001
                    evidence_registry_blocked = True
                    errors.append(self._error("evidence_registry", exc))
                    logger.warning(
                        "historical_evidence_resolution_failed request_id=%s "
                        "turn_id=%s exception_type=%s",
                        request.request_id,
                        turn.turn_id,
                        type(exc).__name__,
                    )
                    break

                # These records already exist in the shared Conversation registry.
                continue

            # Backward compatibility for history records created before
            # ConversationTurn.evidence_ids was populated.
            origin_request_id = turn.request_id or request.request_id

            conversation_evidence.append(
                Evidence(
                    evidence_id=f"conversation:{turn.turn_id}:text",
                    request_id=origin_request_id,
                    source_type=EvidenceSourceType.CONVERSATION_TEXT,
                    provenance=EvidenceProvenance(
                        source_id=turn.turn_id,
                        source_timestamp=turn.timestamp,
                        source_locator={"role": turn.role},
                    ),
                    content={"text": turn.content},
                    created_at=turn.timestamp,
                    metadata={"role": turn.role},
                    conversation_id=conversation.conversation_id,
                    created_by_request_id=origin_request_id,
                    origin_turn_id=turn.turn_id,
                )
            )

            if turn.vision is not None:
                historical_artifact_id = turn.vision.artifact_id or turn.turn_id

                conversation_evidence.append(
                    Evidence(
                        evidence_id=f"conversation:{turn.turn_id}:vision",
                        request_id=origin_request_id,
                        source_type=EvidenceSourceType.VISION_SNAPSHOT,
                        provenance=EvidenceProvenance(
                            source_id=turn.turn_id,
                            source_timestamp=turn.timestamp,
                            source_version=turn.vision.model_version,
                            source_locator={
                                "artifact_id": historical_artifact_id,
                            },
                        ),
                        content=turn.vision.model_dump(mode="json"),
                        created_at=turn.timestamp,
                        metadata={
                            "origin": "conversation_history",
                            "turn_id": turn.turn_id,
                        },
                        conversation_id=conversation.conversation_id,
                        created_by_request_id=origin_request_id,
                        origin_turn_id=turn.turn_id,
                    )
                )

        register_evidence_batch(conversation_evidence)

        # Build an evidence catalog from the actual registry after legacy
        # evidence has been materialized. Only selected history turns are
        # eligible; retrieval chunks are never conversation evidence.
        context_evidence_catalog: list[dict[str, str]] = []

        if evidence_registry is not None and not evidence_registry_blocked:
            try:
                selected_context_turn_ids = {
                    turn.turn_id
                    for turn in request.conversation_history[
                        -self.context_config.max_history_turns:
                    ]
                }
                catalog_records = evidence_registry.list_all(
                    actor_id=actor_id,
                )

                if not isinstance(catalog_records, list):
                    raise TypeError(
                        "Evidence registry list_all() must return a list."
                    )

                for record in catalog_records:
                    if not isinstance(record, Evidence):
                        raise TypeError(
                            "Evidence registry returned an invalid record."
                        )
                    if record.conversation_id != conversation.conversation_id:
                        raise ValueError(
                            "Evidence catalog contains another Conversation's record."
                        )
                    if record.origin_turn_id not in selected_context_turn_ids:
                        continue
                    if record.source_type is EvidenceSourceType.RETRIEVAL_CHUNK:
                        continue

                    context_evidence_catalog.append(
                        {
                            "evidence_id": record.evidence_id,
                            "origin_turn_id": record.origin_turn_id,
                            "source_type": record.source_type.value,
                        }
                    )

                context_evidence_catalog.sort(
                    key=lambda item: (
                        item["origin_turn_id"],
                        item["evidence_id"],
                    )
                )

            except Exception as exc:  # noqa: BLE001
                evidence_registry_blocked = True
                errors.append(self._error("evidence_registry", exc))
                logger.warning(
                    "context_evidence_catalog_failed request_id=%s "
                    "exception_type=%s",
                    request.request_id,
                    type(exc).__name__,
                )


        def execute_stage(
            stage_name: str,
            stage_callable: Callable[..., Any],
            *args: Any,
        ) -> tuple[Any | None, ErrorInfo | None]:
            stage_started = perf_counter()

            logger.info(
                "stage_started request_id=%s stage=%s",
                request.request_id,
                stage_name,
            )

            try:
                result = stage_callable(*args)

                expected_types = {
                    "vision": VisionResult,
                    "retrieval": RetrievalResult,
                    "generation": GenerationResult,
                }

                expected_type = expected_types[stage_name]

                if not isinstance(result, expected_type):
                    raise TypeError(
                        f"The {stage_name} stage returned an invalid result type."
                    )

            except Exception as exc:  # noqa: BLE001
                duration_ms = (perf_counter() - stage_started) * 1000

                logger.warning(
                    "stage_failed request_id=%s stage=%s "
                    "duration_ms=%.2f exception_type=%s",
                    request.request_id,
                    stage_name,
                    duration_ms,
                    type(exc).__name__,
                )

                return None, self._error(stage_name, exc)

            duration_ms = (perf_counter() - stage_started) * 1000

            logger.info(
                "stage_completed request_id=%s stage=%s duration_ms=%.2f",
                request.request_id,
                stage_name,
                duration_ms,
            )

            return result, None

        def analyze_context() -> tuple[ContextAssessment | None, ErrorInfo | None]:
            started = perf_counter()

            try:
                output = self.context_analyzer.analyze(
                    request.question or "",
                    request.conversation_history,
                    evidence_catalog=context_evidence_catalog,
                )

                if evidence_registry is None or evidence_registry_blocked:
                    assessment = ContextAssessment(
                        status=ContextStatus.INSUFFICIENT,
                        signals=("evidence_registry_unavailable",),
                        reason="Evidence registry is unavailable.",
                        analysis_error=True,
                    )
                    error = self._error(
                        "evidence_registry",
                        RuntimeError(
                            "Context validation requires an available registry."
                        ),
                    )
                    return assessment, error

                assessment = self.context_validator.validate(
                    output,
                    request.conversation_history,
                    evidence_registry,
                    actor_id=actor_id,
                    current_turn_id=request.turn_id,
                    current_turn_timestamp=request.turn_timestamp,
                )

                if assessment.analysis_error:
                    error = self._error(
                        "context_validation",
                        RuntimeError("Context validation failed."),
                    )
                    return assessment, error

                return assessment, None

            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "stage_failed request_id=%s stage=context_analysis "
                    "duration_ms=%.2f exception_type=%s",
                    request.request_id,
                    (perf_counter() - started) * 1000,
                    type(exc).__name__,
                )

                return None, self._error("context_analysis", exc)

        # Context analysis is only relevant to question-based retrieval.
        needs_context = (
            options.run_retrieval and options.retrieval_query_source == "question"
        )

        # Vision and Context Analysis are independent and may run concurrently.
        if options.run_vision and request.image is not None and needs_context:
            with ThreadPoolExecutor(max_workers=2) as executor:
                vision_future = executor.submit(
                    execute_stage,
                    "vision",
                    self.vision_stage,
                    request,
                )
                context_future = executor.submit(analyze_context)

                vision_result, vision_error = vision_future.result()
                assessment, context_error = context_future.result()

            vision = vision_result
            context_assessment = assessment

            if vision_error is not None:
                errors.append(vision_error)

            if context_error is not None:
                context_analysis_error = context_error

        else:
            if options.run_vision and request.image is not None:
                vision_result, vision_error = execute_stage(
                    "vision",
                    self.vision_stage,
                    request,
                )
                vision = vision_result

                if vision_error is not None:
                    errors.append(vision_error)

            if needs_context:
                context_assessment, context_error = analyze_context()

                if context_error is not None:
                    context_analysis_error = context_error

        # Register successful current Vision output before Retrieval.
        if vision is not None and not evidence_registry_blocked:
            image_artifact_id = getattr(
                getattr(request, "image", None),
                "artifact_id",
                None,
            )

            artifact_id = vision.artifact_id or image_artifact_id or request.request_id

            vision_evidence = [
                Evidence(
                    evidence_id=f"vision:{artifact_id}:snapshot",
                    request_id=request.request_id,
                    source_type=EvidenceSourceType.VISION_SNAPSHOT,
                    provenance=EvidenceProvenance(
                        source_id=artifact_id,
                        source_version=vision.model_version,
                        source_locator={"artifact_id": artifact_id},
                    ),
                    content=vision.model_dump(mode="json"),
                    created_at=evidence_created_at,
                    metadata={
                        "image_width": vision.image_width,
                        "image_height": vision.image_height,
                        "detection_count": len(vision.detections),
                    },
                    conversation_id=request.conversation_id,
                    created_by_request_id=request.request_id,
                    origin_turn_id=request.turn_id,
                )
            ]

            for index, detection in enumerate(vision.detections):
                vision_evidence.append(
                    Evidence(
                        evidence_id=(f"vision:{artifact_id}:detection:{index:04d}"),
                        request_id=request.request_id,
                        source_type=EvidenceSourceType.VISION_DETECTION,
                        provenance=EvidenceProvenance(
                            source_id=artifact_id,
                            source_version=vision.model_version,
                            source_locator={
                                "artifact_id": artifact_id,
                                "detection_index": str(index),
                            },
                        ),
                        content=detection.model_dump(mode="json"),
                        created_at=evidence_created_at,
                        metadata={
                            "image_width": vision.image_width,
                            "image_height": vision.image_height,
                            "detection_index": index,
                        },
                        conversation_id=request.conversation_id,
                        created_by_request_id=request.request_id,
                        origin_turn_id=request.turn_id,
                    )
                )

            register_evidence_batch(vision_evidence)

        # Unresolved question context must pass bounded recovery first.
        if needs_context:
            context_is_usable = (
                context_assessment is not None
                and not context_assessment.analysis_error
                and context_assessment.status
                in (
                    ContextStatus.INDEPENDENT,
                    ContextStatus.RESOLVED,
                )
            )

            if not context_is_usable:
                if evidence_registry is None or evidence_registry_blocked:
                    context_blocked = True
                    errors.append(
                        self._error(
                            "context_analysis",
                            RuntimeError(
                                "Context recovery requires an available evidence registry."
                            ),
                        )
                    )
                    warnings.append(
                        "Question-based retrieval and answer generation were "
                        "stopped because the evidence registry is unavailable."
                    )
                else:
                    recovery_result = self.context_recovery_orchestrator.resolve(
                        question=request.question or "",
                        conversation_history=request.conversation_history,
                        initial_assessment=context_assessment,
                        registry=evidence_registry,
                        actor_id=actor_id,
                        current_turn_id=request.turn_id,
                        current_turn_timestamp=request.turn_timestamp,
                        evidence_catalog=context_evidence_catalog,
                    )
                    context_assessment = recovery_result.assessment

                    for record in recovery_result.attempts:
                        logger.info(
                            "context_recovery_attempt request_id=%s "
                            "attempt=%d outcome=%s reason=%s evidence_ids=%s",
                            request.request_id,
                            record.attempt,
                            record.outcome,
                            "reason_redacted",
                            record.evidence_ids,
                        )

                    logger.info(
                        "context_recovery_completed request_id=%s "
                        "resolved=%s attempt_count=%d reason=%s",
                        request.request_id,
                        recovery_result.resolved,
                        len(recovery_result.attempts),
                        "reason_redacted",
                    )

                    if not recovery_result.resolved:
                        context_blocked = True
                        if context_analysis_error is not None:
                            errors.append(context_analysis_error)
                        else:
                            errors.append(
                                self._error(
                                    "context_analysis",
                                    RuntimeError(
                                        "Context recovery did not resolve the context."
                                    ),
                                )
                            )
                        status_label = (
                            context_assessment.status.value
                            if context_assessment is not None
                            else "missing_assessment"
                        )
                        warnings.append(
                            "Question-based retrieval and answer generation were "
                            "stopped because context recovery could not establish "
                            f"a valid context ({status_label})."
                        )

        # Detection-based retrieval deliberately bypasses ContextAnalyzer.
        if options.run_retrieval and not needs_context:
            if options.retrieval_query_source == "vision_detections":
                if request.image is None:
                    errors.append(
                        self._error(
                            "retrieval",
                            RuntimeError(
                                "An image is required for detection-based retrieval."
                            ),
                        )
                    )
                elif vision is None:
                    warnings.append(
                        "Retrieval was not run because Vision produced no result."
                    )
                elif not vision.detections:
                    warnings.append(
                        "Retrieval was not run because Vision found no detections."
                    )
                else:
                    result, error = execute_stage(
                        "retrieval",
                        self.retrieval_stage,
                        request,
                        vision,
                    )
                    retrieval = result

                    if error is not None:
                        errors.append(error)

            else:
                result, error = execute_stage(
                    "retrieval",
                    self.retrieval_stage,
                    request,
                    vision,
                )
                retrieval = result

                if error is not None:
                    errors.append(error)

        elif options.run_retrieval and needs_context and not context_blocked:
            result, error = execute_stage(
                "retrieval",
                self.retrieval_stage,
                request,
                vision,
                context_assessment,
            )
            retrieval = result

            if error is not None:
                errors.append(error)

        # Promote adapter trace to the top-level response contract.
        if retrieval is not None:
            query_rewrite_trace = retrieval.query_rewrite_trace

        if query_rewrite_trace is not None:
            logger.info(
                "query_rewrite_completed request_id=%s action=%s reason=%s "
                "duration_ms=%.2f failure_category=%s failure_type=%s",
                request.request_id,
                query_rewrite_trace.action,
                query_rewrite_trace.reason,
                query_rewrite_trace.duration_ms,
                query_rewrite_trace.failure_category or "none",
                query_rewrite_trace.failure_type or "none",
            )

        # Register each retrieved chunk as an independent evidence item.
        if (
            retrieval is not None
            and retrieval.results
            and not evidence_registry_blocked
        ):
            index_version = retrieval.index_version or "unknown"
            retrieval_evidence: list[Evidence] = []

            for position, item in enumerate(retrieval.results):
                retrieval_evidence.append(
                    Evidence(
                        evidence_id=(
                            f"retrieval:{index_version}:"
                            f"{item.rank}:{position:04d}:{item.chunk_id}"
                        ),
                        request_id=request.request_id,
                        source_type=EvidenceSourceType.RETRIEVAL_CHUNK,
                        provenance=EvidenceProvenance(
                            source_id=item.chunk_id,
                            source_version=retrieval.index_version,
                            source_locator={"source": item.source},
                        ),
                        content=item.model_dump(mode="json"),
                        created_at=evidence_created_at,
                        metadata={
                            "index_version": index_version,
                            "rank": item.rank,
                        },
                        conversation_id=request.conversation_id,
                        created_by_request_id=request.request_id,
                        origin_turn_id=request.turn_id,
                    )
                )

            register_evidence_batch(retrieval_evidence)

        # Final Decision runs after all available evidence is registered.
        final_decision = self.final_decision_engine.decide(
            request=request,
            evidence_registry=(
                None if evidence_registry_blocked else evidence_registry
            ),
            retrieval=retrieval,
            vision=vision,
            context_assessment=context_assessment,
            context_recovery=recovery_result,
            errors=errors,
            actor_id=actor_id,
        )

        if (
            final_decision.execution_status == "DECIDED"
            and final_decision.action == "DIRECT"
        ):
            try:
                if evidence_registry is None or evidence_registry_blocked:
                    raise RuntimeError(
                        "Evidence registry is unavailable for Final Evidence Context."
                    )

                final_evidence_context = self.final_evidence_context_builder.build(
                    request_id=request.request_id,
                    conversation_id=conversation.conversation_id,
                    final_decision=final_decision,
                    evidence_registry=evidence_registry,
                    actor_id=actor_id,
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(self._error("final_evidence_context", exc))

        # Unresolved ContextAssessment blocks generation, but does not itself
        # become a legacy retrieval-level HUMAN_REVIEW result.
        if (
            options.generate_answer
            and not context_blocked
            and not evidence_registry_blocked
            and final_decision.execution_status == "DECIDED"
            and final_decision.action == "DIRECT"
            and final_evidence_context is not None
        ):
            result, error = execute_stage(
                "generation",
                self.generation_stage,
                request,
                final_evidence_context,
            )
            generation = result

            if error is not None:
                errors.append(error)

        useful_output = (
            (
                context_assessment is not None
                and context_assessment.status != ContextStatus.INDEPENDENT
            )
            or vision is not None
            or retrieval is not None
            or final_evidence_context is not None
            or (generation is not None and generation.status != "skipped")
        )

        human_review_required = (
            final_decision.execution_status == "DECIDED"
            and final_decision.action == "HUMAN_REVIEW"
        )
        clarification_message = None

        if human_review_required:
            clarification_message = "I need a little more information to answer safely. Please clarify your request."
            warnings.append(clarification_message)
            status = "needs_clarification"
        elif errors or warnings:
            status = "partial_success" if useful_output else "failed"
        else:
            status = "success"

        response = AnalyzeResponse(
            request_id=request.request_id,
            status=status,
            requested_options=options,
            image_provided=request.image is not None,
            vision=vision,
            retrieval=retrieval,
            generation=generation,
            context_assessment=context_assessment,
            final_decision=final_decision,
            final_evidence_context=final_evidence_context,
            clarification_message=clarification_message,
            query_rewrite_trace=query_rewrite_trace,
            errors=errors,
            warnings=warnings,
        )

        total_duration_ms = (perf_counter() - request_started) * 1000

        logger.info(
            "request_completed request_id=%s status=%s "
            "duration_ms=%.2f error_count=%d warning_count=%d",
            request.request_id,
            response.status,
            total_duration_ms,
            len(errors),
            len(warnings),
        )

        return response
