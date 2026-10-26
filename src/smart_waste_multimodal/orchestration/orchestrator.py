"""Dependency-injected orchestration for the multimodal pipeline."""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from typing import Any

from smart_waste_multimodal.adapters.interfaces import (
    GenerationStage,
    RetrievalStage,
    VisionStage,
)
from smart_waste_multimodal.contracts import (
    AnalyzeRequest,
    AnalyzeResponse,
    ErrorInfo,
    GenerationResult,
    RetrievalResult,
    VisionResult,
)
from smart_waste_multimodal.observability.logging_config import get_logger

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
    ) -> None:
        self.vision_stage = vision_stage
        self.retrieval_stage = retrieval_stage
        self.generation_stage = generation_stage

    @staticmethod
    def _error(stage: str, exc: Exception) -> ErrorInfo:
        """Create a safe public error without exposing exception details."""
        return ErrorInfo(
            code=f"{stage}_stage_failed",
            stage=stage,
            message=f"The {stage} stage could not complete.",
            retryable=False,
        )

    def run(self, request: AnalyzeRequest) -> AnalyzeResponse:
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
        generation: GenerationResult | None = None
        errors: list[ErrorInfo] = []
        warnings: list[str] = []

        def execute_stage(
            stage_name: str,
            stage_callable: Callable[..., Any],
            *args: Any,
        ) -> tuple[Any | None, ErrorInfo | None]:
            """Execute one stage and log safe timing and outcome metadata."""
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
            except Exception as exc:  # noqa: BLE001 -- isolate stage failures
                duration_ms = (perf_counter() - stage_started) * 1000

                # Do not log exception messages or tracebacks: they may
                # contain paths, prompts, credentials, or other sensitive data.
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

        if options.retrieval_query_source == "vision_detections":
            # Run Vision if requested, even when Retrieval is disabled.
            if options.run_vision:
                result, error = execute_stage(
                    "vision", self.vision_stage, request
                )
                vision = result
                if error is not None:
                    errors.append(error)

            # Only evaluate Retrieval prerequisites when Retrieval is enabled.
            if options.run_retrieval:
                if vision is None:
                    warnings.append(
                        "Retrieval was not run because Vision produced no result."
                    )
                elif not vision.detections:
                    warnings.append(
                        "Retrieval was not run because Vision found no detections."
                    )
                else:
                    result, error = execute_stage(
                        "retrieval", self.retrieval_stage, request, vision
                    )
                    retrieval = result
                    if error is not None:
                        errors.append(error)

        elif options.run_vision and options.run_retrieval:
            # The two stages are independent when Retrieval uses the question.
            # Collect errors in the same deterministic order as before.
            with ThreadPoolExecutor(max_workers=2) as executor:
                vision_future = executor.submit(
                    execute_stage, "vision", self.vision_stage, request
                )
                retrieval_future = executor.submit(
                    execute_stage, "retrieval",
                    self.retrieval_stage, request, None
                )

                vision_result, vision_error = vision_future.result()
                retrieval_result, retrieval_error = retrieval_future.result()

                vision = vision_result
                retrieval = retrieval_result

                if vision_error is not None:
                    errors.append(vision_error)
                if retrieval_error is not None:
                    errors.append(retrieval_error)

        else:
            if options.run_vision:
                result, error = execute_stage(
                    "vision", self.vision_stage, request
                )
                vision = result
                if error is not None:
                    errors.append(error)

            if options.run_retrieval:
                result, error = execute_stage(
                    "retrieval", self.retrieval_stage, request, vision
                )
                retrieval = result
                if error is not None:
                    errors.append(error)

        # Generation receives whichever upstream results are available.
        # It can return an insufficient_evidence result when evidence is absent.
        if options.generate_answer:
            result, error = execute_stage(
                "generation",
                self.generation_stage,
                request,
                vision,
                retrieval,
            )
            generation = result
            if error is not None:
                errors.append(error)

        useful_output = (
            vision is not None
            or retrieval is not None
            or (
                generation is not None
                and generation.status != "skipped"
            )
        )

        if errors or warnings:
            status = "partial_success" if useful_output else "failed"
        else:
            status = "success"

        response = AnalyzeResponse(
            request_id=request.request_id,
            status=status,
            requested_options=options,
            vision=vision,
            retrieval=retrieval,
            generation=generation,
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
