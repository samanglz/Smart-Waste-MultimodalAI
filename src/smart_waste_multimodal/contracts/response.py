"""Top-level pipeline response contracts."""

from typing import Literal

from pydantic import Field, model_validator

from .common import ContractModel, ContractVersion, ErrorInfo
from .context import ContextAssessment, ContextStatus
from .evidence import EvidenceSourceType
from .final_decision import FinalDecisionResult
from .final_evidence_context import FinalEvidenceContext
from .generation import GenerationResult
from .query_rewrite import QueryRewriteTrace
from .request import RequestOptions
from .retrieval import RetrievalResult
from .vision import VisionResult

ResponseStatus = Literal[
    "success",
    "partial_success",
    "failed",
    "needs_clarification",
]


class AnalyzeResponse(ContractModel):
    contract_version: ContractVersion = "1.0"
    request_id: str = Field(min_length=1, max_length=200)
    status: ResponseStatus

    requested_options: RequestOptions = Field(
        default_factory=RequestOptions
    )
    image_provided: bool = True

    vision: VisionResult | None = None
    retrieval: RetrievalResult | None = None
    generation: GenerationResult | None = None
    context_assessment: ContextAssessment | None = None
    final_decision: FinalDecisionResult | None = None
    final_evidence_context: FinalEvidenceContext | None = None
    query_rewrite_trace: QueryRewriteTrace | None = None

    # Explicit user-facing clarification text, localized by the orchestrator.
    clarification_message: str | None = Field(
        default=None,
        min_length=1,
        max_length=2000,
    )

    errors: list[ErrorInfo] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_response(self):
        options = self.requested_options

        review_required = (
            self.final_decision is not None
            and self.final_decision.execution_status == "DECIDED"
            and self.final_decision.action == "HUMAN_REVIEW"
        )

        if self.final_evidence_context is not None:
            if (
                self.final_decision is None
                or self.final_decision.execution_status != "DECIDED"
                or self.final_decision.action != "DIRECT"
            ):
                raise ValueError(
                    "Final Evidence Context requires a DECIDED/DIRECT decision."
                )
            if self.final_evidence_context.request_id != self.request_id:
                raise ValueError(
                    "Final Evidence Context request_id must match the response."
                )
            if self.final_evidence_context.evidence_ids != tuple(
                self.final_decision.evidence_ids
            ):
                raise ValueError(
                    "Final Evidence Context IDs must match Final Decision evidence IDs."
                )

        if not options.run_vision and self.vision is not None:
            raise ValueError(
                "vision must be omitted when run_vision is false."
            )
        if not self.image_provided and self.vision is not None:
            raise ValueError(
                "vision must be omitted when no image was provided."
            )

        if not options.run_retrieval and self.retrieval is not None:
            raise ValueError(
                "retrieval must be omitted when run_retrieval is false."
            )

        if (
            not options.generate_answer
            and self.generation is not None
            and self.generation.status != "skipped"
        ):
            raise ValueError(
                "Generation must be skipped when generate_answer is false."
            )

        if (
            options.generate_answer
            and self.generation is not None
            and self.generation.status == "skipped"
        ):
            raise ValueError(
                "Requested answer generation cannot be marked skipped."
            )

        if self.generation and self.generation.citation_chunk_ids:
            if self.final_evidence_context is None:
                raise ValueError(
                    "Citations require a validated Final Evidence Context."
                )

            available = {
                item.provenance.source_id
                for item in self.final_evidence_context.evidence
                if (
                    item.source_type == EvidenceSourceType.RETRIEVAL_CHUNK
                    and isinstance(item.content, dict)
                    and item.content.get("chunk_id")
                    == item.provenance.source_id
                )
            }
            unknown = (
                set(self.generation.citation_chunk_ids) - available
            )
            if unknown:
                raise ValueError(
                    "Citations must reference authorized retrieval evidence "
                    "in Final Evidence Context."
                )

        if self.generation:
            citation_ids = self.generation.citation_chunk_ids
            if len(citation_ids) != len(set(citation_ids)):
                raise ValueError(
                    "Citation chunk IDs must not contain duplicates."
                )

        if self.status == "needs_clarification":
            if not review_required:
                raise ValueError(
                    "needs_clarification requires a FinalDecisionResult "
                    "with action HUMAN_REVIEW."
                )
            if not self.clarification_message:
                raise ValueError(
                    "needs_clarification requires a user-facing message."
                )
            if self.clarification_message not in self.warnings:
                raise ValueError(
                    "The clarification message must also appear in warnings."
                )
            if self.generation is not None:
                raise ValueError(
                    "Generation must not run when clarification is required."
                )

        else:
            if review_required:
                raise ValueError(
                    "A HUMAN_REVIEW final decision requires "
                    "needs_clarification response status."
                )
            if self.clarification_message is not None:
                raise ValueError(
                    "clarification_message is only valid for "
                    "needs_clarification responses."
                )

        has_useful_output = (
            (
                self.context_assessment is not None
                and self.context_assessment.status != ContextStatus.INDEPENDENT
            )
            or self.vision is not None
            or self.retrieval is not None
            or self.final_evidence_context is not None
            or (
                self.generation is not None
                and self.generation.status != "skipped"
            )
        )

        required_outputs_complete = (
            (
                not options.run_vision
                or not self.image_provided
                or self.vision is not None
            )
            and (not options.run_retrieval or self.retrieval is not None)
            and (
                not options.generate_answer
                or (
                    self.final_decision is not None
                    and (
                        self.final_decision.execution_status != "DECIDED"
                        or self.final_decision.action != "DIRECT"
                    )
                )
                or (
                    self.generation is not None
                    and self.generation.status != "skipped"
                )
            )
        )

        if self.status == "success":
            if self.errors:
                raise ValueError("success must not contain errors.")
            if self.warnings:
                raise ValueError(
                    "Responses with warnings must use partial_success."
                )
            if not required_outputs_complete:
                raise ValueError(
                    "success requires results for every requested stage."
                )

        elif self.status == "partial_success":
            if not (self.errors or self.warnings):
                raise ValueError(
                    "partial_success requires an error or warning."
                )
            if not has_useful_output:
                raise ValueError(
                    "partial_success requires at least one useful stage result."
                )

        elif self.status == "failed":
            if not self.errors:
                raise ValueError("failed requires at least one error.")
            if has_useful_output:
                raise ValueError(
                    "failed must not contain useful stage results; "
                    "use partial_success when some stages produced results."
                )

        return self
