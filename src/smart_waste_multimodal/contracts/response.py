"""Top-level pipeline response contracts."""

from typing import Literal

from pydantic import Field, model_validator

from .common import ContractModel, ContractVersion, ErrorInfo
from .generation import GenerationResult
from .request import RequestOptions
from .retrieval import RetrievalResult
from .vision import VisionResult

ResponseStatus = Literal["success", "partial_success", "failed"]


class AnalyzeResponse(ContractModel):
    contract_version: ContractVersion = "1.0"
    request_id: str = Field(min_length=1, max_length=200)
    status: ResponseStatus

    # Keep the requested stages with the response so its status can be
    # validated without guessing which stages were intentionally disabled.
    requested_options: RequestOptions = Field(
        default_factory=RequestOptions
    )

    vision: VisionResult | None = None
    retrieval: RetrievalResult | None = None
    generation: GenerationResult | None = None
    errors: list[ErrorInfo] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_response(self):
        options = self.requested_options

        # A disabled stage must not report a normal result. Generation is
        # allowed to explicitly report that it was skipped.
        if not options.run_vision and self.vision is not None:
            raise ValueError(
                "vision must be omitted when run_vision is false."
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

        # Validate that all citations refer to evidence in this response.
        if self.generation and self.generation.citation_chunk_ids:
            if self.retrieval is None:
                raise ValueError("Citations require retrieval results.")

            available = {
                item.chunk_id for item in self.retrieval.results
            }
            unknown = (
                set(self.generation.citation_chunk_ids) - available
            )
            if unknown:
                raise ValueError(
                    "Citations must reference chunk IDs in retrieval results."
                )

        # Duplicate citations add no information and can cause repeated
        # references in downstream renderers.
        if self.generation:
            citation_ids = self.generation.citation_chunk_ids
            if len(citation_ids) != len(set(citation_ids)):
                raise ValueError(
                    "Citation chunk IDs must not contain duplicates."
                )

        # These outcomes are useful stage results, not technical failures:
        # no_detections and no_evidence are normal outcomes. A skipped
        # generation stage, by itself, is not a useful completed result.
        has_useful_output = (
            self.vision is not None
            or self.retrieval is not None
            or (
                self.generation is not None
                and self.generation.status != "skipped"
            )
        )

        required_outputs_complete = (
            (not options.run_vision or self.vision is not None)
            and (not options.run_retrieval or self.retrieval is not None)
            and (
                not options.generate_answer
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
