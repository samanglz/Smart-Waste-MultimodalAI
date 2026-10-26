"""Answer-generation output contracts."""

from typing import Literal

from pydantic import Field, model_validator

from .common import ContractModel

GenerationStatus = Literal["success", "insufficient_evidence", "skipped"]


class GenerationResult(ContractModel):
    status: GenerationStatus
    answer: str | None = Field(default=None, max_length=20000)
    citation_chunk_ids: list[str] = Field(default_factory=list)
    reason: str | None = Field(default=None, max_length=1000)
    model_version: str | None = Field(default=None, max_length=200)

    @model_validator(mode="after")
    def validate_status(self):
        if len(self.citation_chunk_ids) != len(set(self.citation_chunk_ids)):
            raise ValueError(
                "Citation chunk IDs must not contain duplicates."
            )

        if self.status == "success":
            if not self.answer or not self.answer.strip():
                raise ValueError("success requires a non-empty answer.")
            if self.reason is not None:
                raise ValueError("success must not include a skip reason.")

        elif self.status == "insufficient_evidence":
            if not self.answer or not self.answer.strip():
                raise ValueError(
                    "insufficient_evidence requires an explicit answer."
                )
            if self.citation_chunk_ids:
                raise ValueError(
                    "Insufficient-evidence answers must not cite evidence."
                )
            if not self.reason or not self.reason.strip():
                raise ValueError(
                    "insufficient_evidence requires a reason."
                )

        elif self.status == "skipped":
            if self.answer is not None or self.citation_chunk_ids:
                raise ValueError(
                    "skipped generation must not contain an answer."
                )
            if not self.reason or not self.reason.strip():
                raise ValueError("skipped generation requires a reason.")

        return self
