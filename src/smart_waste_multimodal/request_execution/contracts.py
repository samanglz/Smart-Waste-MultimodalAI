"""Contracts for request execution tracking."""

from datetime import UTC, datetime
from typing import Literal

from pydantic import Field, field_validator, model_validator

from smart_waste_multimodal.contracts.common import ContractModel, ErrorInfo
from smart_waste_multimodal.contracts.response import AnalyzeResponse


RequestExecutionStatus = Literal["PROCESSING", "COMPLETED", "FAILED"]


class RequestExecution(ContractModel):
    """Snapshot of one idempotently tracked request execution."""

    actor_id: str = Field(min_length=1, max_length=200)
    request_id: str = Field(min_length=1, max_length=200)
    conversation_id: str | None = Field(default=None, min_length=1, max_length=200)
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: RequestExecutionStatus = "PROCESSING"
    turn_id: str | None = Field(default=None, min_length=1, max_length=200)
    response: AnalyzeResponse | None = None
    safe_error: ErrorInfo | None = None
    internal_error_type: str | None = Field(default=None, min_length=1, max_length=200)
    created_at: datetime
    completed_at: datetime | None = None

    @field_validator("actor_id", "request_id", "conversation_id", "turn_id")
    @classmethod
    def validate_identifiers(cls, value):
        if value is not None and (
            not value or value != value.strip()
        ):
            raise ValueError("Identifiers must be non-empty and canonical.")
        return value

    @field_validator("created_at", "completed_at")
    @classmethod
    def validate_timestamps(cls, value):
        if value is not None and (
            value.tzinfo is None or value.utcoffset() is None
        ):
            raise ValueError("Timestamps must be timezone-aware.")
        return value

    @model_validator(mode="after")
    def validate_state_consistency(self):
        if self.status == "PROCESSING":
            if any((
                self.response is not None,
                self.safe_error is not None,
                self.internal_error_type is not None,
                self.completed_at is not None,
            )):
                raise ValueError(
                    "PROCESSING executions cannot contain completion data."
                )

        elif self.status == "COMPLETED":
            if (
                self.response is None
                or self.safe_error is not None
                or self.internal_error_type is not None
                or self.completed_at is None
            ):
                raise ValueError(
                    "COMPLETED executions require a response and completion time."
                )

            if self.response.request_id != self.request_id:
                raise ValueError(
                    "COMPLETED response request_id must match execution request_id."
                )

        elif self.status == "FAILED":
            if (
                self.safe_error is None
                or self.internal_error_type is None
                or self.completed_at is None
                or self.response is not None
            ):
                raise ValueError(
                    "FAILED executions require safe and internal error details."
                )

        if self.completed_at is not None and self.completed_at < self.created_at:
            raise ValueError("Completion time cannot precede creation time.")

        return self


class RequestReservation(ContractModel):
    """Result of an atomic request reservation attempt."""

    execution: RequestExecution
    is_new: bool
