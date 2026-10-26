"""Shared contract primitives."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ContractVersion = Literal["1.0"]
ErrorStage = Literal[
    "input", "vision", "retrieval", "generation", "orchestration"
]


class ContractModel(BaseModel):
    """Strict base model for inter-module data."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class ErrorInfo(ContractModel):
    """Safe error information for API responses."""

    code: str = Field(min_length=1, max_length=100)
    stage: ErrorStage
    message: str = Field(min_length=1, max_length=1000)
    retryable: bool = False
