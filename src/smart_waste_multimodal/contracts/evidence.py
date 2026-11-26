"""Storage-independent evidence contracts."""

from __future__ import annotations

import math
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import ConfigDict, Field, field_validator, model_validator

from .common import ContractModel


class EvidenceSourceType(str, Enum):
    """Origin category of an evidence item."""

    VISION_DETECTION = "vision_detection"
    VISION_SNAPSHOT = "vision_snapshot"
    CONVERSATION_TEXT = "conversation_text"
    RETRIEVAL_CHUNK = "retrieval_chunk"


def _validate_json_value(value: Any, path: str = "value") -> None:
    """Reject values that cannot be represented faithfully as JSON."""

    if value is None or isinstance(value, (str, bool, int)):
        return

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number.")
        return

    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{path}[{index}]")
        return

    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(  # noqa: TRY004
                    f"{path} contains a non-string object key."
                )
            _validate_json_value(item, f"{path}.{key}")
        return

    raise ValueError(
        f"{path} contains a non-JSON-compatible value: {type(value).__name__}"
    )


class EvidenceProvenance(ContractModel):
    """Structured reference back to the original source."""

    source_id: str
    source_timestamp: datetime | None = None
    source_version: str | None = None
    source_locator: dict[str, str] = Field(default_factory=dict)

    @field_validator("source_id", "source_version")
    @classmethod
    def validate_optional_non_empty_strings(cls, value):
        if value is not None and not value.strip():
            raise ValueError("Provenance identifiers must not be empty.")
        return value

    @field_validator("source_timestamp")
    @classmethod
    def validate_source_timestamp(cls, value):
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("source_timestamp must be timezone-aware.")
        return value

    @field_validator("source_locator", mode="before")
    @classmethod
    def validate_locator_shape(cls, value):
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError(  # noqa: TRY004
                "source_locator must be a dictionary."
            )
        if any(
            not isinstance(key, str) or not isinstance(item, str)
            for key, item in value.items()
        ):
            raise ValueError("source_locator keys and values must be strings.")
        return value


class Evidence(ContractModel):
    """Conversation-scoped evidence record, independent of storage technology.

    The request_id field is retained temporarily for compatibility with
    request-scoped consumers. Durable evidence identity and provenance are
    defined by evidence_id, conversation_id, created_by_request_id,
    and origin_turn_id.
    """

    model_config = ConfigDict(frozen=True)

    evidence_id: str
    request_id: str = Field(
        deprecated=True,
        description=(
            "Temporary compatibility field for request-scoped consumers. "
            "Not part of the durable Evidence identity; scheduled for "
            "removal after dependent components are migrated."
        ),
    )
    conversation_id: str
    created_by_request_id: str
    origin_turn_id: str
    schema_version: int = Field(default=2, ge=1, strict=True)
    source_type: EvidenceSourceType
    provenance: EvidenceProvenance
    content: Any
    created_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "evidence_id",
        "request_id",
        "conversation_id",
        "created_by_request_id",
        "origin_turn_id",
    )
    @classmethod
    def validate_identifiers(cls, value):
        if not value.strip():
            raise ValueError("Evidence identifiers must not be empty.")
        return value

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware.")
        return value

    @field_validator("metadata", mode="before")
    @classmethod
    def validate_metadata_shape(cls, value):
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError(  # noqa: TRY004
                "metadata must be a dictionary."
            )
        return value

    @model_validator(mode="after")
    def validate_json_payloads(self):
        _validate_json_value(self.content, "content")
        _validate_json_value(self.metadata, "metadata")
        return self
