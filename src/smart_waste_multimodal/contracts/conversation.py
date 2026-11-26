"""Conversation and conversation-history contracts."""

from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator, model_validator

from .common import ContractModel
from .vision import VisionResult


class Conversation(ContractModel):
    """Conversation ownership and lifecycle metadata."""

    conversation_id: str = Field(min_length=1, max_length=200)
    owner_id: str = Field(min_length=1, max_length=200)
    status: Literal["active", "closed"] = "active"
    created_at: datetime
    updated_at: datetime
    closed_at: datetime | None = None

    @field_validator("conversation_id", "owner_id")
    @classmethod
    def validate_identifiers(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Conversation identifiers must not be empty.")
        return value

    @field_validator("created_at", "updated_at", "closed_at")
    @classmethod
    def require_timezone_aware_timestamps(cls, value):
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("Conversation timestamps must include a timezone.")
        return value

    @model_validator(mode="after")
    def validate_lifecycle_timestamps(self):
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must not precede created_at.")
        if self.closed_at is not None and self.closed_at < self.created_at:
            raise ValueError("closed_at must not precede created_at.")
        if self.closed_at is not None and self.closed_at > self.updated_at:
            raise ValueError("closed_at must not follow updated_at.")
        if self.status == "active" and self.closed_at is not None:
            raise ValueError("An active conversation cannot have closed_at.")
        if self.status == "closed" and self.closed_at is None:
            raise ValueError("A closed conversation must have closed_at.")
        return self


class ConversationTurn(ContractModel):
    """A conversation turn with optional links to its request and evidence."""

    turn_id: str = Field(min_length=1, max_length=200)
    conversation_id: str | None = Field(default=None, min_length=1, max_length=200)
    request_id: str | None = Field(default=None, min_length=1, max_length=200)
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)
    timestamp: datetime
    vision: VisionResult | None = None
    evidence_ids: tuple[str, ...] = ()
    image_artifact_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
    )

    @field_validator(
        "turn_id",
        "conversation_id",
        "request_id",
        "image_artifact_id",
    )
    @classmethod
    def validate_optional_identifiers(cls, value):
        if value is not None and not value.strip():
            raise ValueError("Conversation turn identifiers must not be empty.")
        return value

    @field_validator("evidence_ids")
    @classmethod
    def validate_evidence_ids(cls, value):
        if any(not item.strip() for item in value):
            raise ValueError("Evidence identifiers must not be empty.")
        if len(value) != len(set(value)):
            raise ValueError("Evidence identifiers must be unique within a turn.")
        return value

    @field_validator("timestamp")
    @classmethod
    def require_timezone_aware_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Conversation timestamps must include a timezone.")
        return value
