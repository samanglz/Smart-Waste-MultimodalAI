"""Input contracts."""

from datetime import UTC, datetime
from itertools import pairwise
from typing import Literal

from pydantic import Field, field_validator, model_validator

from .common import ContractModel, ContractVersion
from .conversation import ConversationTurn


def _utc_now() -> datetime:
    """Return the current timezone-aware UTC time."""
    return datetime.now(UTC)


class ImageArtifact(ContractModel):
    """Reference to an image accepted by an input adapter.

    Does not expose arbitrary URLs or local filesystem paths.
    """

    artifact_id: str = Field(min_length=1, max_length=200)
    media_type: Literal["image/jpeg", "image/png"]
    sha256: str | None = Field(
        default=None, pattern=r"^[a-fA-F0-9]{64}$"
    )
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)


class RequestOptions(ContractModel):
    run_vision: bool = True
    run_retrieval: bool = True
    generate_answer: bool = True

    # Retained for compatibility during the staged migration.
    retrieval_query_source: Literal[
        "question",
        "vision_detections",
    ] = "question"

    @model_validator(mode="after")
    def validate_stage_options(self):
        if not (
            self.run_vision
            or self.run_retrieval
            or self.generate_answer
        ):
            raise ValueError(
                "At least one pipeline stage must be enabled."
            )

        if (
            self.run_retrieval
            and self.retrieval_query_source == "vision_detections"
            and not self.run_vision
        ):
            raise ValueError(
                "Vision must be enabled when retrieval query source "
                "is vision_detections."
            )

        return self


class AnalyzeRequest(ContractModel):
    contract_version: ContractVersion = "1.0"
    request_id: str = Field(min_length=1, max_length=200)

    conversation_id: str = Field(min_length=1, max_length=200)
    turn_id: str = Field(min_length=1, max_length=200)
    turn_timestamp: datetime = Field(default_factory=_utc_now)

    task_type: Literal[
        "detection",
        "knowledge_question",
    ] = "knowledge_question"

    image: ImageArtifact | None = None
    conversation_history: list[ConversationTurn] = Field(
        default_factory=list
    )
    question: str | None = Field(default=None, max_length=4000)
    locale: Literal["en", "fa"] = "en"
    options: RequestOptions = Field(default_factory=RequestOptions)

    @field_validator("turn_timestamp")
    @classmethod
    def validate_turn_timestamp_timezone(cls, value: datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(
                "turn_timestamp must include a timezone."
            )
        return value

    @model_validator(mode="after")
    def validate_pipeline_options(self):
        if (
            self.options.run_retrieval
            and self.options.retrieval_query_source == "question"
            and not self.question
        ):
            raise ValueError(
                "A non-empty question is required when retrieval "
                "uses the question as its query source."
            )

        turn_ids = [
            turn.turn_id for turn in self.conversation_history
        ]

        if len(turn_ids) != len(set(turn_ids)):
            raise ValueError(
                "Conversation turn_id values must be unique."
            )

        timestamps = [
            turn.timestamp.astimezone(UTC)
            for turn in self.conversation_history
        ]

        if any(
            current < previous
            for previous, current in pairwise(timestamps)
        ):
            raise ValueError(
                "conversation_history must be ordered "
                "from oldest to newest."
            )

        request_time = self.turn_timestamp.astimezone(UTC)

        if any(timestamp > request_time for timestamp in timestamps):
            raise ValueError(
                "Conversation timestamps must not be in the future "
                "relative to turn_timestamp."
            )

        return self
