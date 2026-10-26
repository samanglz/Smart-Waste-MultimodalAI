"""Input contracts."""

from typing import Literal

from pydantic import Field, model_validator

from .common import ContractModel, ContractVersion


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

    # Determines whether retrieval can run independently of vision.
    retrieval_query_source: Literal[
        "question",
        "vision_detections",
    ] = "question"

    @model_validator(mode="after")
    def validate_stage_options(self):
        # At least one pipeline stage must be enabled.
        if not (
            self.run_vision
            or self.run_retrieval
            or self.generate_answer
        ):
            raise ValueError(
                "At least one pipeline stage must be enabled."
            )

        # Detection-based retrieval requires the vision stage.
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
    image: ImageArtifact
    question: str | None = Field(default=None, max_length=4000)
    locale: Literal["en", "fa"] = "en"
    options: RequestOptions = Field(default_factory=RequestOptions)

    @model_validator(mode="after")
    def validate_pipeline_options(self):
        # A question-based retrieval query needs a non-empty question.
        if (
            self.options.run_retrieval
            and self.options.retrieval_query_source == "question"
            and not self.question
        ):
            raise ValueError(
                "A non-empty question is required when retrieval "
                "uses the question as its query source."
            )

        return self
