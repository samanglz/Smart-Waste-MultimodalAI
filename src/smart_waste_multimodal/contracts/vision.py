"""Vision-stage output contracts."""

import math
from typing import Literal

from pydantic import Field, model_validator

from .common import ContractModel


class Detection(ContractModel):
    class_name: str = Field(min_length=1, max_length=100)
    confidence: float = Field(ge=0.0, le=1.0)
    # Pixel coordinates on the original image: x1, y1, x2, y2.
    bbox_xyxy: tuple[float, float, float, float]

    @model_validator(mode="after")
    def validate_box(self):
        x1, y1, x2, y2 = self.bbox_xyxy

        if not all(math.isfinite(value) for value in self.bbox_xyxy):
            raise ValueError("Bounding-box coordinates must be finite.")

        if x1 < 0 or y1 < 0:
            raise ValueError("Bounding-box coordinates must be non-negative.")
        if x1 >= x2 or y1 >= y2:
            raise ValueError("Bounding box must satisfy x1 < x2 and y1 < y2.")
        return self


VisionStatus = Literal["success", "no_detections"]


class VisionResult(ContractModel):
    status: VisionStatus
    detections: list[Detection] = Field(default_factory=list)
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    model_version: str | None = Field(default=None, max_length=200)
    artifact_id: str | None = Field(default=None, max_length=200)

    @model_validator(mode="after")
    def validate_result(self):
        if self.status == "no_detections" and self.detections:
            raise ValueError("no_detections requires an empty detection list.")
        if self.status == "success" and not self.detections:
            raise ValueError("success requires at least one detection.")
        for detection in self.detections:
            _, _, x2, y2 = detection.bbox_xyxy
            if x2 > self.image_width or y2 > self.image_height:
                raise ValueError("Bounding box exceeds image dimensions.")
        return self
