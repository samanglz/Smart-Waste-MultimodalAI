"""Dependency-injected ONNX vision stage adapter."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from smart_waste_multimodal.contracts import (
    AnalyzeRequest,
    VisionResult,
)
from smart_waste_multimodal.vision.image_loader import (
    ImageLoader,
    validate_loaded_image,
)
from smart_waste_multimodal.vision.postprocessing import (
    DetectionPostprocessor,
)
from smart_waste_multimodal.vision.preprocessing import ImagePreprocessor


class ONNXInput(Protocol):
    """Minimal ONNX input metadata required by this adapter."""

    name: str


class ONNXSession(Protocol):
    """Small session interface, compatible with ONNX Runtime sessions."""

    def get_inputs(self) -> list[ONNXInput]: ...

    def run(self, output_names, input_feed: dict[str, np.ndarray]): ...


class ONNXVisionAdapter:
    """Run image loading, preprocessing, ONNX inference and postprocessing.

    The image source and model session are injected. This class does not
    create a session, read model files, or assume a storage location.
    """

    def __init__(
        self,
        *,
        image_loader: ImageLoader,
        session: ONNXSession,
        class_names: tuple[str, ...] = (
            "cardboard",
            "glass",
            "metal",
            "paper",
            "plastic",
        ),
        input_size: int = 640,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detections: int = 300,
        model_version: str | None = None,
    ) -> None:
        inputs = session.get_inputs()
        if len(inputs) != 1:
            raise ValueError(
                "Expected an ONNX model with exactly one input."
            )

        self._image_loader = image_loader
        self._session = session
        self._input_name = inputs[0].name
        self._preprocessor = ImagePreprocessor(input_size)
        self._postprocessor = DetectionPostprocessor(
            class_names=class_names,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
        )
        self._model_version = model_version

    def __call__(self, request: AnalyzeRequest) -> VisionResult:
        """Execute the Vision stage for one pipeline request."""
        image = validate_loaded_image(
            self._image_loader.load(request.image)
        )

        image_height, image_width = image.shape[:2]

        # If dimensions were supplied by the caller, detect mismatches early.
        if (
            request.image.width is not None
            and request.image.width != image_width
        ):
            raise ValueError(
                "Loaded image width does not match artifact metadata."
            )
        if (
            request.image.height is not None
            and request.image.height != image_height
        ):
            raise ValueError(
                "Loaded image height does not match artifact metadata."
            )

        tensor, ratio, pad = self._preprocessor.preprocess(image)

        outputs = self._session.run(
            None,
            {self._input_name: tensor},
        )
        if not outputs:
            raise ValueError("ONNX session returned no outputs.")

        detections = self._postprocessor.process(
            outputs[0],
            ratio=ratio,
            pad=pad,
            image_width=image_width,
            image_height=image_height,
        )

        return VisionResult(
            status="success" if detections else "no_detections",
            detections=detections,
            image_width=image_width,
            image_height=image_height,
            model_version=self._model_version,
            artifact_id=request.image.artifact_id,
        )
