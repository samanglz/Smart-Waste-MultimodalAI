"""Image preprocessing for YOLO-compatible ONNX models."""

from __future__ import annotations

import math

import cv2
import numpy as np


class ImagePreprocessor:
    """Convert a BGR image into a normalized YOLO input tensor."""

    def __init__(
        self,
        input_size: int = 640,
        padding_value: int = 114,
    ) -> None:
        if input_size <= 0:
            raise ValueError("input_size must be positive.")
        if not 0 <= padding_value <= 255:
            raise ValueError("padding_value must be in [0, 255].")

        self.input_size = input_size
        self.padding_value = padding_value

    def preprocess(
        self,
        image: np.ndarray,
    ) -> tuple[np.ndarray, float, tuple[float, float]]:
        """Return (tensor, resize_ratio, (pad_x, pad_y)).

        The input image must be a non-empty OpenCV BGR uint8 image.
        The tensor shape is (1, 3, input_size, input_size).
        """
        self._validate_image(image)

        original_height, original_width = image.shape[:2]
        ratio = min(
            self.input_size / original_height,
            self.input_size / original_width,
        )

        resized_width = round(original_width * ratio)
        resized_height = round(original_height * ratio)

        resized = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )

        pad_x = (self.input_size - resized_width) / 2
        pad_y = (self.input_size - resized_height) / 2

        left = round(pad_x - 0.1)
        top = round(pad_y - 0.1)
        right = self.input_size - resized_width - left
        bottom = self.input_size - resized_height - top

        letterboxed = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(
                self.padding_value,
                self.padding_value,
                self.padding_value,
            ),
        )

        rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
        tensor = np.ascontiguousarray(tensor, dtype=np.float32)

        expected_shape = (
            1,
            3,
            self.input_size,
            self.input_size,
        )
        if tensor.shape != expected_shape:
            raise RuntimeError(
                f"Unexpected tensor shape: {tensor.shape}; "
                f"expected {expected_shape}."
            )

        return tensor, ratio, (pad_x, pad_y)

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a numpy.ndarray.")
        if image.size == 0:
            raise ValueError("image must not be empty.")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                "image must have shape (height, width, 3) in BGR order."
            )
        if image.dtype != np.uint8:
            raise ValueError("image must have dtype uint8.")
        if image.shape[0] <= 0 or image.shape[1] <= 0:
            raise ValueError("image dimensions must be positive.")


def is_finite_positive(value: float) -> bool:
    """Small utility for validating finite positive numeric values."""
    return math.isfinite(value) and value > 0
