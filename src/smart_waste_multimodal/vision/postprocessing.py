"""YOLO-style detection postprocessing for ONNX outputs."""

from __future__ import annotations

import math
from collections.abc import Sequence

import cv2
import numpy as np

from smart_waste_multimodal.contracts import Detection


class DetectionPostprocessor:
    """Decode YOLO output, apply class-aware NMS, and clip boxes."""

    def __init__(
        self,
        class_names: Sequence[str],
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detections: int = 300,
    ) -> None:
        if not class_names:
            raise ValueError("class_names must not be empty.")
        if any(not name.strip() for name in class_names):
            raise ValueError("class_names must not contain empty names.")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1].")
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in [0, 1].")
        if max_detections <= 0:
            raise ValueError("max_detections must be positive.")

        self.class_names = tuple(class_names)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections

    def process(
        self,
        output: np.ndarray,
        ratio: float,
        pad: tuple[float, float],
        image_width: int,
        image_height: int,
    ) -> list[Detection]:
        """Decode YOLOv8/YOLO11-style output shaped (1, 4+C, N).

        Each candidate is expected to contain cx, cy, width, height,
        followed by one score per class. Objectness scores are not
        expected in this output format.
        """
        if not math.isfinite(ratio) or ratio <= 0:
            raise ValueError("ratio must be finite and positive.")
        if image_width <= 0 or image_height <= 0:
            raise ValueError("Image dimensions must be positive.")
        if (
            len(pad) != 2
            or not all(math.isfinite(value) for value in pad)
            or pad[0] < 0
            or pad[1] < 0
        ):
            raise ValueError("pad must contain two finite non-negative values.")

        raw = np.asarray(output)
        if raw.dtype.kind not in "fiu":
            raise ValueError("Model output must contain numeric values.")
        if not np.isfinite(raw).all():
            raise ValueError("Model output contains non-finite values.")

        expected_channels = 4 + len(self.class_names)

        # Remove only the batch dimension, preserving a single candidate.
        if raw.ndim == 3:
            if raw.shape[0] != 1:
                raise ValueError(
                    "Expected batch size 1, "
                    f"got output shape {raw.shape}."
                )
            predictions = raw[0]
        elif raw.ndim == 2:
            predictions = raw
        else:
            raise ValueError(
                "Expected output shaped (1, C, N), (C, N), or (N, C); "
                f"got shape {raw.shape}."
            )

        if (
            predictions.shape[0] == expected_channels
            and predictions.shape[1] == expected_channels
        ):
            raise ValueError(
                "Ambiguous model output orientation: "
                f"both dimensions match {expected_channels} channels "
                f"for shape {predictions.shape}."
            )
        if predictions.shape[0] == expected_channels:
            predictions = predictions.T
        elif predictions.shape[1] != expected_channels:
            raise ValueError(
                "Unexpected YOLO output shape. Expected a channel "
                f"dimension of {expected_channels}, got {raw.shape}."
            )

        pad_x, pad_y = pad
        boxes_by_class: dict[int, list[list[float]]] = {}
        scores_by_class: dict[int, list[float]] = {}

        for pred in predictions:
            class_scores = pred[4:]
            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id])

            if confidence < self.confidence_threshold:
                continue

            cx, cy, box_width, box_height = map(float, pred[:4])
            if box_width <= 0 or box_height <= 0:
                continue

            x1 = (cx - box_width / 2 - pad_x) / ratio
            y1 = (cy - box_height / 2 - pad_y) / ratio
            x2 = (cx + box_width / 2 - pad_x) / ratio
            y2 = (cy + box_height / 2 - pad_y) / ratio

            x1 = max(0.0, min(x1, float(image_width)))
            y1 = max(0.0, min(y1, float(image_height)))
            x2 = max(0.0, min(x2, float(image_width)))
            y2 = max(0.0, min(y2, float(image_height)))

            if x1 >= x2 or y1 >= y2:
                continue

            # OpenCV NMSBoxes expects [x, y, width, height].
            box = [x1, y1, x2 - x1, y2 - y1]
            boxes_by_class.setdefault(class_id, []).append(box)
            scores_by_class.setdefault(class_id, []).append(confidence)

        candidates: list[tuple[float, int, list[float]]] = []

        for class_id, boxes in boxes_by_class.items():
            scores = scores_by_class[class_id]
            # Confidence filtering has already been applied above.
            # Let NMS handle overlap suppression only.
            indices = cv2.dnn.NMSBoxes(
                boxes,
                scores,
                0.0,
                self.iou_threshold,
            )

            if len(indices) == 0:
                continue

            for index in np.asarray(indices).reshape(-1):
                candidates.append(
                    (scores[int(index)], class_id, boxes[int(index)])
                )

        candidates.sort(key=lambda item: item[0], reverse=True)
        detections: list[Detection] = []

        for confidence, class_id, box in candidates[: self.max_detections]:
            x, y, width, height = box
            detections.append(
                Detection(
                    class_name=self.class_names[class_id],
                    confidence=confidence,
                    bbox_xyxy=(
                        float(round(x, 2)),
                        float(round(y, 2)),
                        float(round(x + width, 2)),
                        float(round(y + height, 2)),
                    ),
                )
            )

        return detections
