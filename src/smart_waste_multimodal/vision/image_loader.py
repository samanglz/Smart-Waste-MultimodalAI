"""Image-loading contracts for resolving image artifact identifiers."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from smart_waste_multimodal.contracts import ImageArtifact


class ImageLoader(Protocol):
    """Resolve an image artifact into an OpenCV-compatible BGR image.

    Implementations own the mapping from artifact IDs to their storage.
    This interface deliberately exposes no filesystem path or URL.
    """

    def load(self, artifact: ImageArtifact) -> np.ndarray:
        """Return a non-empty uint8 BGR image with shape (height, width, 3)."""
        ...


def validate_loaded_image(image: np.ndarray) -> np.ndarray:
    """Validate a loaded image without silently changing its representation."""
    if not isinstance(image, np.ndarray):
        raise TypeError("Loaded image must be a NumPy ndarray.")

    if image.dtype != np.uint8:
        raise ValueError("Loaded image must have dtype uint8.")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "Loaded image must have shape (height, width, 3)."
        )

    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError("Loaded image dimensions must be positive.")

    return image
