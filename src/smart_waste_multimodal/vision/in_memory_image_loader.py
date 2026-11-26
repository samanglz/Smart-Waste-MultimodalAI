"""In-memory image loading adapter for the vision pipeline."""

from __future__ import annotations

from hashlib import sha256
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from smart_waste_multimodal.contracts import ImageArtifact

from .image_loader import validate_loaded_image
from .image_store import ImageStore
from .image_store_errors import (
    ImageArtifactMismatchError,
    ImageIntegrityError,
    InvalidImageError,
)


class InMemoryImageLoader:
    """Load encoded image artifacts from an ImageStore as BGR arrays.

    The adapter does not register, mutate, or delete stored artifacts.
    It resolves one atomic storage snapshot and validates its metadata
    and content before returning an OpenCV-compatible image.
    """

    def __init__(self, image_store: ImageStore) -> None:
        if image_store is None:
            raise TypeError("image_store must not be None.")

        self._image_store = image_store

    def load(self, artifact: ImageArtifact) -> np.ndarray:
        """Return a non-empty uint8 BGR image for the supplied artifact."""

        if not isinstance(artifact, ImageArtifact):
            raise TypeError("artifact must be an ImageArtifact.")

        snapshot = self._image_store.get_image(artifact.artifact_id)
        stored_metadata = snapshot.metadata

        expected_metadata = (
            artifact.artifact_id,
            artifact.media_type,
            artifact.sha256,
            artifact.width,
            artifact.height,
        )

        actual_metadata = (
            stored_metadata.artifact_id,
            stored_metadata.media_type,
            stored_metadata.sha256,
            stored_metadata.width,
            stored_metadata.height,
        )

        if expected_metadata != actual_metadata:
            raise ImageArtifactMismatchError(
                "Supplied image artifact metadata does not match "
                "the stored artifact metadata."
            )

        actual_digest = sha256(snapshot.image_bytes).hexdigest()

        if (
            stored_metadata.sha256 is None
            or actual_digest != stored_metadata.sha256
        ):
            raise ImageIntegrityError(
                "Image bytes do not match the stored SHA-256 metadata."
            )

        try:
            with Image.open(BytesIO(snapshot.image_bytes)) as encoded_image:
                actual_format = encoded_image.format
                frame_count = getattr(encoded_image, "n_frames", 1)

                expected_format = {
                    "image/jpeg": "JPEG",
                    "image/png": "PNG",
                }.get(stored_metadata.media_type)

                if actual_format != expected_format:
                    raise InvalidImageError(
                        "Encoded image format does not match stored media type."
                    )

                if frame_count != 1:
                    raise InvalidImageError(
                        "Multi-frame images are not supported."
                    )

                encoded_image.load()

                actual_width, actual_height = encoded_image.size

                if (
                    actual_width != stored_metadata.width
                    or actual_height != stored_metadata.height
                ):
                    raise InvalidImageError(
                        "Decoded image dimensions do not match "
                        "stored image metadata."
                    )

                rgb_image = encoded_image.convert("RGB")
                rgb_array = np.asarray(rgb_image, dtype=np.uint8)

            bgr_image = cv2.cvtColor(
                rgb_array,
                cv2.COLOR_RGB2BGR,
            )

        except (
            UnidentifiedImageError,
            OSError,
            SyntaxError,
            ValueError,
            EOFError,
            Image.DecompressionBombError,
            cv2.error,
        ) as exc:
            raise InvalidImageError(
                "Stored image could not be decoded into a BGR image."
            ) from exc

        return validate_loaded_image(bgr_image)
