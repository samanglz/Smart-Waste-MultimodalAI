"""Thread-safe in-memory storage for validated image artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from threading import RLock
from typing import Final

from PIL import Image, UnidentifiedImageError

from smart_waste_multimodal.contracts import ImageArtifact

from .image_store import (
    ImageStore,
    ImageStoreSnapshot,
    SupportedImageMediaType,
)
from .image_store_errors import (
    DuplicateImageArtifactError,
    ImageIntegrityError,
    ImageNotFoundError,
    ImageSizeLimitError,
    ImageStoreCapacityError,
    InvalidImageError,
    UnsupportedImageFormatError,
)


MAX_IMAGE_BYTES: Final[int] = 15 * 1024 * 1024
MAX_STORE_BYTES: Final[int] = 1024 * 1024 * 1024
MAX_WIDTH: Final[int] = 8192
MAX_HEIGHT: Final[int] = 8192
MAX_PIXELS: Final[int] = 25_000_000

_FORMAT_TO_MEDIA_TYPE: Final[dict[str, SupportedImageMediaType]] = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
}


@dataclass(frozen=True, slots=True)
class _StoredImage:
    """Internal immutable record."""

    image_bytes: bytes
    metadata: ImageArtifact


class InMemoryImageStore:
    """In-memory ImageStore implementation.

    Encoded image bytes and their metadata are kept together.
    Image validation and decoding happen before atomic insertion.

    Duplicate IDs are rejected while an artifact is stored. This
    implementation does not retain deleted-ID tombstones, so callers
    must never reuse an ID after deletion or clear().
    """

    def __init__(
        self,
        *,
        max_image_bytes: int = MAX_IMAGE_BYTES,
        max_store_bytes: int = MAX_STORE_BYTES,
        max_width: int = MAX_WIDTH,
        max_height: int = MAX_HEIGHT,
        max_pixels: int = MAX_PIXELS,
    ) -> None:
        limits = {
            "max_image_bytes": max_image_bytes,
            "max_store_bytes": max_store_bytes,
            "max_width": max_width,
            "max_height": max_height,
            "max_pixels": max_pixels,
        }

        architecture_limits = {
            "max_image_bytes": MAX_IMAGE_BYTES,
            "max_store_bytes": MAX_STORE_BYTES,
            "max_width": MAX_WIDTH,
            "max_height": MAX_HEIGHT,
            "max_pixels": MAX_PIXELS,
        }

        for name, value in limits.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
            if value > architecture_limits[name]:
                raise ValueError(
                    f"{name} cannot exceed the approved architecture limit "
                    f"of {architecture_limits[name]}."
                )

        if max_image_bytes > max_store_bytes:
            raise ValueError(
                "max_image_bytes cannot exceed max_store_bytes."
            )

        self._max_image_bytes = max_image_bytes
        self._max_store_bytes = max_store_bytes
        self._max_width = max_width
        self._max_height = max_height
        self._max_pixels = max_pixels

        self._records: dict[str, _StoredImage] = {}
        self._total_bytes = 0
        self._lock = RLock()

    def register(
        self,
        artifact_id: str,
        image_bytes: bytes | bytearray | memoryview,
        media_type: SupportedImageMediaType,
    ) -> ImageArtifact:
        """Validate and atomically register an image artifact.

        The caller must provide an ID that has never been used before,
        including IDs of artifacts previously deleted or cleared.
        """

        self._validate_artifact_id(artifact_id)

        if not isinstance(image_bytes, (bytes, bytearray, memoryview)):
            raise TypeError(
                "image_bytes must be bytes, bytearray, or memoryview."
            )

        input_size = (
            image_bytes.nbytes
            if isinstance(image_bytes, memoryview)
            else len(image_bytes)
        )

        if input_size > self._max_image_bytes:
            raise ImageSizeLimitError(
                "Image exceeds the maximum encoded size."
            )

        immutable_bytes = bytes(image_bytes)

        if not immutable_bytes:
            raise InvalidImageError("Image content must not be empty.")

        if media_type not in ("image/jpeg", "image/png"):
            raise UnsupportedImageFormatError(
                f"Unsupported media type: {media_type!r}"
            )

        # Fast duplicate check. The final check is repeated under lock.
        with self._lock:
            if artifact_id in self._records:
                raise DuplicateImageArtifactError(
                    f"Artifact already exists: {artifact_id}"
                )

        width, height = self._validate_image(
            immutable_bytes,
            media_type,
        )

        digest = sha256(immutable_bytes).hexdigest()

        try:
            metadata = ImageArtifact(
                artifact_id=artifact_id,
                media_type=media_type,
                sha256=digest,
                width=width,
                height=height,
            )
        except Exception as exc:
            raise InvalidImageError(
                "Could not construct valid image metadata."
            ) from exc

        record = _StoredImage(
            image_bytes=immutable_bytes,
            metadata=metadata,
        )

        # Atomic duplicate check, capacity check, and insertion.
        with self._lock:
            if artifact_id in self._records:
                raise DuplicateImageArtifactError(
                    f"Artifact already exists: {artifact_id}"
                )

            new_total = self._total_bytes + len(immutable_bytes)

            if new_total > self._max_store_bytes:
                raise ImageStoreCapacityError(
                    "Registering this image would exceed store capacity."
                )

            self._records[artifact_id] = record
            self._total_bytes = new_total

        return metadata.model_copy(deep=True)

    def get_bytes(self, artifact_id: str) -> bytes:
        """Return encoded bytes after integrity verification."""

        snapshot = self.get_image(artifact_id)
        return snapshot.image_bytes

    def get_metadata(self, artifact_id: str) -> ImageArtifact:
        """Return a defensive metadata copy after integrity verification."""

        snapshot = self.get_image(artifact_id)
        return snapshot.metadata

    def get_image(self, artifact_id: str) -> ImageStoreSnapshot:
        """Return an atomic bytes-and-metadata snapshot."""

        self._validate_artifact_id(artifact_id)

        with self._lock:
            record = self._records.get(artifact_id)

            if record is None:
                raise ImageNotFoundError(
                    f"Image artifact not found: {artifact_id}"
                )

            # Immutable bytes and metadata reference are captured together.
            image_bytes = record.image_bytes
            metadata = record.metadata.model_copy(deep=True)

        actual_digest = sha256(image_bytes).hexdigest()

        if metadata.sha256 is None or actual_digest != metadata.sha256:
            raise ImageIntegrityError(
                f"Stored image failed SHA-256 verification: {artifact_id}"
            )

        return ImageStoreSnapshot(
            image_bytes=image_bytes,
            metadata=metadata,
        )

    def contains(self, artifact_id: str) -> bool:
        """Return whether an artifact ID is present."""

        self._validate_artifact_id(artifact_id)

        with self._lock:
            return artifact_id in self._records

    def delete(self, artifact_id: str) -> bool:
        """Delete an artifact and update capacity accounting.

        Deletion does not make the artifact ID available for reuse.
        """

        self._validate_artifact_id(artifact_id)

        with self._lock:
            record = self._records.pop(artifact_id, None)

            if record is None:
                return False

            self._total_bytes -= len(record.image_bytes)
            return True

    def clear(self) -> int:
        """Remove all records without releasing their IDs for reuse."""

        with self._lock:
            removed_count = len(self._records)
            self._records.clear()
            self._total_bytes = 0
            return removed_count

    @property
    def total_bytes(self) -> int:
        """Return total encoded bytes currently stored."""

        with self._lock:
            return self._total_bytes

    @property
    def count(self) -> int:
        """Return the number of stored artifacts."""

        with self._lock:
            return len(self._records)

    @staticmethod
    def _validate_artifact_id(artifact_id: str) -> None:
        if not isinstance(artifact_id, str):
            raise TypeError("artifact_id must be a string.")

        if not artifact_id:
            raise ValueError("artifact_id must not be empty.")

        if artifact_id != artifact_id.strip():
            raise ValueError(
                "artifact_id must not have leading or trailing whitespace."
            )

        if len(artifact_id) > 200:
            raise ValueError(
                "artifact_id must not exceed 200 characters."
            )

    def _validate_image(
        self,
        image_bytes: bytes,
        media_type: SupportedImageMediaType,
    ) -> tuple[int, int]:
        """Validate format, dimensions, frame count, and full decoding."""

        try:
            with Image.open(BytesIO(image_bytes)) as image:
                actual_format = image.format
                actual_media_type = _FORMAT_TO_MEDIA_TYPE.get(
                    actual_format or ""
                )

                if actual_media_type is None:
                    raise UnsupportedImageFormatError(
                        f"Unsupported actual image format: {actual_format!r}"
                    )

                if actual_media_type != media_type:
                    raise UnsupportedImageFormatError(
                        "Declared media type does not match actual image content."
                    )

                if getattr(image, "n_frames", 1) != 1:
                    raise InvalidImageError(
                        "Multi-frame images are not supported."
                    )

                width, height = image.size

                self._validate_dimensions(width, height)

                # verify() checks the encoded image structure.
                image.verify()

            # Reopen because verify() invalidates the decoder state.
            with Image.open(BytesIO(image_bytes)) as decoded:
                if getattr(decoded, "n_frames", 1) != 1:
                    raise InvalidImageError(
                        "Multi-frame images are not supported."
                    )

                if decoded.format != actual_format:
                    raise InvalidImageError(
                        "Image format changed during validation."
                    )

                if decoded.size != (width, height):
                    raise InvalidImageError(
                        "Image dimensions changed during validation."
                    )

                decoded.load()

            return width, height

        except (UnsupportedImageFormatError, InvalidImageError,
                ImageSizeLimitError):
            raise
        except (
            UnidentifiedImageError,
            OSError,
            SyntaxError,
            ValueError,
            EOFError,
        ) as exc:
            raise InvalidImageError(
                "Image content is invalid, incomplete, or cannot be decoded."
            ) from exc

    def _validate_dimensions(self, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            raise InvalidImageError(
                "Image dimensions must be positive."
            )

        if width > self._max_width or height > self._max_height:
            raise ImageSizeLimitError(
                "Image width or height exceeds the configured limit."
            )

        if width * height > self._max_pixels:
            raise ImageSizeLimitError(
                "Image pixel count exceeds the configured limit."
            )
