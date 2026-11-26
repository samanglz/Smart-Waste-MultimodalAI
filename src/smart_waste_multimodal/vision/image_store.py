"""Contracts and data structures for image storage."""

from dataclasses import dataclass
from typing import Literal, Protocol

from smart_waste_multimodal.contracts import ImageArtifact


SupportedImageMediaType = Literal["image/jpeg", "image/png"]


@dataclass(frozen=True, slots=True)
class ImageStoreSnapshot:
    """An immutable snapshot of stored image bytes and metadata."""

    image_bytes: bytes
    metadata: ImageArtifact


class ImageStore(Protocol):
    """Storage interface for encoded image artifacts.

    Artifact IDs are immutable identities. Callers must never reuse an ID,
    even after its artifact has been deleted or the store has been cleared.
    Implementations are not required to retain deleted-ID tombstones;
    uniqueness across the artifact lifecycle is the caller's responsibility.
    """

    def register(
        self,
        artifact_id: str,
        image_bytes: bytes | bytearray | memoryview,
        media_type: SupportedImageMediaType,
    ) -> ImageArtifact:
        """Register an artifact under a never-before-used ID.

        Callers must not reuse an ID after deletion or store clearing.
        """

    def get_bytes(self, artifact_id: str) -> bytes:
        """Return the encoded bytes for an artifact."""

    def get_metadata(self, artifact_id: str) -> ImageArtifact:
        """Return metadata for an artifact."""

    def get_image(self, artifact_id: str) -> ImageStoreSnapshot:
        """Return an atomic snapshot of image bytes and metadata."""

    def contains(self, artifact_id: str) -> bool:
        """Return whether an artifact exists."""

    def delete(self, artifact_id: str) -> bool:
        """Delete an artifact without releasing its ID for reuse.

        Return False if the artifact does not exist.
        """

    def clear(self) -> int:
        """Remove all stored artifacts without releasing their IDs.

        Return the number of artifacts removed.
        """

    @property
    def total_bytes(self) -> int:
        """Return the total number of encoded bytes stored."""

    @property
    def count(self) -> int:
        """Return the number of stored artifacts."""
