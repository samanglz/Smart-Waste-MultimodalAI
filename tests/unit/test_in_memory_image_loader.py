"""Unit tests for the in-memory image loading adapter."""

from hashlib import sha256
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from smart_waste_multimodal.contracts import ImageArtifact
from smart_waste_multimodal.vision.image_store import ImageStoreSnapshot
from smart_waste_multimodal.vision.image_store_errors import (
    ImageArtifactMismatchError,
    ImageIntegrityError,
    ImageNotFoundError,
    InvalidImageError,
)
from smart_waste_multimodal.vision.in_memory_image_loader import (
    InMemoryImageLoader,
)
from smart_waste_multimodal.vision.in_memory_image_store import (
    InMemoryImageStore,
)


def make_image(
    color=(255, 0, 0),
    image_format="PNG",
    size=(8, 6),
):
    """Create a small deterministic encoded image."""
    buffer = BytesIO()
    image = Image.new("RGB", size, color)
    image.save(buffer, format=image_format)
    return buffer.getvalue()


def make_apng():
    """Create a two-frame PNG image."""
    first = Image.new("RGBA", (8, 6), (255, 0, 0, 255))
    second = Image.new("RGBA", (8, 6), (0, 255, 0, 255))
    buffer = BytesIO()
    first.save(
        buffer,
        format="PNG",
        save_all=True,
        append_images=[second],
        duration=100,
        loop=0,
    )
    return buffer.getvalue()


class SnapshotStore:
    """Minimal controlled Store implementation for adapter error tests."""

    def __init__(self, image_bytes, metadata):
        self.snapshot = ImageStoreSnapshot(
            image_bytes=image_bytes,
            metadata=metadata,
        )

    def get_image(self, artifact_id):
        if artifact_id != self.snapshot.metadata.artifact_id:
            raise ImageNotFoundError(
                f"Image artifact not found: {artifact_id}"
            )
        return self.snapshot


class UncheckedSnapshotStore:
    """Return a controlled snapshot without validating its artifact ID."""

    def __init__(self, image_bytes, metadata):
        self.snapshot = ImageStoreSnapshot(
            image_bytes=image_bytes,
            metadata=metadata,
        )

    def get_image(self, artifact_id):
        return self.snapshot


@pytest.fixture
def store():
    return InMemoryImageStore()


def register_png(store, artifact_id="test-image"):
    image_bytes = make_image()
    artifact = store.register(
        artifact_id=artifact_id,
        image_bytes=image_bytes,
        media_type="image/png",
    )
    return image_bytes, artifact


def test_load_returns_valid_bgr_uint8_array(store):
    _, artifact = register_png(store)

    result = InMemoryImageLoader(store).load(artifact)

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.uint8
    assert result.ndim == 3
    assert result.shape == (6, 8, 3)


def test_load_converts_rgb_to_bgr_correctly(store):
    _, artifact = register_png(store)

    result = InMemoryImageLoader(store).load(artifact)

    # Source pixel is RGB red: (255, 0, 0).
    # OpenCV output must be BGR: (0, 0, 255).
    assert tuple(result[0, 0]) == (0, 0, 255)


def test_load_preserves_image_dimensions(store):
    _, artifact = register_png(store)

    result = InMemoryImageLoader(store).load(artifact)

    assert result.shape[:2] == (artifact.height, artifact.width)


def test_load_rejects_non_image_artifact():
    loader = InMemoryImageLoader(SnapshotStore.__new__(SnapshotStore))

    with pytest.raises(TypeError):
        loader.load("not-an-artifact")


def test_load_rejects_none_store():
    with pytest.raises(TypeError):
        InMemoryImageLoader(None)


def test_load_rejects_metadata_mismatch(store):
    _, artifact = register_png(store)

    mismatched_artifact = artifact.model_copy(
        update={"width": artifact.width + 1}
    )

    loader = InMemoryImageLoader(store)

    with pytest.raises(ImageArtifactMismatchError):
        loader.load(mismatched_artifact)


def test_load_rejects_artifact_id_mismatch():
    image_bytes = make_image()
    digest = sha256(image_bytes).hexdigest()

    metadata = ImageArtifact(
        artifact_id="stored-id",
        media_type="image/png",
        sha256=digest,
        width=8,
        height=6,
    )

    loader = InMemoryImageLoader(
        SnapshotStore(image_bytes, metadata)
    )

    supplied_artifact = metadata.model_copy(
        update={"artifact_id": "different-id"}
    )

    with pytest.raises(ImageNotFoundError):
        loader.load(supplied_artifact)


def test_load_rejects_corrupted_bytes_even_if_store_returns_snapshot():
    original_bytes = make_image()
    corrupted_bytes = bytearray(original_bytes)
    corrupted_bytes[-1] ^= 0x01

    metadata = ImageArtifact(
        artifact_id="corrupted-image",
        media_type="image/png",
        sha256=sha256(original_bytes).hexdigest(),
        width=8,
        height=6,
    )

    loader = InMemoryImageLoader(
        SnapshotStore(bytes(corrupted_bytes), metadata)
    )

    with pytest.raises(ImageIntegrityError):
        loader.load(metadata)


def test_load_rejects_invalid_encoded_image():
    invalid_bytes = b"this-is-not-a-valid-image"

    metadata = ImageArtifact(
        artifact_id="invalid-image",
        media_type="image/png",
        sha256=sha256(invalid_bytes).hexdigest(),
        width=1,
        height=1,
    )

    loader = InMemoryImageLoader(
        SnapshotStore(invalid_bytes, metadata)
    )

    with pytest.raises(InvalidImageError):
        loader.load(metadata)


def test_load_rejects_encoded_format_mismatch():
    jpeg_bytes = make_image(image_format="JPEG")

    metadata = ImageArtifact(
        artifact_id="format-mismatch",
        media_type="image/png",
        sha256=sha256(jpeg_bytes).hexdigest(),
        width=8,
        height=6,
    )

    loader = InMemoryImageLoader(
        SnapshotStore(jpeg_bytes, metadata)
    )

    with pytest.raises(InvalidImageError):
        loader.load(metadata)


def test_load_rejects_multiframe_png():
    apng_bytes = make_apng()

    metadata = ImageArtifact(
        artifact_id="animated-png",
        media_type="image/png",
        sha256=sha256(apng_bytes).hexdigest(),
        width=8,
        height=6,
    )

    loader = InMemoryImageLoader(
        SnapshotStore(apng_bytes, metadata)
    )

    with pytest.raises(InvalidImageError):
        loader.load(metadata)


def test_load_does_not_mutate_store(store):
    _, artifact = register_png(store)
    original_count = store.count
    original_total_bytes = store.total_bytes

    InMemoryImageLoader(store).load(artifact)

    assert store.count == original_count
    assert store.total_bytes == original_total_bytes
    assert store.contains(artifact.artifact_id)


def test_load_propagates_missing_artifact_error(store):
    missing_artifact = ImageArtifact(
        artifact_id="missing-image",
        media_type="image/png",
    )

    loader = InMemoryImageLoader(store)

    with pytest.raises(ImageNotFoundError):
        loader.load(missing_artifact)


def test_load_accepts_jpeg_artifact(store):
    jpeg_bytes = make_image(image_format="JPEG")

    artifact = store.register(
        artifact_id="jpeg-image",
        image_bytes=jpeg_bytes,
        media_type="image/jpeg",
    )

    result = InMemoryImageLoader(store).load(artifact)

    assert result.dtype == np.uint8
    assert result.ndim == 3
    assert result.shape[2] == 3
    assert result.shape[:2] == (artifact.height, artifact.width)


def test_load_rejects_media_type_metadata_mismatch():
    image_bytes = make_image()
    digest = sha256(image_bytes).hexdigest()

    stored_metadata = ImageArtifact(
        artifact_id="media-type-mismatch",
        media_type="image/png",
        sha256=digest,
        width=8,
        height=6,
    )

    supplied_artifact = ImageArtifact(
        artifact_id="media-type-mismatch",
        media_type="image/jpeg",
        sha256=digest,
        width=8,
        height=6,
    )

    loader = InMemoryImageLoader(
        SnapshotStore(image_bytes, stored_metadata)
    )

    with pytest.raises(ImageArtifactMismatchError):
        loader.load(supplied_artifact)


def test_load_rejects_sha256_metadata_mismatch():
    image_bytes = make_image()
    digest = sha256(image_bytes).hexdigest()

    stored_metadata = ImageArtifact(
        artifact_id="sha-metadata-mismatch",
        media_type="image/png",
        sha256=digest,
        width=8,
        height=6,
    )

    supplied_artifact = ImageArtifact(
        artifact_id="sha-metadata-mismatch",
        media_type="image/png",
        sha256="0" * 64,
        width=8,
        height=6,
    )

    loader = InMemoryImageLoader(
        SnapshotStore(image_bytes, stored_metadata)
    )

    with pytest.raises(ImageArtifactMismatchError):
        loader.load(supplied_artifact)


def test_load_rejects_snapshot_artifact_id_mismatch():
    image_bytes = make_image()
    digest = sha256(image_bytes).hexdigest()

    stored_metadata = ImageArtifact(
        artifact_id="stored-image",
        media_type="image/png",
        sha256=digest,
        width=8,
        height=6,
    )

    supplied_artifact = ImageArtifact(
        artifact_id="requested-image",
        media_type="image/png",
        sha256=digest,
        width=8,
        height=6,
    )

    loader = InMemoryImageLoader(
        UncheckedSnapshotStore(image_bytes, stored_metadata)
    )

    with pytest.raises(ImageArtifactMismatchError):
        loader.load(supplied_artifact)


def test_load_rejects_decoded_width_mismatch():
    image_bytes = make_image()
    digest = sha256(image_bytes).hexdigest()

    metadata = ImageArtifact(
        artifact_id="decoded-width-mismatch",
        media_type="image/png",
        sha256=digest,
        width=7,
        height=6,
    )

    loader = InMemoryImageLoader(
        SnapshotStore(image_bytes, metadata)
    )

    with pytest.raises(InvalidImageError):
        loader.load(metadata)


def test_load_rejects_decoded_height_mismatch():
    image_bytes = make_image()
    digest = sha256(image_bytes).hexdigest()

    metadata = ImageArtifact(
        artifact_id="decoded-height-mismatch",
        media_type="image/png",
        sha256=digest,
        width=8,
        height=5,
    )

    loader = InMemoryImageLoader(
        SnapshotStore(image_bytes, metadata)
    )

    with pytest.raises(InvalidImageError):
        loader.load(metadata)
