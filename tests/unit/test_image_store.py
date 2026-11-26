
from dataclasses import replace
from hashlib import sha256
from io import BytesIO
from threading import Barrier, Thread

import pytest
from PIL import Image

from smart_waste_multimodal.contracts import ImageArtifact
from smart_waste_multimodal.vision.image_store import ImageStoreSnapshot
from smart_waste_multimodal.vision.image_store_errors import (
    DuplicateImageArtifactError,
    ImageIntegrityError,
    ImageNotFoundError,
    ImageSizeLimitError,
    ImageStoreCapacityError,
    InvalidImageError,
    UnsupportedImageFormatError,
)
from smart_waste_multimodal.vision.in_memory_image_store import (
    MAX_HEIGHT,
    MAX_IMAGE_BYTES,
    MAX_PIXELS,
    MAX_STORE_BYTES,
    MAX_WIDTH,
    InMemoryImageStore,
)


@pytest.fixture
def make_image_bytes():
    def factory(
        width=8,
        height=6,
        image_format="PNG",
        color=(20, 80, 140),
    ):
        buffer = BytesIO()
        image = Image.new("RGB", (width, height), color)
        image.save(buffer, format=image_format)
        return buffer.getvalue()

    return factory


@pytest.fixture
def png_bytes(make_image_bytes):
    return make_image_bytes()


@pytest.fixture
def store():
    return InMemoryImageStore()


def test_register_returns_valid_image_artifact(store, png_bytes):
    artifact = store.register("image-1", png_bytes, "image/png")

    assert isinstance(artifact, ImageArtifact)
    assert artifact.artifact_id == "image-1"
    assert artifact.media_type == "image/png"
    assert artifact.sha256 == sha256(png_bytes).hexdigest()
    assert artifact.width == 8
    assert artifact.height == 6


def test_register_returns_defensive_metadata_copy(store, png_bytes):
    returned = store.register("image-1", png_bytes, "image/png")
    returned.width = 999

    stored = store.get_metadata("image-1")

    assert stored.width == 8


def test_register_accepts_jpeg_image(make_image_bytes):
    image_bytes = make_image_bytes(image_format="JPEG")
    store = InMemoryImageStore()

    artifact = store.register("jpeg-1", image_bytes, "image/jpeg")

    assert artifact.media_type == "image/jpeg"
    assert artifact.width == 8
    assert artifact.height == 6


def test_register_accepts_bytearray_and_memoryview(png_bytes):
    store = InMemoryImageStore()

    first = store.register("bytearray-image", bytearray(png_bytes), "image/png")
    second = store.register("memoryview-image", memoryview(png_bytes), "image/png")

    assert first.sha256 == second.sha256
    assert store.count == 2


def test_register_rejects_empty_content(store):
    with pytest.raises(InvalidImageError):
        store.register("empty", b"", "image/png")

    assert store.count == 0
    assert store.total_bytes == 0


def test_register_rejects_invalid_image_content(store):
    with pytest.raises(InvalidImageError):
        store.register("invalid", b"not-an-image", "image/png")

    assert store.count == 0
    assert store.total_bytes == 0


def test_register_rejects_unsupported_declared_media_type(store, png_bytes):
    with pytest.raises(UnsupportedImageFormatError):
        store.register("unsupported", png_bytes, "image/webp")

    assert store.count == 0


def test_register_rejects_unsupported_actual_format(store):
    buffer = BytesIO()
    Image.new("RGB", (8, 6)).save(buffer, format="GIF")

    with pytest.raises(UnsupportedImageFormatError):
        store.register("gif-image", buffer.getvalue(), "image/png")

    assert store.count == 0


def test_register_rejects_media_type_content_mismatch(store, png_bytes):
    with pytest.raises(UnsupportedImageFormatError):
        store.register("mismatch", png_bytes, "image/jpeg")

    assert store.count == 0


def test_register_rejects_multiframe_apng(store):
    first = Image.new("RGB", (8, 6), (255, 0, 0))
    second = Image.new("RGB", (8, 6), (0, 255, 0))
    buffer = BytesIO()

    first.save(
        buffer,
        format="PNG",
        save_all=True,
        append_images=[second],
        duration=100,
        loop=0,
    )

    with pytest.raises(InvalidImageError, match="Multi-frame"):
        store.register("animated-png", buffer.getvalue(), "image/png")

    assert store.count == 0


def test_register_rejects_image_exceeding_encoded_size_limit(png_bytes):
    store = InMemoryImageStore(
        max_image_bytes=len(png_bytes) - 1,
    )

    with pytest.raises(ImageSizeLimitError):
        store.register("too-large", png_bytes, "image/png")

    assert store.count == 0
    assert store.total_bytes == 0


def test_register_rejects_width_above_limit(make_image_bytes):
    image_bytes = make_image_bytes(width=9, height=6)
    store = InMemoryImageStore(max_width=8)

    with pytest.raises(ImageSizeLimitError):
        store.register("wide-image", image_bytes, "image/png")


def test_register_rejects_height_above_limit(make_image_bytes):
    image_bytes = make_image_bytes(width=8, height=7)
    store = InMemoryImageStore(max_height=6)

    with pytest.raises(ImageSizeLimitError):
        store.register("tall-image", image_bytes, "image/png")


def test_register_rejects_pixel_count_above_limit(make_image_bytes):
    image_bytes = make_image_bytes(width=5, height=5)
    store = InMemoryImageStore(max_pixels=24)

    with pytest.raises(ImageSizeLimitError):
        store.register("too-many-pixels", image_bytes, "image/png")


def test_constructor_rejects_limits_above_architecture():
    with pytest.raises(ValueError):
        InMemoryImageStore(max_image_bytes=MAX_IMAGE_BYTES + 1)

    with pytest.raises(ValueError):
        InMemoryImageStore(max_store_bytes=MAX_STORE_BYTES + 1)

    with pytest.raises(ValueError):
        InMemoryImageStore(max_width=MAX_WIDTH + 1)

    with pytest.raises(ValueError):
        InMemoryImageStore(max_height=MAX_HEIGHT + 1)

    with pytest.raises(ValueError):
        InMemoryImageStore(max_pixels=MAX_PIXELS + 1)


def test_constructor_rejects_invalid_limit_types():
    with pytest.raises(TypeError):
        InMemoryImageStore(max_image_bytes=True)

    with pytest.raises(TypeError):
        InMemoryImageStore(max_width=8.5)


def test_constructor_rejects_image_limit_above_store_limit():
    with pytest.raises(ValueError):
        InMemoryImageStore(
            max_image_bytes=100,
            max_store_bytes=99,
        )


def test_register_rejects_duplicate_artifact_id(store, png_bytes):
    store.register("duplicate", png_bytes, "image/png")

    with pytest.raises(DuplicateImageArtifactError):
        store.register("duplicate", png_bytes, "image/png")

    assert store.count == 1
    assert store.total_bytes == len(png_bytes)


@pytest.mark.parametrize(
    "artifact_id",
    [
        "",
        " leading",
        "trailing ",
        " both ",
        "x" * 201,
    ],
)
def test_register_rejects_invalid_artifact_ids(store, png_bytes, artifact_id):
    with pytest.raises((TypeError, ValueError)):
        store.register(artifact_id, png_bytes, "image/png")

    assert store.count == 0


def test_register_rejects_non_bytes_input(store):
    with pytest.raises(TypeError):
        store.register("wrong-type", "not-bytes", "image/png")


def test_get_image_returns_atomic_snapshot(store, png_bytes):
    store.register("snapshot", png_bytes, "image/png")

    snapshot = store.get_image("snapshot")

    assert isinstance(snapshot, ImageStoreSnapshot)
    assert snapshot.image_bytes == png_bytes
    assert snapshot.metadata.artifact_id == "snapshot"
    assert snapshot.metadata.sha256 == sha256(png_bytes).hexdigest()


def test_get_bytes_returns_original_encoded_bytes(store, png_bytes):
    store.register("bytes", png_bytes, "image/png")

    assert store.get_bytes("bytes") == png_bytes


def test_get_metadata_returns_independent_copy(store, png_bytes):
    store.register("metadata", png_bytes, "image/png")

    first = store.get_metadata("metadata")
    first.width = 999

    second = store.get_metadata("metadata")

    assert second.width == 8


def test_get_image_detects_stored_byte_corruption(store, png_bytes):
    store.register("corrupted", png_bytes, "image/png")

    original = store._records["corrupted"]
    corrupted_bytes = png_bytes + b"corruption"

    store._records["corrupted"] = replace(
        original,
        image_bytes=corrupted_bytes,
    )

    with pytest.raises(ImageIntegrityError):
        store.get_image("corrupted")


def test_get_image_rejects_unknown_id(store):
    with pytest.raises(ImageNotFoundError):
        store.get_image("missing")


def test_get_bytes_rejects_unknown_id(store):
    with pytest.raises(ImageNotFoundError):
        store.get_bytes("missing")


def test_get_metadata_rejects_unknown_id(store):
    with pytest.raises(ImageNotFoundError):
        store.get_metadata("missing")


def test_contains_reports_presence_and_absence(store, png_bytes):
    assert store.contains("image") is False

    store.register("image", png_bytes, "image/png")

    assert store.contains("image") is True
    assert store.contains("other") is False


def test_delete_updates_count_and_capacity(store, png_bytes):
    store.register("delete-me", png_bytes, "image/png")

    assert store.count == 1
    assert store.total_bytes == len(png_bytes)

    assert store.delete("delete-me") is True
    assert store.count == 0
    assert store.total_bytes == 0
    assert store.delete("delete-me") is False


def test_clear_removes_all_records_and_returns_count(png_bytes):
    store = InMemoryImageStore()

    store.register("first", png_bytes, "image/png")
    store.register("second", png_bytes, "image/png")

    removed = store.clear()

    assert removed == 2
    assert store.count == 0
    assert store.total_bytes == 0
    assert store.clear() == 0


def test_capacity_failure_does_not_insert_partial_record(png_bytes):
    image_size = len(png_bytes)
    store = InMemoryImageStore(
        max_image_bytes=image_size,
        max_store_bytes=(2 * image_size) - 1,
    )

    store.register("first", png_bytes, "image/png")

    with pytest.raises(ImageStoreCapacityError):
        store.register("second", png_bytes, "image/png")

    assert store.count == 1
    assert store.total_bytes == image_size
    assert store.contains("first") is True
    assert store.contains("second") is False


def test_failed_validation_does_not_change_store_state(store):
    with pytest.raises(InvalidImageError):
        store.register("invalid", b"broken", "image/png")

    assert store.count == 0
    assert store.total_bytes == 0


def test_total_bytes_tracks_multiple_registrations(png_bytes, make_image_bytes):
    second_bytes = make_image_bytes(width=10, height=10)
    store = InMemoryImageStore()

    store.register("first", png_bytes, "image/png")
    store.register("second", second_bytes, "image/png")

    assert store.count == 2
    assert store.total_bytes == len(png_bytes) + len(second_bytes)

    store.delete("first")

    assert store.count == 1
    assert store.total_bytes == len(second_bytes)


def test_concurrent_duplicate_registration_allows_only_one_success(png_bytes):
    store = InMemoryImageStore()
    barrier = Barrier(2)
    outcomes = []

    def register():
        barrier.wait()
        try:
            store.register("shared-id", png_bytes, "image/png")
            outcomes.append("success")
        except DuplicateImageArtifactError:
            outcomes.append("duplicate")

    threads = [Thread(target=register) for _ in range(2)]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    assert sorted(outcomes) == ["duplicate", "success"]
    assert store.count == 1
    assert store.total_bytes == len(png_bytes)
def test_register_accepts_image_at_exact_store_capacity(png_bytes):
    store = InMemoryImageStore(
        max_image_bytes=len(png_bytes),
        max_store_bytes=len(png_bytes),
    )

    artifact = store.register("exact-capacity", png_bytes, "image/png")

    assert artifact.artifact_id == "exact-capacity"
    assert store.count == 1
    assert store.total_bytes == len(png_bytes)


def test_concurrent_distinct_registrations_respect_store_capacity(png_bytes):
    store = InMemoryImageStore(
        max_image_bytes=len(png_bytes),
        max_store_bytes=len(png_bytes),
    )
    barrier = Barrier(2)
    outcomes = []

    def register(artifact_id):
        barrier.wait()
        try:
            store.register(artifact_id, png_bytes, "image/png")
            outcomes.append("success")
        except ImageStoreCapacityError:
            outcomes.append("capacity")

    threads = [
        Thread(target=register, args=("concurrent-a",)),
        Thread(target=register, args=("concurrent-b",)),
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    assert sorted(outcomes) == ["capacity", "success"]
    assert store.count == 1
    assert store.total_bytes == len(png_bytes)


def test_get_image_detects_metadata_sha256_mismatch(store, png_bytes):
    store.register("metadata-corruption", png_bytes, "image/png")

    original = store._records["metadata-corruption"]
    corrupted_metadata = original.metadata.model_copy(
        update={"sha256": "0" * 64}
    )
    store._records["metadata-corruption"] = replace(
        original,
        metadata=corrupted_metadata,
    )

    with pytest.raises(ImageIntegrityError):
        store.get_image("metadata-corruption")


@pytest.mark.parametrize("artifact_id", [None, 123])
def test_register_rejects_non_string_artifact_ids(
    store,
    png_bytes,
    artifact_id,
):
    with pytest.raises(TypeError):
        store.register(artifact_id, png_bytes, "image/png")

    assert store.count == 0
    assert store.total_bytes == 0
