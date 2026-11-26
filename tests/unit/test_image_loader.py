import numpy as np
import pytest

from smart_waste_multimodal.contracts import ImageArtifact
from smart_waste_multimodal.vision.image_loader import (
    ImageLoader,
    validate_loaded_image,
)


def test_valid_bgr_image_is_returned_unchanged():
    image = np.zeros((24, 32, 3), dtype=np.uint8)

    result = validate_loaded_image(image)

    assert result is image
    assert result.shape == (24, 32, 3)
    assert result.dtype == np.uint8


@pytest.mark.parametrize(
    "image, error_type",
    [
        (None, TypeError),
        (np.zeros((10, 10), dtype=np.uint8), ValueError),
        (np.zeros((10, 10, 4), dtype=np.uint8), ValueError),
        (np.zeros((10, 10, 3), dtype=np.float32), ValueError),
        (np.zeros((0, 10, 3), dtype=np.uint8), ValueError),
    ],
)
def test_invalid_loaded_images_are_rejected(image, error_type):
    with pytest.raises(error_type):
        validate_loaded_image(image)


def test_image_loader_protocol_accepts_a_stub():
    class StubImageLoader:
        def load(self, artifact: ImageArtifact) -> np.ndarray:
            assert artifact.artifact_id == "test-image-001"
            return np.zeros((20, 30, 3), dtype=np.uint8)

    artifact = ImageArtifact(
        artifact_id="test-image-001",
        media_type="image/png",
        width=30,
        height=20,
    )

    loader: ImageLoader = StubImageLoader()
    image = loader.load(artifact)

    assert image.shape == (20, 30, 3)
    validate_loaded_image(image)
