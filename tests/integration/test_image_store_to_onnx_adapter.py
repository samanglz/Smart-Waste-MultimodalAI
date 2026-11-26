"""Integration test for the in-memory image-to-ONNX vision path."""

from io import BytesIO

import numpy as np
from PIL import Image

from smart_waste_multimodal.contracts import AnalyzeRequest, RequestOptions
from smart_waste_multimodal.vision.in_memory_image_loader import (
    InMemoryImageLoader,
)
from smart_waste_multimodal.vision.in_memory_image_store import (
    InMemoryImageStore,
)
from smart_waste_multimodal.vision.onnx_adapter import ONNXVisionAdapter


class FakeInput:
    name = "images"


class RecordingONNXSession:
    """Controlled ONNX session that records its input tensor."""

    def __init__(self):
        self.last_feed = None

    def get_inputs(self):
        return [FakeInput()]

    def run(self, output_names, input_feed):
        self.last_feed = input_feed

        # Original image: 80x40.
        # Letterbox ratio: 8.0.
        # Resized image: 640x320.
        # Vertical padding: 160 pixels on each side.
        #
        # Original box: (10, 5, 30, 25).
        # Model-space box: (80, 200, 240, 360).
        output = np.zeros((1, 9, 1), dtype=np.float32)
        output[0, :4, 0] = [160.0, 280.0, 160.0, 160.0]
        output[0, 4 + 4, 0] = 0.95
        return [output]


def make_png():
    """Create a deterministic 80x40 red PNG in memory."""
    buffer = BytesIO()
    Image.new("RGB", (80, 40), (255, 0, 0)).save(
        buffer,
        format="PNG",
    )
    return buffer.getvalue()


def test_image_store_to_onnx_adapter_end_to_end():
    image_store = InMemoryImageStore()
    image_bytes = make_png()

    artifact = image_store.register(
        artifact_id="integration-image-001",
        image_bytes=image_bytes,
        media_type="image/png",
    )

    assert image_store.count == 1
    assert image_store.contains(artifact.artifact_id)
    assert artifact.width == 80
    assert artifact.height == 40
    assert artifact.sha256 is not None

    image_loader = InMemoryImageLoader(image_store)

    loaded_image = image_loader.load(artifact)

    assert isinstance(loaded_image, np.ndarray)
    assert loaded_image.dtype == np.uint8
    assert loaded_image.shape == (40, 80, 3)

    # The encoded source is RGB red; the Loader must return BGR.
    np.testing.assert_array_equal(
        loaded_image[0, 0],
        np.array([0, 0, 255], dtype=np.uint8),
    )

    session = RecordingONNXSession()

    adapter = ONNXVisionAdapter(
        image_loader=image_loader,
        session=session,
        confidence_threshold=0.25,
        model_version="integration-test-model",
    )

    request = AnalyzeRequest(
        request_id="integration-request-001",
        conversation_id="integration-conversation-001",
        turn_id="integration-turn-001",
        image=artifact,
        question="How should this item be recycled?",
        options=RequestOptions(
            run_vision=True,
            run_retrieval=False,
            generate_answer=False,
        ),
    )

    result = adapter(request)

    assert session.last_feed is not None
    assert set(session.last_feed) == {"images"}

    tensor = session.last_feed["images"]
    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32
    assert tensor.flags.c_contiguous

    # The source red channel must become the first RGB tensor channel.
    np.testing.assert_allclose(
        tensor[0, :, 200, 80],
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        rtol=0,
        atol=1e-6,
    )

    assert result.status == "success"
    assert result.image_width == 80
    assert result.image_height == 40
    assert result.artifact_id == artifact.artifact_id
    assert result.model_version == "integration-test-model"

    assert len(result.detections) == 1
    detection = result.detections[0]

    assert detection.class_name == "plastic"
    assert np.isclose(detection.confidence, 0.95, rtol=0, atol=1e-6)
    np.testing.assert_allclose(
        detection.bbox_xyxy,
        (10.0, 5.0, 30.0, 25.0),
        rtol=0,
        atol=1e-5,
    )

    # Loading and inference must not mutate or remove the stored image.
    assert image_store.count == 1
    assert image_store.contains(artifact.artifact_id)
    assert image_store.get_bytes(artifact.artifact_id) == image_bytes
