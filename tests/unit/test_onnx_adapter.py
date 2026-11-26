import numpy as np
import pytest

from smart_waste_multimodal.contracts import (
    AnalyzeRequest,
    ImageArtifact,
    RequestOptions,
)
from smart_waste_multimodal.vision.onnx_adapter import ONNXVisionAdapter


class FakeInput:
    name = "images"


class FakeSession:
    def __init__(self, output):
        self.output = output
        self.last_feed = None

    def get_inputs(self):
        return [FakeInput()]

    def run(self, output_names, input_feed):
        self.last_feed = input_feed
        return [self.output]


class FakeImageLoader:
    def __init__(self, image):
        self.image = image
        self.requested_artifact = None

    def load(self, artifact):
        self.requested_artifact = artifact
        return self.image


def make_request(width=640, height=640):
    return AnalyzeRequest(
        request_id="adapter-test-001",
        conversation_id="test-conversation-test_onnx_adapter-40",
        turn_id="test-turn-test_onnx_adapter-40",
        image=ImageArtifact(
            artifact_id="test-image-001",
            media_type="image/png",
            width=width,
            height=height,
        ),
        question="How should this item be recycled?",
        options=RequestOptions(
            run_vision=True,
            run_retrieval=False,
            generate_answer=False,
        ),
    )


def make_output(class_id=4, confidence=0.95):
    # Shape: (batch=1, channels=4+5, candidates=1)
    output = np.zeros((1, 9, 1), dtype=np.float32)
    output[0, :4, 0] = [320, 320, 100, 100]
    output[0, 4 + class_id, 0] = confidence
    return output


def test_adapter_returns_valid_vision_result():
    image = np.zeros((640, 640, 3), dtype=np.uint8)
    loader = FakeImageLoader(image)
    session = FakeSession(make_output())

    adapter = ONNXVisionAdapter(
        image_loader=loader,
        session=session,
        model_version="test-model-v1",
    )

    result = adapter(make_request())

    assert result.status == "success"
    assert result.image_width == 640
    assert result.image_height == 640
    assert result.artifact_id == "test-image-001"
    assert result.model_version == "test-model-v1"
    assert len(result.detections) == 1
    assert result.detections[0].class_name == "plastic"
    assert result.detections[0].confidence == pytest.approx(0.95)
    assert session.last_feed["images"].shape == (1, 3, 640, 640)
    assert loader.requested_artifact.artifact_id == "test-image-001"


def test_adapter_returns_no_detections_when_below_threshold():
    image = np.zeros((640, 640, 3), dtype=np.uint8)
    adapter = ONNXVisionAdapter(
        image_loader=FakeImageLoader(image),
        session=FakeSession(make_output(confidence=0.1)),
    )

    result = adapter(make_request())

    assert result.status == "no_detections"
    assert result.detections == []


def test_adapter_rejects_mismatched_image_width():
    image = np.zeros((640, 640, 3), dtype=np.uint8)
    adapter = ONNXVisionAdapter(
        image_loader=FakeImageLoader(image),
        session=FakeSession(make_output()),
    )

    with pytest.raises(ValueError, match="width"):
        adapter(make_request(width=320))


def test_adapter_rejects_session_with_multiple_inputs():
    class TwoInputSession(FakeSession):
        def get_inputs(self):
            return [FakeInput(), FakeInput()]

    with pytest.raises(ValueError, match="exactly one input"):
        ONNXVisionAdapter(
            image_loader=FakeImageLoader(
                np.zeros((640, 640, 3), dtype=np.uint8)
            ),
            session=TwoInputSession(make_output()),
        )


def test_adapter_rejects_empty_session_outputs():
    class EmptyOutputSession(FakeSession):
        def run(self, output_names, input_feed):
            return []

    adapter = ONNXVisionAdapter(
        image_loader=FakeImageLoader(
            np.zeros((640, 640, 3), dtype=np.uint8)
        ),
        session=EmptyOutputSession(None),
    )

    with pytest.raises(ValueError, match="no outputs"):
        adapter(make_request())
