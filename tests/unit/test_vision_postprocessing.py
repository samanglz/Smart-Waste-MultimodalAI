import numpy as np
import pytest

from smart_waste_multimodal.vision.postprocessing import (
    DetectionPostprocessor,
)

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic"]


def make_output(candidates):
    output = np.zeros(
        (1, 4 + len(CLASSES), len(candidates)),
        dtype=np.float32,
    )
    for i, (cx, cy, w, h, class_id, score) in enumerate(candidates):
        output[0, :4, i] = [cx, cy, w, h]
        output[0, 4 + class_id, i] = score
    return output


def test_nms_suppresses_same_class_but_keeps_other_classes():
    output = make_output([
        (100, 100, 40, 40, 4, 0.95),
        (101, 101, 40, 40, 4, 0.80),
        (100, 100, 40, 40, 1, 0.90),
    ])

    detections = DetectionPostprocessor(CLASSES).process(
        output, 1.0, (0.0, 0.0), 640, 640
    )

    assert len(detections) == 2
    assert {d.class_name for d in detections} == {"plastic", "glass"}
    assert detections[0].bbox_xyxy == (80.0, 80.0, 120.0, 120.0)
    assert detections[0].confidence == pytest.approx(0.95)


def test_boxes_are_clipped_to_image_boundaries():
    output = make_output([(5, 5, 20, 20, 0, 0.9)])

    detections = DetectionPostprocessor(CLASSES).process(
        output, 1.0, (0.0, 0.0), 100, 100
    )

    assert len(detections) == 1
    assert detections[0].bbox_xyxy == (0.0, 0.0, 15.0, 15.0)


def test_below_threshold_prediction_is_ignored():
    output = make_output([(100, 100, 20, 20, 2, 0.1)])

    detections = DetectionPostprocessor(CLASSES).process(
        output, 1.0, (0.0, 0.0), 640, 640
    )

    assert detections == []


@pytest.mark.parametrize(
    "output, ratio, width, height",
    [
        (np.zeros((1, 10, 5), dtype=np.float32), 1.0, 100, 100),
        (np.zeros((1, 9, 1), dtype=np.float32), 0.0, 100, 100),
        (np.zeros((1, 9, 1), dtype=np.float32), 1.0, 0, 100),
        (np.full((1, 9, 1), np.nan, dtype=np.float32), 1.0, 100, 100),
    ],
)
def test_invalid_model_outputs_are_rejected(output, ratio, width, height):
    with pytest.raises(ValueError):
        DetectionPostprocessor(CLASSES).process(
            output, ratio, (0.0, 0.0), width, height
        )


def test_maximum_detection_count_is_enforced():
    output = make_output([
        (50, 50, 10, 10, 0, 0.8),
        (150, 150, 10, 10, 1, 0.9),
    ])

    detections = DetectionPostprocessor(
        CLASSES, max_detections=1
    ).process(output, 1.0, (0.0, 0.0), 640, 640)

    assert len(detections) == 1
    assert detections[0].class_name == "glass"


def test_inverse_letterbox_restores_original_box_coordinates():
    import numpy as np

    from smart_waste_multimodal.vision.postprocessing import (
        DetectionPostprocessor,
    )

    class_names = ("cardboard", "glass", "metal", "paper", "plastic")
    postprocessor = DetectionPostprocessor(class_names=class_names)

    # Original box: (x1, y1, x2, y2) = (1, 0, 4, 2).
    # Forward letterbox transform: ratio=1.6, pad=(0.0, 1.5).
    cx, cy, width, height = 4.0, 3.1, 4.8, 3.2

    output = np.zeros((1, 4 + len(class_names), 1), dtype=np.float32)
    output[0, 0:4, 0] = [cx, cy, width, height]
    output[0, 4, 0] = 0.9

    detections = postprocessor.process(
        output,
        ratio=1.6,
        pad=(0.0, 1.5),
        image_width=5,
        image_height=3,
    )

    assert len(detections) == 1
    assert detections[0].class_name == "cardboard"
    np.testing.assert_allclose(
        detections[0].bbox_xyxy,
        (1.0, 0.0, 4.0, 2.0),
        rtol=0,
        atol=1e-5,
    )


def test_single_candidate_output_preserves_candidate_dimension():
    import numpy as np

    output = np.zeros((1, 9, 1), dtype=np.float32)
    output[0, :, 0] = [
        100.0, 100.0, 40.0, 40.0,
        0.90, 0.05, 0.02, 0.02, 0.01,
    ]

    detections = DetectionPostprocessor(CLASSES).process(
        output,
        ratio=1.0,
        pad=(0.0, 0.0),
        image_width=640,
        image_height=640,
    )

    assert len(detections) == 1
    assert detections[0].class_name == CLASSES[0]
    assert detections[0].confidence == pytest.approx(0.90)


def test_ambiguous_square_output_is_rejected():
    output = np.zeros((9, 9), dtype=np.float32)

    with pytest.raises(ValueError, match="ambiguous|orientation|shape"):
        DetectionPostprocessor(CLASSES).process(
            output,
            ratio=1.0,
            pad=(0.0, 0.0),
            image_width=640,
            image_height=640,
        )


def test_detections_are_sorted_by_confidence():
    output = make_output([
        (50, 50, 10, 10, 0, 0.60),
        (150, 150, 10, 10, 1, 0.95),
        (250, 250, 10, 10, 2, 0.80),
    ])

    detections = DetectionPostprocessor(CLASSES).process(
        output, 1.0, (0.0, 0.0), 640, 640
    )

    assert [d.confidence for d in detections] == pytest.approx(
        [0.95, 0.80, 0.60]
    )
    assert [d.class_name for d in detections] == [
        "glass", "metal", "cardboard"
    ]


def test_prediction_at_confidence_threshold_is_kept():
    output = make_output([
        (100, 100, 20, 20, 0, 0.25),
    ])

    detections = DetectionPostprocessor(
        CLASSES,
        confidence_threshold=0.25,
    ).process(
        output, 1.0, (0.0, 0.0), 640, 640
    )

    assert len(detections) == 1
    assert detections[0].class_name == "cardboard"
    assert detections[0].confidence == pytest.approx(0.25)
