import numpy as np
import pytest

from smart_waste_multimodal.vision.preprocessing import ImagePreprocessor


def test_preprocess_returns_normalized_rgb_tensor_and_padding():
    image = np.zeros((2, 4, 3), dtype=np.uint8)
    image[:] = (10, 20, 30)  # BGR

    tensor, ratio, pad = ImagePreprocessor(4).preprocess(image)

    assert tensor.shape == (1, 3, 4, 4)
    assert tensor.dtype == np.float32
    assert tensor.flags.c_contiguous
    assert ratio == 1.0
    assert pad == (0.0, 1.0)
    np.testing.assert_allclose(
        tensor[0, :, 1, 1],
        [30 / 255, 20 / 255, 10 / 255],
    )
    np.testing.assert_allclose(
        tensor[0, :, 0, 0],
        [114 / 255] * 3,
    )


def test_preprocessor_rejects_wrong_shape():
    with pytest.raises(ValueError, match="shape"):
        ImagePreprocessor(8).preprocess(
            np.zeros((10, 10), dtype=np.uint8)
        )


def test_preprocessor_rejects_non_uint8():
    with pytest.raises(ValueError, match="uint8"):
        ImagePreprocessor(8).preprocess(
            np.zeros((10, 10, 3), dtype=np.float32)
        )


def test_preprocessor_rejects_non_positive_size():
    with pytest.raises(ValueError, match="input_size"):
        ImagePreprocessor(0)


def test_rectangular_image_with_uneven_padding():
    import numpy as np

    from smart_waste_multimodal.vision.preprocessing import ImagePreprocessor

    image = np.zeros((3, 5, 3), dtype=np.uint8)
    preprocessor = ImagePreprocessor(input_size=8)

    tensor, ratio, pad = preprocessor.preprocess(image)

    assert tensor.shape == (1, 3, 8, 8)
    assert ratio == 1.6
    assert pad == (0.0, 1.5)

    # One padding row at the top and two at the bottom.
    expected_padding = 114 / 255.0
    np.testing.assert_allclose(
        tensor[0, :, 0, :],
        expected_padding,
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        tensor[0, :, 6:8, :],
        expected_padding,
        rtol=0,
        atol=1e-6,
    )
