"""The letterbox is the only coordinate transform in the system.

If it is wrong, every box in the product is wrong by the same amount, and nothing
else in the suite would notice -- the model would still "work", the API would still
return 200, and the overlay would sit slightly off the thing it claims to mark.
"""

from __future__ import annotations

import numpy as np
import pytest

from shrimp_screening.detection.letterbox import PAD_VALUE, LetterboxTransform, letterbox_image


def test_landscape_image_is_padded_top_and_bottom_only() -> None:
    transform = LetterboxTransform.from_size(1000, 500, 640)
    assert transform.scale == pytest.approx(0.64)
    assert (transform.resized_width, transform.resized_height) == (640, 320)
    assert transform.pad_left == 0
    assert transform.pad_top == 160


def test_half_pixel_padding_rounds_down_not_to_even() -> None:
    """The ``-0.1`` in Ultralytics' ``round(dh - 0.1)`` is load-bearing.

    For a 1000x996 source at 640 the ideal pad is exactly 1.5 rows. Python's
    banker's rounding sends ``round(1.5)`` to 2; Ultralytics' ``round(1.5 - 0.1)``
    sends it to 1. Getting this wrong shifts every recovered box by one pixel,
    which is a ten percent error on an eleven-pixel target.
    """
    transform = LetterboxTransform.from_size(1000, 996, 640)
    assert transform.resized_height == 637
    assert (640 - 637) / 2 == 1.5
    assert round(1.5) == 2, "the tie-break this test exists to catch"
    assert transform.pad_top == 1


def test_square_image_needs_no_padding() -> None:
    transform = LetterboxTransform.from_size(2048, 2048, 640)
    assert (transform.pad_left, transform.pad_top) == (0, 0)
    assert (transform.resized_width, transform.resized_height) == (640, 640)


@pytest.mark.parametrize(
    ("width", "height"),
    [(2048, 2048), (1000, 500), (500, 1000), (641, 480), (1080, 810), (3000, 1000)],
)
def test_forward_and_inverse_round_trip_within_one_pixel(width: int, height: int) -> None:
    """A box mapped into the letterbox frame and back must land where it started.

    This proves internal consistency only. It **cannot** catch a wrong assumption
    about what Ultralytics actually emits -- for that, the loaded model's own
    metadata is asserted in ``onnx_provider.py``. Nobody should read a green here
    as conformance to an external implementation.
    """
    rng = np.random.default_rng(20260730)
    transform = LetterboxTransform.from_size(width, height, 640)
    for _ in range(50):
        x1, x2 = sorted(rng.uniform(0, width, size=2))
        y1, y2 = sorted(rng.uniform(0, height, size=2))
        mapped = transform.forward_xyxy((x1, y1, x2, y2))
        recovered = transform.inverse_xyxy_pixels(np.array([mapped]))[0]
        assert recovered == pytest.approx([x1, y1, x2, y2], abs=1.0)


def test_letterbox_image_produces_a_normalized_nchw_batch() -> None:
    image = np.full((500, 1000, 3), 200, dtype=np.uint8)
    batch, transform = letterbox_image(image, 640)

    assert batch.shape == (1, 3, 640, 640)
    assert batch.dtype == np.float32
    assert batch.flags["C_CONTIGUOUS"]
    assert batch.min() >= 0.0
    assert batch.max() <= 1.0
    # The padded band carries Ultralytics' fill colour, scaled by 255 and nothing else.
    assert batch[0, :, 0, 0] == pytest.approx(PAD_VALUE / 255.0)
    # The image band carries the source pixels, also only scaled.
    assert batch[0, :, transform.pad_top + 1, 5] == pytest.approx(200 / 255.0)


def test_letterbox_preserves_channel_order() -> None:
    """RGB in, RGB out. A silent BGR swap would poison every colour-sensitive metric."""
    image = np.zeros((640, 640, 3), dtype=np.uint8)
    image[:, :, 0] = 255
    batch, _ = letterbox_image(image, 640)
    assert batch[0, 0].max() == pytest.approx(1.0)
    assert batch[0, 1].max() == pytest.approx(0.0)
    assert batch[0, 2].max() == pytest.approx(0.0)


def test_normalized_inverse_clips_boxes_that_extend_into_the_padding() -> None:
    transform = LetterboxTransform.from_size(1000, 500, 640)
    # A box entirely inside the top padding band maps above the photograph.
    boxes = np.array([[0.0, 0.0, 640.0, 100.0]])
    normalized = transform.inverse_xyxy_normalized(boxes)
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0


def test_rejects_a_non_rgb_array() -> None:
    with pytest.raises(ValueError, match="RGB"):
        letterbox_image(np.zeros((10, 10), dtype=np.uint8), 640)


def test_rejects_a_degenerate_size() -> None:
    with pytest.raises(ValueError, match="positive"):
        LetterboxTransform.from_size(0, 10, 640)
