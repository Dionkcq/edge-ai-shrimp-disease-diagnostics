"""Decoding the from-scratch, anchor-based ``CUSTOM_YOLO_ANCHOR_V1`` detect head.

Sits alongside ``test_decode.py`` (the Ultralytics path, untouched) rather than
replacing it -- this is an additive runtime contract, not a swap.
"""

from __future__ import annotations

import numpy as np
import pytest

from shrimp_screening.detection.decode import (
    CUSTOM_YOLO_ANCHORS_PER_SCALE,
    CUSTOM_YOLO_STRIDES,
    OutputLayoutError,
    decode_custom_yolo_anchor_v1,
    expected_anchor_count,
)
from shrimp_screening.detection.letterbox import LetterboxTransform

NAMES = {0: "dark_gill", 1: "white_spot"}
ANCHORS = tuple((10.0 * (index + 1), 10.0 * (index + 1)) for index in range(9))
INPUT_SIZE = 640
TOTAL_ANCHOR_POSITIONS = CUSTOM_YOLO_ANCHORS_PER_SCALE * expected_anchor_count(
    INPUT_SIZE, CUSTOM_YOLO_STRIDES
)


def _tensor(channels: int = 7, anchors: int = TOTAL_ANCHOR_POSITIONS) -> np.ndarray:
    raw = np.zeros((1, channels, anchors), dtype=np.float32)
    # A very negative objectness logit -> ~0 score everywhere by default.
    raw[0, 4, :] = -20.0
    return raw


def _flat_index(*, scale_index: int, anchor_index: int, grid_y: int, grid_x: int) -> int:
    """Mirrors ``decode._build_custom_yolo_grid``'s anchor-major/row-major/scale-major order."""
    offset = 0
    for index, stride in enumerate(CUSTOM_YOLO_STRIDES):
        grid = INPUT_SIZE // stride
        if index == scale_index:
            return offset + anchor_index * grid * grid + grid_y * grid + grid_x
        offset += CUSTOM_YOLO_ANCHORS_PER_SCALE * grid * grid
    raise AssertionError("scale_index out of range")


def _plant(
    raw: np.ndarray,
    *,
    scale_index: int,
    anchor_index: int,
    grid_y: int,
    grid_x: int,
    tx: float,
    ty: float,
    tw: float,
    th: float,
    objectness: float,
    class_index: int,
    class_logit: float,
) -> None:
    position = _flat_index(
        scale_index=scale_index, anchor_index=anchor_index, grid_y=grid_y, grid_x=grid_x
    )
    raw[0, 0, position] = tx
    raw[0, 1, position] = ty
    raw[0, 2, position] = tw
    raw[0, 3, position] = th
    raw[0, 4, position] = objectness
    raw[0, 5 + class_index, position] = class_logit


def test_total_anchor_position_count() -> None:
    assert TOTAL_ANCHOR_POSITIONS == 3 * (80**2 + 40**2 + 20**2) == 25200


def test_decodes_a_planted_box_back_to_normalized_original_coordinates() -> None:
    transform = LetterboxTransform.from_size(INPUT_SIZE, INPUT_SIZE, INPUT_SIZE)
    raw = _tensor()
    _plant(
        raw,
        scale_index=0,
        anchor_index=0,
        grid_y=5,
        grid_x=5,
        tx=0.0,  # sigmoid(0) = 0.5 -> cx = (0.5 + 5) * 8 = 44
        ty=0.0,
        tw=0.0,  # exp(0) = 1 -> width = anchor_w * 1 = 10
        th=0.0,
        objectness=20.0,  # sigmoid(20) ~= 1.0
        class_index=1,
        class_logit=20.0,
    )

    found = decode_custom_yolo_anchor_v1(
        raw, NAMES, transform, ANCHORS, score_threshold=0.5, iou_threshold=0.45
    )

    assert len(found) == 1
    assert found[0].class_name == "white_spot"
    assert found[0].class_index == 1
    assert found[0].score == pytest.approx(1.0, abs=1e-3)
    # center (44, 44), width/height 10 -> xyxy (39, 39, 49, 49) on a 640 canvas.
    assert found[0].box == pytest.approx((39 / 640, 39 / 640, 49 / 640, 49 / 640), abs=2e-3)


def test_score_is_objectness_times_class_not_either_alone() -> None:
    """Unlike the Ultralytics path, this head *does* have an objectness row."""
    transform = LetterboxTransform.from_size(INPUT_SIZE, INPUT_SIZE, INPUT_SIZE)
    raw = _tensor()
    # sigmoid(0) = 0.5 for both objectness and the class logit -> combined 0.25.
    _plant(
        raw,
        scale_index=0,
        anchor_index=0,
        grid_y=0,
        grid_x=0,
        tx=0.0,
        ty=0.0,
        tw=0.0,
        th=0.0,
        objectness=0.0,
        class_index=0,
        class_logit=0.0,
    )

    found = decode_custom_yolo_anchor_v1(
        raw, NAMES, transform, ANCHORS, score_threshold=0.2, iou_threshold=0.45
    )
    assert len(found) == 1
    assert found[0].score == pytest.approx(0.25, abs=1e-3)

    dropped = decode_custom_yolo_anchor_v1(
        raw, NAMES, transform, ANCHORS, score_threshold=0.3, iou_threshold=0.45
    )
    assert dropped == []


def test_rejects_a_channel_count_that_disagrees_with_class_names() -> None:
    transform = LetterboxTransform.from_size(INPUT_SIZE, INPUT_SIZE, INPUT_SIZE)
    wrong_channels = np.zeros((1, 6, TOTAL_ANCHOR_POSITIONS), dtype=np.float32)
    with pytest.raises(OutputLayoutError, match=r"expected \(1, 7, anchors\)"):
        decode_custom_yolo_anchor_v1(
            wrong_channels, NAMES, transform, ANCHORS, score_threshold=0.5, iou_threshold=0.45
        )


def test_rejects_an_anchor_position_count_that_disagrees_with_a_three_scale_head() -> None:
    transform = LetterboxTransform.from_size(INPUT_SIZE, INPUT_SIZE, INPUT_SIZE)
    wrong_positions = np.zeros((1, 7, 100), dtype=np.float32)
    with pytest.raises(OutputLayoutError, match="anchor positions"):
        decode_custom_yolo_anchor_v1(
            wrong_positions, NAMES, transform, ANCHORS, score_threshold=0.5, iou_threshold=0.45
        )


def test_rejects_a_wrong_anchor_count() -> None:
    transform = LetterboxTransform.from_size(INPUT_SIZE, INPUT_SIZE, INPUT_SIZE)
    with pytest.raises(OutputLayoutError, match="requires exactly 9 anchors"):
        decode_custom_yolo_anchor_v1(
            _tensor(), NAMES, transform, ANCHORS[:3], score_threshold=0.5, iou_threshold=0.45
        )


def test_refuses_to_decode_without_class_names() -> None:
    transform = LetterboxTransform.from_size(INPUT_SIZE, INPUT_SIZE, INPUT_SIZE)
    with pytest.raises(OutputLayoutError, match="no class names"):
        decode_custom_yolo_anchor_v1(
            _tensor(), {}, transform, ANCHORS, score_threshold=0.5, iou_threshold=0.45
        )


def test_a_class_index_the_metadata_does_not_name_is_an_error_not_a_guess() -> None:
    transform = LetterboxTransform.from_size(INPUT_SIZE, INPUT_SIZE, INPUT_SIZE)
    raw = _tensor()
    _plant(
        raw,
        scale_index=0,
        anchor_index=0,
        grid_y=0,
        grid_x=0,
        tx=0.0,
        ty=0.0,
        tw=0.0,
        th=0.0,
        objectness=20.0,
        class_index=1,
        class_logit=20.0,
    )
    gapped = {0: "dark_gill", 2: "white_spot"}
    with pytest.raises(OutputLayoutError, match="does not name"):
        decode_custom_yolo_anchor_v1(
            raw, gapped, transform, ANCHORS, score_threshold=0.5, iou_threshold=0.45
        )


def test_overlapping_same_class_candidates_are_suppressed() -> None:
    """Three different anchors at one cell, all decoded to the same 10x10 box."""
    transform = LetterboxTransform.from_size(INPUT_SIZE, INPUT_SIZE, INPUT_SIZE)
    raw = _tensor()
    for anchor_index, objectness in ((0, 20.0), (1, 10.0), (2, 5.0)):
        anchor_w, _ = ANCHORS[anchor_index]
        # exp(tw) * anchor_w == 10 for every anchor, so all three land on one box.
        equalizing_tw = float(np.log(10.0 / anchor_w))
        _plant(
            raw,
            scale_index=0,
            anchor_index=anchor_index,
            grid_y=10,
            grid_x=10,
            tx=0.0,
            ty=0.0,
            tw=equalizing_tw,
            th=equalizing_tw,
            objectness=objectness,
            class_index=0,
            class_logit=20.0,
        )
    found = decode_custom_yolo_anchor_v1(
        raw, NAMES, transform, ANCHORS, score_threshold=0.5, iou_threshold=0.45
    )
    assert len(found) == 1
    assert found[0].class_name == "dark_gill"


def test_a_non_square_photograph_stays_inside_its_own_frame() -> None:
    transform = LetterboxTransform.from_size(1280, 720, INPUT_SIZE)
    raw = _tensor()
    _plant(
        raw,
        scale_index=2,
        anchor_index=2,
        grid_y=10,
        grid_x=10,
        tx=0.0,
        ty=0.0,
        tw=0.0,
        th=0.0,
        objectness=20.0,
        class_index=0,
        class_logit=20.0,
    )
    found = decode_custom_yolo_anchor_v1(
        raw, NAMES, transform, ANCHORS, score_threshold=0.5, iou_threshold=0.45
    )
    for detection in found:
        assert all(0.0 <= corner <= 1.0 for corner in detection.box)
        assert detection.box[0] <= detection.box[2]
        assert detection.box[1] <= detection.box[3]
