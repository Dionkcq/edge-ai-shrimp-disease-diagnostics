"""Decoding a raw detect-head tensor.

The negative tests matter more than the positive one. A wrong-but-plausible tensor
layout does not crash -- it produces confident boxes in the wrong places, which is
indistinguishable from a badly trained model and would be debugged for days.
"""

from __future__ import annotations

import numpy as np
import pytest

from shrimp_screening.detection.decode import (
    OutputLayoutError,
    decode_ultralytics_v8,
    expected_anchor_count,
)
from shrimp_screening.detection.letterbox import LetterboxTransform

NAMES = {0: "dark_gill", 1: "white_spot"}


def _tensor(anchors: int = 8, channels: int = 6) -> np.ndarray:
    raw = np.zeros((1, channels, anchors), dtype=np.float32)
    raw[0, 4:, :] = 0.01
    return raw


def test_anchor_count_matches_a_three_scale_head() -> None:
    assert expected_anchor_count(640) == 80**2 + 40**2 + 20**2 == 8400
    assert expected_anchor_count(320) == 40**2 + 20**2 + 10**2


def test_decodes_a_planted_box_back_to_normalized_original_coordinates() -> None:
    transform = LetterboxTransform.from_size(1000, 500, 640)
    original = (250.0, 100.0, 750.0, 400.0)
    x1, y1, x2, y2 = transform.forward_xyxy(original)

    raw = _tensor()
    raw[0, 0, 3] = (x1 + x2) / 2
    raw[0, 1, 3] = (y1 + y2) / 2
    raw[0, 2, 3] = x2 - x1
    raw[0, 3, 3] = y2 - y1
    raw[0, 5, 3] = 0.9

    found = decode_ultralytics_v8(raw, NAMES, transform, score_threshold=0.25, iou_threshold=0.45)
    assert len(found) == 1
    assert found[0].class_name == "white_spot"
    assert found[0].class_index == 1
    assert found[0].score == pytest.approx(0.9)
    assert found[0].box == pytest.approx((0.25, 0.20, 0.75, 0.80), abs=2e-3)


def test_scores_are_taken_as_is_with_no_objectness_multiplication() -> None:
    """The v8 head is anchor-free and decoupled: there is no objectness row.

    Multiplying rows 4.. by a non-existent objectness would halve every confidence
    and quietly move results across the decision thresholds.
    """
    transform = LetterboxTransform.from_size(640, 640, 640)
    raw = _tensor()
    raw[0, :4, 0] = [320, 320, 100, 100]
    raw[0, 4, 0] = 0.61
    found = decode_ultralytics_v8(raw, NAMES, transform, score_threshold=0.25, iou_threshold=0.45)
    assert found[0].score == pytest.approx(0.61)


def test_malformed_candidates_are_dropped_or_normalized_before_contract_creation() -> None:
    transform = LetterboxTransform.from_size(640, 640, 640)
    raw = _tensor()
    raw[0, :4, 0] = [320, 320, 100, 100]
    raw[0, 4, 0] = np.inf
    raw[0, :4, 1] = [np.nan, 320, 100, 100]
    raw[0, 4, 1] = 0.95
    raw[0, :4, 2] = [320, 320, -100, 100]
    raw[0, 4, 2] = 1.5

    found = decode_ultralytics_v8(raw, NAMES, transform, score_threshold=0.25, iou_threshold=0.45)

    assert len(found) == 1
    assert found[0].score == 1.0
    x1, y1, x2, y2 = found[0].box
    assert x1 <= x2
    assert y1 <= y2


def test_rejects_the_transposed_yolov5_layout() -> None:
    transform = LetterboxTransform.from_size(640, 640, 640)
    transposed = np.zeros((1, 8400, 6), dtype=np.float32)
    with pytest.raises(OutputLayoutError, match="transposed"):
        decode_ultralytics_v8(
            transposed, NAMES, transform, score_threshold=0.25, iou_threshold=0.45
        )


def test_rejects_an_nms_true_export() -> None:
    """``nms=True`` yields ``(1, max_det, 6)``; with two classes that is 6 channels
    of a *different meaning*, so it must be refused rather than misread."""
    transform = LetterboxTransform.from_size(640, 640, 640)
    exported_with_nms = np.zeros((1, 300, 6), dtype=np.float32)
    with pytest.raises(OutputLayoutError):
        decode_ultralytics_v8(
            exported_with_nms, NAMES, transform, score_threshold=0.25, iou_threshold=0.45
        )


def test_rejects_a_channel_count_that_disagrees_with_the_class_names() -> None:
    transform = LetterboxTransform.from_size(640, 640, 640)
    three_classes = np.zeros((1, 7, 8400), dtype=np.float32)
    with pytest.raises(OutputLayoutError, match=r"expected \(1, 6, anchors\)"):
        decode_ultralytics_v8(
            three_classes, NAMES, transform, score_threshold=0.25, iou_threshold=0.45
        )


def test_refuses_to_decode_without_class_names() -> None:
    transform = LetterboxTransform.from_size(640, 640, 640)
    with pytest.raises(OutputLayoutError, match="no class names"):
        decode_ultralytics_v8(_tensor(), {}, transform, score_threshold=0.25, iou_threshold=0.45)


def test_background_anchors_below_the_threshold_are_dropped() -> None:
    transform = LetterboxTransform.from_size(640, 640, 640)
    assert (
        decode_ultralytics_v8(_tensor(), NAMES, transform, score_threshold=0.25, iou_threshold=0.45)
        == []
    )


def test_overlapping_same_class_candidates_are_suppressed() -> None:
    transform = LetterboxTransform.from_size(640, 640, 640)
    raw = _tensor()
    for slot, score in enumerate((0.9, 0.8, 0.7)):
        raw[0, :4, slot] = [320 + slot, 320 + slot, 200, 200]
        raw[0, 5, slot] = score
    found = decode_ultralytics_v8(raw, NAMES, transform, score_threshold=0.25, iou_threshold=0.45)
    assert len(found) == 1
    assert found[0].score == pytest.approx(0.9)


def test_overlapping_different_class_candidates_both_survive() -> None:
    """Both markers can legitimately appear on one animal, in the same place."""
    transform = LetterboxTransform.from_size(640, 640, 640)
    raw = _tensor()
    raw[0, :4, 0] = [320, 320, 200, 200]
    raw[0, 4, 0] = 0.9
    raw[0, :4, 1] = [320, 320, 200, 200]
    raw[0, 5, 1] = 0.8
    found = decode_ultralytics_v8(raw, NAMES, transform, score_threshold=0.25, iou_threshold=0.45)
    assert {item.class_name for item in found} == {"dark_gill", "white_spot"}


def test_max_detections_is_honoured() -> None:
    transform = LetterboxTransform.from_size(640, 640, 640)
    raw = _tensor(anchors=40)
    for slot in range(40):
        raw[0, :4, slot] = [10 + slot * 15, 10 + slot * 15, 8, 8]
        raw[0, 4, slot] = 0.9
    found = decode_ultralytics_v8(
        raw, NAMES, transform, score_threshold=0.25, iou_threshold=0.45, max_detections=5
    )
    assert len(found) == 5


def test_a_class_index_the_metadata_does_not_name_is_an_error_not_a_guess() -> None:
    """Gapped class names would otherwise silently mislabel or crash mid-response."""
    transform = LetterboxTransform.from_size(640, 640, 640)
    raw = _tensor()
    raw[0, :4, 0] = [320, 320, 100, 100]
    raw[0, 5, 0] = 0.9  # channel 5 is class index 1
    gapped = {0: "dark_gill", 2: "white_spot"}
    with pytest.raises(OutputLayoutError, match="does not name"):
        decode_ultralytics_v8(raw, gapped, transform, score_threshold=0.25, iou_threshold=0.45)
