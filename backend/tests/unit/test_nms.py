from __future__ import annotations

import numpy as np
import pytest

from shrimp_screening.detection.nms import class_aware_nms


def test_suppression_is_confined_to_a_single_class() -> None:
    """Two identical boxes of different classes must both survive.

    Agnostic NMS would drop one, collapsing MULTIPLE_TARGET_MARKERS_DETECTED into a
    single-marker decision without anything in the response indicating why.
    """
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10], [1, 1, 9, 9]], dtype=np.float64)
    scores = np.array([0.9, 0.8, 0.7])
    classes = np.array([0, 1, 0])
    assert class_aware_nms(boxes, scores, classes, 0.5) == [0, 1]


def test_returns_indices_in_descending_score_order() -> None:
    boxes = np.array([[0, 0, 5, 5], [100, 100, 105, 105], [200, 200, 205, 205]], dtype=np.float64)
    scores = np.array([0.2, 0.9, 0.5])
    classes = np.array([0, 0, 0])
    assert class_aware_nms(boxes, scores, classes, 0.5) == [1, 2, 0]


def test_equal_scores_resolve_deterministically_by_original_order() -> None:
    boxes = np.array([[0, 0, 5, 5], [100, 100, 105, 105]], dtype=np.float64)
    scores = np.array([0.5, 0.5])
    classes = np.array([0, 0])
    assert class_aware_nms(boxes, scores, classes, 0.5) == [0, 1]


def test_a_threshold_of_one_suppresses_nothing_but_exact_supersets() -> None:
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float64)
    scores = np.array([0.9, 0.8])
    classes = np.array([0, 0])
    # IoU is exactly 1.0, and the comparison is strict, so both survive at 1.0.
    assert class_aware_nms(boxes, scores, classes, 1.0) == [0, 1]
    assert class_aware_nms(boxes, scores, classes, 0.99) == [0]


def test_zero_area_boxes_do_not_divide_by_zero() -> None:
    boxes = np.array([[5, 5, 5, 5], [5, 5, 5, 5]], dtype=np.float64)
    scores = np.array([0.9, 0.8])
    classes = np.array([0, 0])
    assert class_aware_nms(boxes, scores, classes, 0.5) == [0, 1]


def test_max_detections_caps_the_result() -> None:
    boxes = np.array([[i * 50, 0, i * 50 + 10, 10] for i in range(10)], dtype=np.float64)
    scores = np.linspace(0.1, 0.9, 10)
    classes = np.zeros(10, dtype=np.int64)
    assert len(class_aware_nms(boxes, scores, classes, 0.5, max_detections=3)) == 3


def test_empty_input_is_not_an_error() -> None:
    empty = np.zeros((0, 4))
    assert class_aware_nms(empty, np.zeros(0), np.zeros(0, dtype=np.int64), 0.5) == []


@pytest.mark.parametrize(
    ("boxes", "message"),
    [
        (np.zeros((3, 5)), "must be"),
        (np.zeros((3,)), "must be"),
    ],
)
def test_malformed_boxes_are_rejected(boxes: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        class_aware_nms(boxes, np.zeros(3), np.zeros(3, dtype=np.int64), 0.5)


def test_an_out_of_range_iou_threshold_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        class_aware_nms(np.zeros((1, 4)), np.zeros(1), np.zeros(1, dtype=np.int64), 1.5)
