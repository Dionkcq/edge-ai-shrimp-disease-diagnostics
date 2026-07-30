"""Class-aware non-maximum suppression, numpy only.

Class-aware (``agnostic=False``) rather than agnostic: a white-spot box and a
dark-gill box may legitimately overlap on the same animal, and suppressing one
because of the other would silently collapse ``MULTIPLE_TARGET_MARKERS_DETECTED``
into a single-marker decision.

The implementation suppresses within a class directly rather than via Ultralytics'
``+ class_idx * max_wh`` coordinate-offset trick. That trick exists to reuse a
single batched kernel; here it would only add a magic constant and a failure mode
for boxes larger than ``max_wh``.
"""

from __future__ import annotations

import numpy as np


def class_aware_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    iou_threshold: float,
    *,
    max_detections: int | None = None,
) -> list[int]:
    """Return the indices to keep, in descending score order.

    ``boxes`` is ``(n, 4)`` xyxy in any one consistent frame.
    """
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"boxes must be (n, 4), got {boxes.shape}")
    if not (boxes.shape[0] == scores.shape[0] == classes.shape[0]):
        raise ValueError("boxes, scores and classes must describe the same candidates")
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must lie in [0, 1]")

    # Stable sort, so equal scores resolve by original index rather than by whatever
    # order the sort happens to produce. A decision must not depend on that.
    order = np.argsort(-scores, kind="stable")
    areas = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])

    kept: list[int] = []
    while order.size:
        current = int(order[0])
        kept.append(current)
        if max_detections is not None and len(kept) >= max_detections:
            break
        rest = order[1:]
        if rest.size == 0:
            break
        overlap_w = np.minimum(boxes[current, 2], boxes[rest, 2]) - np.maximum(
            boxes[current, 0], boxes[rest, 0]
        )
        overlap_h = np.minimum(boxes[current, 3], boxes[rest, 3]) - np.maximum(
            boxes[current, 1], boxes[rest, 1]
        )
        intersection = np.maximum(0.0, overlap_w) * np.maximum(0.0, overlap_h)
        union = areas[current] + areas[rest] - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        suppress = (classes[rest] == classes[current]) & (iou > iou_threshold)
        order = rest[~suppress]
    return kept
