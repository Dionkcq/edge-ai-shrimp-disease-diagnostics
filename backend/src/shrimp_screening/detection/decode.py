"""Turning a raw YOLOv8/v11 detect-head tensor into normalized detections.

Nothing in this module imports onnxruntime, torch or ultralytics. It is pure
numpy over an array, which is what makes the riskiest part of the system testable
before any weights exist -- and what makes the trainer replaceable.

The output contract this module *asserts* rather than infers
--------------------------------------------------------------
For a YOLOv8/v11 detect head exported with ``nms=False``:

* one output of shape ``(1, 4 + nc, anchors)`` -- at 640 with two classes that is
  ``(1, 6, 8400)`` where ``8400 = 80^2 + 40^2 + 20^2``;
* the layout is **channels-then-anchors**, transposed relative to YOLOv5. Reading
  it the other way round is the single most likely silent failure in this project,
  so a ``(1, anchors, 4 + nc)`` tensor is rejected rather than misread;
* rows 0-3 are ``cx, cy, w, h`` in **input-tensor pixels** in the letterboxed
  frame -- not normalized, not original-image coordinates. DFL is decoded inside
  the graph;
* rows 4.. are per-class scores that are **already sigmoid-activated**, and there
  is **no objectness row** (the head is anchor-free and decoupled). Multiplying by
  a non-existent objectness would silently halve every confidence;
* exporting with ``nms=True`` produces ``(1, max_det, 6)`` instead. The channel
  check below rejects that shape too, which is why ``nms=False`` is pinned.
"""

from __future__ import annotations

import numpy as np

from shrimp_screening.detection.letterbox import LetterboxTransform
from shrimp_screening.detection.nms import class_aware_nms
from shrimp_screening.detection.protocol import Detection

#: Ceiling on the candidates fed into NMS, mirroring Ultralytics' ``max_nms``.
MAX_NMS_CANDIDATES = 30_000


class OutputLayoutError(ValueError):
    """The raw tensor does not match the declared output contract."""


def expected_anchor_count(input_size: int, strides: tuple[int, ...] = (8, 16, 32)) -> int:
    """Anchors emitted by a three-scale detect head at ``input_size``."""
    return sum((input_size // stride) ** 2 for stride in strides)


def decode_ultralytics_v8(
    output: np.ndarray,
    class_names: dict[int, str],
    transform: LetterboxTransform,
    *,
    score_threshold: float,
    iou_threshold: float,
    max_detections: int = 300,
) -> list[Detection]:
    """Decode one raw detect-head output into normalized original-frame detections.

    ``score_threshold`` is applied to the per-class score *before* NMS, exactly as
    Ultralytics does, so the two thresholds compose the same way they would in the
    reference implementation.
    """
    if not class_names:
        raise OutputLayoutError("the model declares no class names; refusing to guess labels")
    expected_channels = 4 + len(class_names)
    if output.ndim != 3 or output.shape[0] != 1 or output.shape[1] != expected_channels:
        raise OutputLayoutError(
            f"unsupported output layout {tuple(output.shape)}; expected "
            f"(1, {expected_channels}, anchors) channels-then-anchors. A "
            f"(1, anchors, {expected_channels}) tensor is the transposed YOLOv5 "
            "layout and a (1, max_det, 6) tensor is an nms=True export; neither is "
            "supported because either would be silently misread."
        )

    raw = np.asarray(output[0], dtype=np.float32)
    scores_per_class = raw[4:]
    class_index = scores_per_class.argmax(axis=0)
    raw_scores = scores_per_class.max(axis=0)
    finite = np.isfinite(raw_scores) & np.isfinite(raw[:4]).all(axis=0)
    scores = np.clip(raw_scores, 0.0, 1.0)

    candidate = np.flatnonzero(finite & (scores >= score_threshold))
    if candidate.size == 0:
        return []
    if candidate.size > MAX_NMS_CANDIDATES:
        # Keep the strongest candidates; an unbounded NMS on a degenerate tensor is
        # a denial-of-service vector on a two-core machine.
        candidate = candidate[np.argsort(-scores[candidate], kind="stable")[:MAX_NMS_CANDIDATES]]

    cx, cy, width, height = (raw[i, candidate].astype(np.float64) for i in range(4))
    first_x = cx - width / 2.0
    second_x = cx + width / 2.0
    first_y = cy - height / 2.0
    second_y = cy + height / 2.0
    boxes = np.stack(
        [
            np.minimum(first_x, second_x),
            np.minimum(first_y, second_y),
            np.maximum(first_x, second_x),
            np.maximum(first_y, second_y),
        ],
        axis=1,
    )
    kept = class_aware_nms(
        boxes,
        scores[candidate].astype(np.float64),
        class_index[candidate],
        iou_threshold,
        max_detections=max_detections,
    )
    if not kept:
        return []

    normalized = transform.inverse_xyxy_normalized(boxes[kept])
    detections: list[Detection] = []
    for position, box in zip(kept, normalized, strict=True):
        index = int(class_index[candidate[position]])
        name = class_names.get(index)
        if name is None:
            raise OutputLayoutError(
                f"the model produced class index {index}, which its own metadata does not name"
            )
        detections.append(
            Detection(
                class_index=index,
                class_name=name,
                score=float(scores[candidate[position]]),
                box=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
            )
        )
    return detections
