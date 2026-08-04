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

from functools import lru_cache

import numpy as np

from shrimp_screening.detection.letterbox import LetterboxTransform
from shrimp_screening.detection.nms import class_aware_nms
from shrimp_screening.detection.protocol import Detection
from shrimp_screening.settings import MAX_NMS_CANDIDATES

#: Strides and anchors-per-scale for CUSTOM_YOLO_ANCHOR_V1. Mirrors
#: ``training/src/shrimp_training/model.py::STRIDES`` and
#: ``training/src/shrimp_training/anchors.py::ANCHORS_PER_SCALE`` -- kept as plain
#: constants rather than imported, since the backend never imports the
#: (AGPL, separately locked) training package.
CUSTOM_YOLO_STRIDES: tuple[int, ...] = (8, 16, 32)
CUSTOM_YOLO_ANCHORS_PER_SCALE = 3


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


def _sigmoid(values: np.ndarray) -> np.ndarray:
    result: np.ndarray = 1.0 / (1.0 + np.exp(-values))
    return result


@lru_cache(maxsize=4)
def _build_custom_yolo_grid(
    input_size: int, anchors: tuple[tuple[float, float], ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-anchor-position ``(stride, anchor_w, anchor_h, grid_x, grid_y)``.

    Built in the exact anchor-major / row-major-grid / scale-major order the
    training export's ``_FlattenHead`` flattens ``(na, H, W, no)`` into -- this
    ordering is what lets a flat index in the raw tensor be mapped back to the
    scale/anchor/cell it came from without needing that information in the graph.
    """
    strides: list[np.ndarray] = []
    anchor_widths: list[np.ndarray] = []
    anchor_heights: list[np.ndarray] = []
    grid_xs: list[np.ndarray] = []
    grid_ys: list[np.ndarray] = []
    for scale_index, stride in enumerate(CUSTOM_YOLO_STRIDES):
        grid = input_size // stride
        grid_y, grid_x = np.meshgrid(np.arange(grid), np.arange(grid), indexing="ij")
        flat_x = grid_x.reshape(-1).astype(np.float64)
        flat_y = grid_y.reshape(-1).astype(np.float64)
        scale_anchors = anchors[
            scale_index * CUSTOM_YOLO_ANCHORS_PER_SCALE : (scale_index + 1)
            * CUSTOM_YOLO_ANCHORS_PER_SCALE
        ]
        for anchor_w, anchor_h in scale_anchors:
            strides.append(np.full(flat_x.shape, float(stride)))
            anchor_widths.append(np.full(flat_x.shape, float(anchor_w)))
            anchor_heights.append(np.full(flat_x.shape, float(anchor_h)))
            grid_xs.append(flat_x)
            grid_ys.append(flat_y)
    return (
        np.concatenate(strides),
        np.concatenate(anchor_widths),
        np.concatenate(anchor_heights),
        np.concatenate(grid_xs),
        np.concatenate(grid_ys),
    )


def decode_custom_yolo_anchor_v1(
    output: np.ndarray,
    class_names: dict[int, str],
    transform: LetterboxTransform,
    anchors: tuple[tuple[float, float], ...],
    *,
    score_threshold: float,
    iou_threshold: float,
    max_detections: int = 300,
) -> list[Detection]:
    """Decode one raw ``CUSTOM_YOLO_ANCHOR_V1`` tensor into normalized detections.

    Output contract: ``(1, 5 + nc, anchors)`` channels-then-anchors, rows
    ``[tx, ty, tw, th, objectness, class0, ...]`` in **raw, pre-activation** form --
    unlike the Ultralytics path, this graph performs no sigmoid/exp itself, so this
    function does exactly what the from-scratch training loss assumes:
    ``score = sigmoid(objectness) * sigmoid(class)``,
    ``cx = (sigmoid(tx) + grid_x) * stride``, ``w = anchor_w * exp(tw)``.
    """
    if not class_names:
        raise OutputLayoutError("the model declares no class names; refusing to guess labels")
    expected_anchors = CUSTOM_YOLO_ANCHORS_PER_SCALE * len(CUSTOM_YOLO_STRIDES)
    if len(anchors) != expected_anchors:
        raise OutputLayoutError(
            f"custom_yolo_anchor_v1 requires exactly {expected_anchors} anchors, got {len(anchors)}"
        )
    expected_channels = 5 + len(class_names)
    if output.ndim != 3 or output.shape[0] != 1 or output.shape[1] != expected_channels:
        raise OutputLayoutError(
            f"unsupported output layout {tuple(output.shape)}; expected "
            f"(1, {expected_channels}, anchors) channels-then-anchors for "
            "custom_yolo_anchor_v1"
        )
    input_size = transform.input_size
    expected_positions = CUSTOM_YOLO_ANCHORS_PER_SCALE * expected_anchor_count(
        input_size, CUSTOM_YOLO_STRIDES
    )
    if output.shape[2] != expected_positions:
        raise OutputLayoutError(
            f"output declares {output.shape[2]} anchor positions; a 3-scale "
            f"anchor-based head with {CUSTOM_YOLO_ANCHORS_PER_SCALE} anchors/scale at "
            f"{input_size} emits {expected_positions}"
        )

    raw = np.asarray(output[0], dtype=np.float32)
    strides, anchor_w, anchor_h, grid_x, grid_y = _build_custom_yolo_grid(input_size, anchors)

    tx, ty, tw, th, objectness_logit = raw[0], raw[1], raw[2], raw[3], raw[4]
    class_logits = raw[5:]
    objectness = _sigmoid(objectness_logit)
    class_scores = _sigmoid(class_logits)
    combined = objectness[None, :] * class_scores
    class_index = combined.argmax(axis=0)
    raw_scores = combined.max(axis=0)
    finite = np.isfinite(raw_scores) & np.isfinite(raw[:5]).all(axis=0)
    scores = np.clip(raw_scores, 0.0, 1.0)

    candidate = np.flatnonzero(finite & (scores >= score_threshold))
    if candidate.size == 0:
        return []
    if candidate.size > MAX_NMS_CANDIDATES:
        candidate = candidate[np.argsort(-scores[candidate], kind="stable")[:MAX_NMS_CANDIDATES]]

    cx = (_sigmoid(tx[candidate]) + grid_x[candidate]) * strides[candidate]
    cy = (_sigmoid(ty[candidate]) + grid_y[candidate]) * strides[candidate]
    # Clipped before exp(): an untrusted/degenerate tensor should not be able to
    # produce inf/nan box sizes that then corrupt NMS.
    width = anchor_w[candidate] * np.exp(np.clip(tw[candidate], -20.0, 20.0))
    height = anchor_h[candidate] * np.exp(np.clip(th[candidate], -20.0, 20.0))
    first_x, second_x = cx - width / 2.0, cx + width / 2.0
    first_y, second_y = cy - height / 2.0, cy + height / 2.0
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
