# SPDX-License-Identifier: AGPL-3.0-or-later
"""From-scratch mAP evaluator.

Shaped to match what ``training/artifacts.py`` already expects from an Ultralytics
``model.val(...)`` call -- a ``results_dict`` with the same string keys
(``"metrics/precision(B)"`` etc.) and a ``box`` object exposing ``.maps``
(per-class mAP50-95) and ``.p_curve``/``.r_curve`` (precision/recall sampled over a
confidence axis) -- so ``evaluate_artifact``'s dataset-binding checks and
``compare_parity``'s threshold-matched-confidence comparison keep working
unmodified against a different trainer.

Implements standard greedy IoU-matched TP/FP accumulation per class, 101-point
interpolated average precision (COCO convention) at IoU 0.5 (mAP50) and averaged
over 10 IoU thresholds 0.5:0.95 (mAP50-95), and precision/recall curves sampled at
1000 evenly spaced confidence thresholds.

Detections and ground truth are both compared in 640-canvas pixel space (the space
``data.py`` already puts targets in) -- there is no need to project back to
original-image coordinates for a training-time metric.
"""

from __future__ import annotations

from functools import lru_cache
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset

from shrimp_training.anchors import AnchorSet
from shrimp_training.data import collate_fn
from shrimp_training.model import STRIDES

_IOU_THRESHOLDS: tuple[float, ...] = tuple(round(0.5 + 0.05 * step, 2) for step in range(10))
_CONFIDENCE_SAMPLES = 1000
_RECALL_INTERPOLATION_POINTS = np.linspace(0.0, 1.0, 101)

_Detection = tuple[int, int, float, tuple[float, float, float, float]]
_GroundTruth = tuple[int, int, tuple[float, float, float, float]]


def _iou(
    box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]
) -> float:
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_w = max(0.0, min(xa2, xb2) - max(xa1, xb1))
    inter_h = max(0.0, min(ya2, yb2) - max(ya1, yb1))
    intersection = inter_w * inter_h
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _class_aware_nms(
    boxes: NDArray[Any], iou_threshold: float, max_detections: int = 300
) -> list[int]:
    """``boxes``: ``(n, 6)`` rows of ``[x1, y1, x2, y2, score, class]``."""
    order = np.argsort(-boxes[:, 4])
    suppressed = np.zeros(len(boxes), dtype=bool)
    keep: list[int] = []
    for i in order:
        if suppressed[i]:
            continue
        keep.append(int(i))
        if len(keep) >= max_detections:
            break
        for j in order:
            if j == i or suppressed[j] or boxes[j, 5] != boxes[i, 5]:
                continue
            if _iou(tuple(boxes[i, :4]), tuple(boxes[j, :4])) > iou_threshold:
                suppressed[j] = True
    return keep


@lru_cache(maxsize=8)
def _grid(grid_h: int, grid_w: int) -> tuple[NDArray[Any], NDArray[Any]]:
    """``(grid_y, grid_x)`` for one scale -- fixed by ``image_size``/stride, so this
    is the same pair on every call within one evaluation run; cached instead of
    rebuilt once per image."""
    return np.meshgrid(np.arange(grid_h), np.arange(grid_w), indexing="ij")


def _decode_predictions(
    scale_outputs: list[torch.Tensor],
    anchors: AnchorSet,
    *,
    score_threshold: float,
    iou_threshold: float,
) -> NDArray[Any]:
    """Decode one image's raw 3-scale output into ``(n, 6)`` ``[x1,y1,x2,y2,score,class]``."""
    rows: list[NDArray[Any]] = []
    for prediction, stride in zip(scale_outputs, STRIDES, strict=True):
        num_anchors, grid_h, grid_w, _ = prediction.shape
        anchor_wh = np.asarray(anchors.for_stride(stride), dtype=np.float64)
        raw = prediction.detach().cpu().numpy().astype(np.float64)
        grid_y, grid_x = _grid(grid_h, grid_w)
        for anchor_index in range(num_anchors):
            layer = raw[anchor_index]
            cx = (1.0 / (1.0 + np.exp(-layer[..., 0])) + grid_x) * stride
            cy = (1.0 / (1.0 + np.exp(-layer[..., 1])) + grid_y) * stride
            bw = anchor_wh[anchor_index, 0] * np.exp(layer[..., 2])
            bh = anchor_wh[anchor_index, 1] * np.exp(layer[..., 3])
            objectness = 1.0 / (1.0 + np.exp(-layer[..., 4]))
            class_scores = 1.0 / (1.0 + np.exp(-layer[..., 5:]))
            score_per_class = objectness[..., None] * class_scores
            class_index = score_per_class.argmax(axis=-1)
            score = score_per_class.max(axis=-1)
            mask = score >= score_threshold
            if not mask.any():
                continue
            cx_m, cy_m, bw_m, bh_m = cx[mask], cy[mask], bw[mask], bh[mask]
            score_m, class_m = score[mask], class_index[mask].astype(np.float64)
            x1, y1 = cx_m - bw_m / 2.0, cy_m - bh_m / 2.0
            x2, y2 = cx_m + bw_m / 2.0, cy_m + bh_m / 2.0
            rows.append(np.stack([x1, y1, x2, y2, score_m, class_m], axis=1))
    if not rows:
        return np.zeros((0, 6), dtype=np.float64)
    boxes = np.concatenate(rows, axis=0)
    keep = _class_aware_nms(boxes, iou_threshold)
    kept: NDArray[Any] = boxes[keep]
    return kept


def _group_ground_truth_by_image_and_class(
    ground_truth: list[_GroundTruth],
) -> dict[tuple[int, int], list[int]]:
    gt_by_image_class: dict[tuple[int, int], list[int]] = {}
    for gt_index, ground in enumerate(ground_truth):
        gt_by_image_class.setdefault((ground[0], ground[1]), []).append(gt_index)
    return gt_by_image_class


def _match(
    predictions: list[_Detection],
    ground_truth: list[_GroundTruth],
    iou_threshold: float,
    *,
    order: list[int] | None = None,
    gt_by_image_class: dict[tuple[int, int], list[int]] | None = None,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Greedy TP/FP assignment in descending-score order; returns ``(tp, scores)``.

    Class-aware even though ``_summarize`` already calls this once per class with
    pre-filtered lists: matching by (image, class) directly here means a caller
    that forgets to pre-filter fails safe instead of silently cross-matching
    classes, mirroring the runtime decoder's own class-aware NMS.

    ``order``/``gt_by_image_class`` are precomputed only by ``_summarize``, which
    calls this once per IoU threshold (ten times) for the same class -- the score
    ranking and ground-truth grouping don't depend on ``iou_threshold``, so a
    caller evaluating multiple thresholds can build them once and reuse them.
    """
    if not predictions:
        return np.zeros(0), np.zeros(0)
    if order is None:
        order = sorted(range(len(predictions)), key=lambda index: -predictions[index][2])
    if gt_by_image_class is None:
        gt_by_image_class = _group_ground_truth_by_image_and_class(ground_truth)
    matched: set[int] = set()
    tp = np.zeros(len(predictions))
    scores = np.zeros(len(predictions))
    for rank, prediction_index in enumerate(order):
        image_index, class_index, score, box = predictions[prediction_index]
        scores[rank] = score
        best_iou, best_gt = 0.0, -1
        for gt_index in gt_by_image_class.get((image_index, class_index), []):
            if gt_index in matched:
                continue
            candidate = _iou(box, ground_truth[gt_index][2])
            if candidate > best_iou:
                best_iou, best_gt = candidate, gt_index
        if best_gt != -1 and best_iou >= iou_threshold:
            tp[rank] = 1.0
            matched.add(best_gt)
    return tp, scores


def _average_precision(
    scores: NDArray[Any], tp: NDArray[Any], num_ground_truth: int
) -> tuple[float, NDArray[Any], NDArray[Any]]:
    if len(scores) == 0:
        empty = np.zeros(0)
        return 0.0, empty, empty
    fp = 1.0 - tp
    cumulative_tp = np.cumsum(tp)
    cumulative_fp = np.cumsum(fp)
    recall = cumulative_tp / max(num_ground_truth, 1)
    precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1e-9)
    average_precision = 0.0
    for point in _RECALL_INTERPOLATION_POINTS:
        reached = recall >= point
        average_precision += float(precision[reached].max()) if reached.any() else 0.0
    average_precision /= len(_RECALL_INTERPOLATION_POINTS)
    return average_precision, precision, recall


def _sample_curve(
    scores: NDArray[Any], curve: NDArray[Any], confidences: NDArray[Any]
) -> NDArray[Any]:
    if len(scores) == 0:
        return np.zeros_like(confidences)
    ascending = scores[::-1]
    counts = len(scores) - np.searchsorted(ascending, confidences, side="left")
    safe_indices = np.clip(counts - 1, 0, len(curve) - 1)
    result: NDArray[Any] = np.where(counts > 0, curve[safe_indices], 0.0)
    return result


def _summarize(
    predictions: list[_Detection], ground_truth: list[_GroundTruth], num_classes: int
) -> Any:
    confidence_axis = np.linspace(0.0, 1.0, _CONFIDENCE_SAMPLES)
    p_curves = np.zeros((num_classes, _CONFIDENCE_SAMPLES))
    r_curves = np.zeros((num_classes, _CONFIDENCE_SAMPLES))
    per_class_map50: list[float] = []
    per_class_map50_95: list[float] = []

    for class_index in range(num_classes):
        class_predictions = [p for p in predictions if p[1] == class_index]
        class_ground_truth = [g for g in ground_truth if g[1] == class_index]
        num_ground_truth = len(class_ground_truth)
        # The score ranking and ground-truth grouping don't depend on
        # iou_threshold; compute them once per class instead of once per
        # (class, threshold) pair -- _match is called ten times below.
        order = sorted(range(len(class_predictions)), key=lambda index: -class_predictions[index][2])
        gt_by_image_class = _group_ground_truth_by_image_and_class(class_ground_truth)
        ap_at_iou: list[float] = []
        ap50 = 0.0
        for iou_threshold in _IOU_THRESHOLDS:
            tp, scores = _match(
                class_predictions,
                class_ground_truth,
                iou_threshold,
                order=order,
                gt_by_image_class=gt_by_image_class,
            )
            average_precision, precision, recall = _average_precision(scores, tp, num_ground_truth)
            ap_at_iou.append(average_precision)
            if iou_threshold == 0.5:
                ap50 = average_precision
                p_curves[class_index] = _sample_curve(scores, precision, confidence_axis)
                r_curves[class_index] = _sample_curve(scores, recall, confidence_axis)
        per_class_map50.append(ap50)
        per_class_map50_95.append(float(np.mean(ap_at_iou)) if ap_at_iou else 0.0)

    map50 = float(np.mean(per_class_map50)) if per_class_map50 else 0.0
    map50_95 = float(np.mean(per_class_map50_95)) if per_class_map50_95 else 0.0
    mean_p_curve = p_curves.mean(axis=0)
    mean_r_curve = r_curves.mean(axis=0)
    f1_curve = 2.0 * mean_p_curve * mean_r_curve / np.maximum(mean_p_curve + mean_r_curve, 1e-9)
    best_index = int(np.argmax(f1_curve))

    results_dict = {
        "metrics/precision(B)": float(mean_p_curve[best_index]),
        "metrics/recall(B)": float(mean_r_curve[best_index]),
        "metrics/mAP50(B)": map50,
        "metrics/mAP50-95(B)": map50_95,
    }
    box = SimpleNamespace(
        maps=per_class_map50_95,
        p_curve=p_curves.tolist(),
        r_curve=r_curves.tolist(),
    )
    return SimpleNamespace(results_dict=results_dict, box=box)


def evaluate(
    model: torch.nn.Module,
    dataset: Dataset[tuple[torch.Tensor, torch.Tensor]],
    anchors: AnchorSet,
    *,
    num_classes: int,
    device: torch.device,
    score_threshold: float = 0.01,
    iou_threshold: float = 0.45,
    batch_size: int = 1,
) -> Any:
    """Run the model over ``dataset`` and return an Ultralytics-shaped validation result."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    predictions: list[_Detection] = []
    ground_truth: list[_GroundTruth] = []
    image_index = 0
    with torch.no_grad():
        for batch_images, targets in loader:
            images = batch_images.to(device)
            outputs = model(images)
            for local_index in range(images.shape[0]):
                scale_outputs = [scale[local_index] for scale in outputs]
                decoded = _decode_predictions(
                    scale_outputs,
                    anchors,
                    score_threshold=score_threshold,
                    iou_threshold=iou_threshold,
                )
                for x1, y1, x2, y2, score, class_value in decoded:
                    box = (x1, y1, x2, y2)
                    predictions.append((image_index, int(class_value), float(score), box))
                rows = targets[targets[:, 0] == local_index]
                for row in rows:
                    class_value, cx, cy, w, h = row[1:].tolist()
                    box = (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)
                    ground_truth.append((image_index, int(class_value), box))
                image_index += 1
    return _summarize(predictions, ground_truth, num_classes)
