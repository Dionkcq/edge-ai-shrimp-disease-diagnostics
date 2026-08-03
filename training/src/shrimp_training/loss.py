# SPDX-License-Identifier: AGPL-3.0-or-later
"""Multi-scale YOLOv2/v3-style detection loss for ``model.YOLO``'s anchor-based head.

Matches ``model.py``'s ``[tx, ty, tw, th, objectness]`` parametrization: each
ground-truth box is assigned to its best-shape-IoU anchor (both box shapes
centered at the origin) at the scale/grid-cell containing its center. Anchors
whose shape-IoU with some ground-truth box exceeds ``ignore_iou`` but were not the
best match are excluded from the objectness-negative loss (the classic "ignore
band"), so a decent-but-not-best anchor is not punished for correctly noticing an
object nearby.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional

from shrimp_training.anchors import AnchorSet
from shrimp_training.model import STRIDES


@dataclass(frozen=True, slots=True)
class LossWeights:
    box: float = 5.0
    objectness: float = 1.0
    no_objectness: float = 0.5
    classification: float = 1.0


def _shape_iou(box_wh: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    """IoU between one box shape and a set of anchor shapes, both centered at the origin."""
    intersection = torch.min(box_wh[0], anchors[:, 0]) * torch.min(box_wh[1], anchors[:, 1])
    union = box_wh[0] * box_wh[1] + anchors[:, 0] * anchors[:, 1] - intersection
    result: torch.Tensor = intersection / torch.clamp(union, min=1e-9)
    return result


class YoloLoss(nn.Module):
    """Call with ``(predictions, targets)`` where ``predictions`` is the model's raw
    3-scale output list and ``targets`` is ``(n, 6)`` rows of
    ``[batch_index, class, cx, cy, w, h]`` in absolute 640-canvas pixels
    (``data.collate_fn``'s output shape).
    """

    def __init__(
        self,
        anchors: AnchorSet,
        num_classes: int,
        *,
        ignore_iou: float = 0.5,
        weights: LossWeights | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_iou = ignore_iou
        self.weights = weights or LossWeights()
        self._anchors = [
            torch.tensor(anchors.for_stride(stride), dtype=torch.float32) for stride in STRIDES
        ]

    def forward(
        self, predictions: list[torch.Tensor], targets: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Each target is assigned to exactly one (scale, anchor, cell): the globally
        best-shape-IoU match across *all* scales, not the best match *within* every
        scale independently -- otherwise one object would be a positive example at
        all three scales at once, which both triples its weight in the loss and
        forces a scale whose anchors are a poor shape match to chase it anyway.
        """
        device = predictions[0].device
        num_scales = len(STRIDES)
        batch = predictions[0].shape[0]
        # Fixed 3x2 anchor tensors, loop-invariant: moved once instead of once per
        # target row (previously re-transferred to `device` on every target *and*
        # every scale inside the loop below).
        anchors_by_scale = [anchors.to(device) for anchors in self._anchors]

        obj_targets = [
            torch.zeros(prediction.shape[:4], device=device) for prediction in predictions
        ]
        ignore_masks = [torch.zeros_like(target, dtype=torch.bool) for target in obj_targets]
        positions: list[list[tuple[int, int, int, int]]] = [[] for _ in range(num_scales)]
        box_rows: list[list[torch.Tensor]] = [[] for _ in range(num_scales)]
        class_rows: list[list[int]] = [[] for _ in range(num_scales)]

        for row in targets:
            image_index = int(row[0].item())
            if image_index >= batch:
                continue
            cls, cx, cy, w, h = row[1:].tolist()
            box_wh = torch.tensor([w, h], device=device)

            per_scale_ious = [
                _shape_iou(box_wh, anchors_by_scale[scale_index])
                for scale_index in range(num_scales)
            ]
            per_scale_best = [int(torch.argmax(ious).item()) for ious in per_scale_ious]
            best_scale = max(
                range(num_scales),
                key=lambda index: float(per_scale_ious[index][per_scale_best[index]]),
            )
            best_anchor = per_scale_best[best_scale]

            for scale_index in range(num_scales):
                stride = STRIDES[scale_index]
                prediction_shape = predictions[scale_index].shape
                grid_h, grid_w = prediction_shape[2], prediction_shape[3]
                grid_x = max(0, min(int(cx / stride), grid_w - 1))
                grid_y = max(0, min(int(cy / stride), grid_h - 1))
                if scale_index == best_scale:
                    positions[scale_index].append((image_index, best_anchor, grid_y, grid_x))
                    box_rows[scale_index].append(torch.tensor([cx, cy, w, h], device=device))
                    class_rows[scale_index].append(int(cls))
                    obj_targets[scale_index][image_index, best_anchor, grid_y, grid_x] = 1.0

                near_miss = (per_scale_ious[scale_index] > self.ignore_iou).clone()
                if scale_index == best_scale:
                    near_miss[best_anchor] = False
                near_indices = near_miss.nonzero(as_tuple=True)[0]
                if near_indices.numel():
                    ignore_masks[scale_index][image_index, near_indices, grid_y, grid_x] = True

        total_box = torch.zeros((), device=device)
        total_obj = torch.zeros((), device=device)
        total_noobj = torch.zeros((), device=device)
        total_cls = torch.zeros((), device=device)
        num_positive = 0

        for scale_index, (prediction, stride) in enumerate(zip(predictions, STRIDES, strict=True)):
            anchors = anchors_by_scale[scale_index]
            scale_positions = positions[scale_index]
            if scale_positions:
                idx_b = torch.tensor([p[0] for p in scale_positions], device=device)
                idx_a = torch.tensor([p[1] for p in scale_positions], device=device)
                idx_y = torch.tensor([p[2] for p in scale_positions], device=device)
                idx_x = torch.tensor([p[3] for p in scale_positions], device=device)

                selected = prediction[idx_b, idx_a, idx_y, idx_x]  # (P, 5 + nc)
                selected_anchor = anchors[idx_a]  # (P, 2)
                target_box = torch.stack(box_rows[scale_index])  # (P, 4): cx, cy, w, h pixels

                target_tx = target_box[:, 0] / stride - idx_x.float()
                target_ty = target_box[:, 1] / stride - idx_y.float()
                width_ratio = target_box[:, 2] / selected_anchor[:, 0]
                height_ratio = target_box[:, 3] / selected_anchor[:, 1]
                target_tw = torch.log(torch.clamp(width_ratio, min=1e-6))
                target_th = torch.log(torch.clamp(height_ratio, min=1e-6))

                total_box = total_box + functional.mse_loss(
                    torch.sigmoid(selected[:, 0]), target_tx, reduction="sum"
                )
                total_box = total_box + functional.mse_loss(
                    torch.sigmoid(selected[:, 1]), target_ty, reduction="sum"
                )
                total_box = total_box + functional.mse_loss(
                    selected[:, 2], target_tw, reduction="sum"
                )
                total_box = total_box + functional.mse_loss(
                    selected[:, 3], target_th, reduction="sum"
                )

                class_index = torch.tensor(class_rows[scale_index], device=device, dtype=torch.long)
                class_target = functional.one_hot(class_index, num_classes=self.num_classes).float()
                total_cls = total_cls + functional.binary_cross_entropy_with_logits(
                    selected[:, 5:], class_target, reduction="sum"
                )
                total_obj = total_obj + functional.binary_cross_entropy_with_logits(
                    selected[:, 4], torch.ones_like(selected[:, 4]), reduction="sum"
                )
                num_positive += len(scale_positions)

            negative_mask = (obj_targets[scale_index] == 0) & (~ignore_masks[scale_index])
            negative_logits = prediction[..., 4][negative_mask]
            if negative_logits.numel():
                # Mean, not sum: a scale has tens of thousands of negative anchor
                # positions against a handful of positives. Summing and then
                # dividing by `num_positive` (correct for the positive-only terms
                # below) would scale the background term by anchors/positives --
                # often four orders of magnitude -- and dominate the gradient.
                total_noobj = total_noobj + functional.binary_cross_entropy_with_logits(
                    negative_logits, torch.zeros_like(negative_logits), reduction="mean"
                )

        denom = float(max(num_positive, 1))
        mean_noobj = total_noobj / num_scales
        loss = (
            self.weights.box * total_box / denom
            + self.weights.objectness * total_obj / denom
            + self.weights.no_objectness * mean_noobj
            + self.weights.classification * total_cls / denom
        )
        return {
            "loss": loss,
            "box_loss": total_box / denom,
            "objectness_loss": total_obj / denom,
            "no_objectness_loss": mean_noobj,
            "classification_loss": total_cls / denom,
            "num_positive": torch.tensor(float(num_positive)),
        }
