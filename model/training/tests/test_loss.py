from __future__ import annotations

import math

import torch

from shrimp_training.anchors import AnchorSet
from shrimp_training.loss import YoloLoss
from shrimp_training.model import STRIDES, YOLO

IMAGE_SIZE = 64


def _anchors() -> AnchorSet:
    boxes = tuple((5.0 * (index + 1), 5.0 * (index + 1)) for index in range(9))
    return AnchorSet(boxes=boxes, strides=STRIDES, seed=20260730, source_box_count=100)


def test_loss_is_finite_and_positive_on_a_synthetic_batch() -> None:
    model = YOLO(num_classes=2, num_anchors=3)
    loss_fn = YoloLoss(_anchors(), num_classes=2)
    images = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    # One box: image 0, class 1, center (20, 20), size (10, 10).
    targets = torch.tensor([[0.0, 1.0, 20.0, 20.0, 10.0, 10.0]])

    outputs = model(images)
    losses = loss_fn(outputs, targets)

    assert torch.isfinite(losses["loss"])
    assert losses["loss"].item() > 0
    assert losses["num_positive"].item() == 1


def test_loss_handles_an_empty_target_batch_as_all_background() -> None:
    model = YOLO(num_classes=2, num_anchors=3)
    loss_fn = YoloLoss(_anchors(), num_classes=2)
    images = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    targets = torch.zeros((0, 6))

    outputs = model(images)
    losses = loss_fn(outputs, targets)

    assert torch.isfinite(losses["loss"])
    assert losses["num_positive"].item() == 0
    assert losses["objectness_loss"].item() == 0.0
    assert losses["box_loss"].item() == 0.0
    assert losses["classification_loss"].item() == 0.0
    assert losses["no_objectness_loss"].item() > 0


def test_loss_ignores_a_near_miss_anchor_instead_of_punishing_it() -> None:
    """An anchor whose shape-IoU exceeds the ignore band, but isn't the best match,
    must not be pushed toward "no object" -- it may legitimately be seeing the box."""
    model = YOLO(num_classes=2, num_anchors=3)
    # Two nearly-identical anchors at the same scale (indices 0 and 1).
    boxes = (
        (10.0, 10.0),
        (10.5, 10.5),
        (30.0, 30.0),
        (20.0, 20.0),
        (25.0, 25.0),
        (35.0, 35.0),
        (40.0, 40.0),
        (45.0, 45.0),
        (50.0, 50.0),
    )
    anchors = AnchorSet(boxes=boxes, strides=STRIDES, seed=20260730, source_box_count=100)
    loss_fn = YoloLoss(anchors, num_classes=2, ignore_iou=0.5)
    images = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    targets = torch.tensor([[0.0, 0.0, 20.0, 20.0, 10.0, 10.0]])

    outputs = model(images)
    losses = loss_fn(outputs, targets)

    assert torch.isfinite(losses["loss"])


def test_loss_decreases_when_overfitting_one_batch() -> None:
    """A classic training-loop sanity check: loss must go down on a single example.

    Batch size 2 (not 1): BatchNorm's per-batch variance is degenerate at batch
    size 1, which is a numerical-stability footgun independent of the loss itself.
    Gradient clipping mirrors what the real training loop in adapter.py does.
    """
    torch.manual_seed(0)
    model = YOLO(num_classes=2, num_anchors=3)
    loss_fn = YoloLoss(_anchors(), num_classes=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    images = torch.rand(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    targets = torch.tensor(
        [
            [0.0, 1.0, 20.0, 20.0, 10.0, 10.0],
            [1.0, 0.0, 30.0, 30.0, 12.0, 12.0],
        ]
    )

    losses_seen: list[float] = []
    for _ in range(15):
        optimizer.zero_grad()
        outputs = model(images)
        step_losses = loss_fn(outputs, targets)
        loss = step_losses["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        losses_seen.append(loss.item())

    assert all(not math.isnan(value) for value in losses_seen)
    assert losses_seen[-1] < losses_seen[0]
