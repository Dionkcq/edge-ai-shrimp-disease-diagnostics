from __future__ import annotations

import torch

from shrimp_training.model import STRIDES, YOLO


def test_forward_produces_three_scales_with_expected_grid_and_channel_shapes() -> None:
    model = YOLO(num_classes=2, num_anchors=3)
    dummy = torch.zeros(2, 3, 640, 640)

    outputs = model(dummy)

    assert len(outputs) == 3
    assert STRIDES == (8, 16, 32)
    for output, stride in zip(outputs, STRIDES, strict=True):
        grid = 640 // stride
        assert output.shape == (2, 3, grid, grid, 7)


def test_forward_is_fully_convolutional_and_works_at_a_smaller_input_size() -> None:
    """Input size is a config choice, not an architecture change."""
    model = YOLO(num_classes=2, num_anchors=3)
    dummy = torch.zeros(1, 3, 64, 64)

    outputs = model(dummy)

    for output, stride in zip(outputs, STRIDES, strict=True):
        grid = 64 // stride
        assert output.shape == (1, 3, grid, grid, 7)


def test_parameter_count_is_finite_and_positive() -> None:
    model = YOLO(num_classes=2, num_anchors=3)
    total = sum(parameter.numel() for parameter in model.parameters())
    assert total > 0


def test_objectness_and_class_bias_start_negative_so_the_model_predicts_mostly_empty() -> None:
    """Without this init, early training is dominated by false positives."""
    model = YOLO(num_classes=2, num_anchors=3)
    for head in (model.head_p3, model.head_p4, model.head_p5):
        final = head[-1]
        assert isinstance(final, torch.nn.Conv2d)
        assert final.bias is not None
        bias = final.bias.view(model.na, -1)
        assert bool((bias[:, 4] < 0).all())
        assert bool((bias[:, 5:] < 0).all())
