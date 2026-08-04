from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from shrimp_training.anchors import AnchorSet
from shrimp_training.data import ShrimpDetectionDataset
from shrimp_training.dataset import PreparedDataset
from shrimp_training.evaluate import _average_precision, _match, _summarize, evaluate
from shrimp_training.model import STRIDES, YOLO

IMAGE_SIZE = 64


def _anchors() -> AnchorSet:
    boxes = tuple((5.0 * (index + 1), 5.0 * (index + 1)) for index in range(9))
    return AnchorSet(boxes=boxes, strides=STRIDES, seed=20260730, source_box_count=100)


# ---------------------------------------------------------------------------
# Pure-numpy pieces, tested directly against synthetic predictions/ground truth.
# ---------------------------------------------------------------------------


def test_average_precision_is_one_for_a_perfectly_ranked_set() -> None:
    scores = np.array([0.9, 0.8, 0.7])
    true_positive = np.array([1.0, 1.0, 1.0])

    average_precision, precision, recall = _average_precision(scores, true_positive, 3)

    assert average_precision == 1.0
    assert recall[-1] == 1.0
    assert precision[-1] == 1.0


def test_average_precision_is_zero_with_no_predictions() -> None:
    average_precision, precision, recall = _average_precision(np.zeros(0), np.zeros(0), 3)
    assert average_precision == 0.0
    assert len(precision) == 0
    assert len(recall) == 0


def test_match_assigns_true_positive_only_to_the_higher_scoring_duplicate() -> None:
    predictions = [
        (0, 0, 0.9, (0.0, 0.0, 10.0, 10.0)),
        (0, 0, 0.8, (0.0, 0.0, 10.0, 10.0)),
    ]
    ground_truth = [(0, 0, (0.0, 0.0, 10.0, 10.0))]

    true_positive, scores = _match(predictions, ground_truth, iou_threshold=0.5)

    assert list(true_positive) == [1.0, 0.0]
    assert list(scores) == [0.9, 0.8]


def test_match_ignores_a_different_class_ground_truth() -> None:
    predictions = [(0, 0, 0.9, (0.0, 0.0, 10.0, 10.0))]
    ground_truth = [(0, 1, (0.0, 0.0, 10.0, 10.0))]

    true_positive, _ = _match(predictions, ground_truth, iou_threshold=0.5)

    assert list(true_positive) == [0.0]


def test_summarize_produces_finite_metrics_for_a_perfect_prediction_set() -> None:
    predictions = [
        (0, 0, 0.9, (0.0, 0.0, 10.0, 10.0)),
        (0, 1, 0.9, (20.0, 20.0, 30.0, 30.0)),
    ]
    ground_truth = [
        (0, 0, (0.0, 0.0, 10.0, 10.0)),
        (0, 1, (20.0, 20.0, 30.0, 30.0)),
    ]

    result = _summarize(predictions, ground_truth, num_classes=2)

    assert result.results_dict["metrics/mAP50(B)"] == 1.0
    assert len(result.box.maps) == 2
    assert all(np.isfinite(value) for value in result.box.maps)


def test_summarize_handles_no_predictions_or_ground_truth() -> None:
    result = _summarize([], [], num_classes=2)
    assert result.results_dict["metrics/mAP50(B)"] == 0.0
    assert result.box.maps == [0.0, 0.0]


# ---------------------------------------------------------------------------
# End to end: a real (untrained) model over a tiny real dataset.
# ---------------------------------------------------------------------------


def _prepared_with_real_images(root: Path, *, count: int = 2) -> PreparedDataset:
    manifest = root / "manifest.json"
    images: list[dict[str, object]] = []
    for index in range(count):
        image_path = root / "images" / f"{index}.jpg"
        label_path = root / "labels" / f"{index}.txt"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=(120, 120, 120)).save(image_path)
        label_path.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        images.append(
            {
                "partition": "validation",
                "image_output": image_path.relative_to(root).as_posix(),
                "label_output": label_path.relative_to(root).as_posix(),
            }
        )
    manifest.write_text(json.dumps({"images": images}), encoding="utf-8")
    return PreparedDataset(
        root=root,
        manifest=manifest,
        manifest_sha256="a" * 64,
        inventory_sha256="b" * 64,
        image_count=count,
        specimen_count=count,
        split_counts={"train": 0, "validation": count, "test": 0},
        class_names={0: "dark_gill", 1: "white_spot"},
    )


def test_evaluate_runs_end_to_end_on_a_tiny_synthetic_dataset(tmp_path: Path) -> None:
    prepared = _prepared_with_real_images(tmp_path)
    dataset = ShrimpDetectionDataset(prepared, "validation", image_size=IMAGE_SIZE, augment=False)
    model = YOLO(num_classes=2, num_anchors=3)

    result = evaluate(
        model, dataset, _anchors(), num_classes=2, device=torch.device("cpu"), batch_size=2
    )

    assert set(result.results_dict) == {
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    }
    assert all(np.isfinite(value) for value in result.results_dict.values())
    assert len(result.box.maps) == 2
    assert all(np.isfinite(value) for value in result.box.maps)
