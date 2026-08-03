from __future__ import annotations

import json
from pathlib import Path

import pytest

from shrimp_training.anchors import (
    ANCHORS_PER_SCALE,
    AnchorError,
    anchor_set_to_document,
    compute_anchors,
)
from shrimp_training.dataset import PreparedDataset


def _prepared(root: Path, *, box_count: int = 30) -> PreparedDataset:
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "manifest.json"
    images = []
    for index in range(box_count):
        label = root / "labels" / f"{index}.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        # Vary width/height so k-means has more than one cluster to find.
        width = 0.05 + 0.01 * (index % 5)
        height = 0.05 + 0.01 * (index % 5)
        label.write_text(f"0 0.5 0.5 {width} {height}\n", encoding="utf-8")
        images.append(
            {
                "partition": "train",
                "label_output": label.relative_to(root).as_posix(),
            }
        )
    manifest.write_text(json.dumps({"images": images}), encoding="utf-8")
    return PreparedDataset(
        root=root,
        manifest=manifest,
        manifest_sha256="a" * 64,
        inventory_sha256="b" * 64,
        image_count=box_count,
        specimen_count=box_count,
        split_counts={"train": box_count, "validation": 0, "test": 0},
        class_names={0: "dark_gill", 1: "white_spot"},
    )


def test_compute_anchors_returns_nine_sorted_positive_boxes(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)

    anchors = compute_anchors(prepared, seed=20260730)

    assert len(anchors.boxes) == 9
    areas = [width * height for width, height in anchors.boxes]
    assert areas == sorted(areas)
    assert all(width > 0 and height > 0 for width, height in anchors.boxes)
    assert anchors.source_box_count == 30
    assert anchors.strides == (8, 16, 32)


def test_compute_anchors_is_deterministic_given_a_seed(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    first = compute_anchors(prepared, seed=1)
    second = compute_anchors(prepared, seed=1)
    assert first.boxes == second.boxes


def test_for_stride_returns_three_anchors_per_scale(tmp_path: Path) -> None:
    anchors = compute_anchors(_prepared(tmp_path), seed=20260730)
    for stride in anchors.strides:
        assert len(anchors.for_stride(stride)) == ANCHORS_PER_SCALE
    assert anchors.for_stride(8) == anchors.boxes[0:3]
    assert anchors.for_stride(16) == anchors.boxes[3:6]
    assert anchors.for_stride(32) == anchors.boxes[6:9]


def test_too_few_boxes_is_refused(tmp_path: Path) -> None:
    with pytest.raises(AnchorError, match="at least 9"):
        compute_anchors(_prepared(tmp_path, box_count=3), seed=20260730)


def test_non_train_boxes_are_ignored(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    document = json.loads(prepared.manifest.read_text(encoding="utf-8"))
    for record in document["images"]:
        record["partition"] = "validation"
    prepared.manifest.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(AnchorError, match="at least 9"):
        compute_anchors(prepared, seed=20260730)


def test_anchor_set_to_document_round_trips_through_json(tmp_path: Path) -> None:
    anchors = compute_anchors(_prepared(tmp_path), seed=20260730)

    document = anchor_set_to_document(anchors)

    assert json.loads(json.dumps(document)) == document
    assert document["boxes"] == [list(box) for box in anchors.boxes]
    assert document["source_box_count"] == anchors.source_box_count
