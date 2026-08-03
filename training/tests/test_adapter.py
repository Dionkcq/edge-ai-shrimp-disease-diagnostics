"""End-to-end wiring proof: train (1 tiny epoch) -> val -> export -> validate_onnx_contract.

Everything here runs on a synthetic 64x64 dataset on CPU, so it is a correctness/
wiring smoke test -- not a claim about accuracy. That mirrors the rest of this
project's testing philosophy: no real training run has ever completed, and this
suite proves the pipeline is wired correctly, not that a trained model works.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from shrimp_training.adapter import ANCHORS_PER_SCALE, AdapterError, CustomYoloModel, _FlattenHead
from shrimp_training.anchors import compute_anchors
from shrimp_training.artifacts import validate_onnx_contract
from shrimp_training.dataset import PreparedDataset, validate_prepared_dataset
from shrimp_training.model import STRIDES, YOLO

IMAGE_SIZE = 64


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_image_and_label(
    root: Path, partition: str, name: str, *, class_id: int | None
) -> dict[str, str]:
    image_path = root / "images" / partition / f"{name}.jpg"
    label_path = root / "labels" / partition / f"{name}.txt"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    # Each image must have distinct bytes -- validate_prepared_dataset rejects a
    # canonical image hash that appears more than once, and a solid fill colour
    # would otherwise make every synthetic image byte-identical.
    seed = int(hashlib.sha256(f"{partition}/{name}".encode()).hexdigest()[:6], 16)
    color = (seed % 256, (seed // 256) % 256, (seed // 65536) % 256)
    Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=color).save(image_path)
    label_path.write_text(
        "" if class_id is None else f"{class_id} 0.5 0.5 0.3 0.3\n", encoding="utf-8"
    )
    digest = _sha(image_path)
    return {
        "specimen_key": f"{partition}:{name}",
        "sha256": digest,
        "partition": partition,
        "image_output": image_path.relative_to(root).as_posix(),
        "image_output_sha256": digest,
        "label_output": label_path.relative_to(root).as_posix(),
        "label_output_sha256": _sha(label_path),
    }


def _build_prepared_dataset(root: Path) -> PreparedDataset:
    """A minimal but contract-valid prepared dataset: enough train boxes for
    k-means (>= 9), and every partition covering both classes plus a negative."""
    records: list[dict[str, str]] = []
    for index in range(5):
        records.append(_write_image_and_label(root, "train", f"dark-{index}", class_id=0))
        records.append(_write_image_and_label(root, "train", f"spot-{index}", class_id=1))
    records.append(_write_image_and_label(root, "train", "healthy", class_id=None))
    for partition in ("validation", "test"):
        records.append(_write_image_and_label(root, partition, "dark", class_id=0))
        records.append(_write_image_and_label(root, partition, "spot", class_id=1))
        records.append(_write_image_and_label(root, partition, "healthy", class_id=None))

    counts = {
        "train": sum(1 for r in records if r["partition"] == "train"),
        "validation": sum(1 for r in records if r["partition"] == "validation"),
        "test": sum(1 for r in records if r["partition"] == "test"),
    }
    manifest = {
        "schema_version": "1.0.0",
        "classes": {"0": "dark_gill", "1": "white_spot"},
        "split": {"seed": 20260730, "counts": counts},
        "summary": {"canonical_images": len(records)},
        "images": records,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return validate_prepared_dataset(root)


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory) -> PreparedDataset:
    return _build_prepared_dataset(tmp_path_factory.mktemp("prepared"))


def test_train_val_export_round_trip(tmp_path: Path, prepared: PreparedDataset) -> None:
    anchors = compute_anchors(prepared, seed=20260730)
    assert len(anchors.boxes) == 9

    model = CustomYoloModel()
    train_result = model.train(
        data=str(prepared.root),
        imgsz=IMAGE_SIZE,
        epochs=1,
        patience=0,
        batch=2,
        workers=0,
        device=0,
        amp=False,
        deterministic=True,
        seed=20260730,
        cache=False,
        learning_rate=0.001,
        weight_decay=0.0,
        anchors=anchors,
        project=str(tmp_path / "runs"),
        name="smoke",
    )

    best_checkpoint = Path(train_result.save_dir) / "weights" / "best.pt"
    assert best_checkpoint.is_file()

    loaded = CustomYoloModel(weights=best_checkpoint)
    val_result = loaded.val(
        data=str(prepared.root),
        split="test",
        imgsz=IMAGE_SIZE,
        anchors=anchors,
    )
    assert set(val_result.results_dict) == {
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    }
    assert len(val_result.box.maps) == 2

    onnx_path = tmp_path / "model.onnx"
    exported = loaded.export(destination=onnx_path, imgsz=IMAGE_SIZE, opset=17)
    assert exported == onnx_path
    assert onnx_path.is_file()

    validate_onnx_contract(onnx_path, IMAGE_SIZE)

    # Refuses to clobber an existing export target.
    with pytest.raises(Exception, match="overwrite"):
        loaded.export(destination=onnx_path, imgsz=IMAGE_SIZE, opset=17)

    # An ONNX-backed model can also be evaluated (a different code path than torch eval).
    onnx_model = CustomYoloModel(weights=onnx_path)
    onnx_result = onnx_model.val(
        data=str(prepared.root),
        split="test",
        imgsz=IMAGE_SIZE,
        anchors=anchors,
    )
    assert len(onnx_result.box.maps) == 2


def test_checkpoint_round_trips_its_own_anchors(tmp_path: Path, prepared: PreparedDataset) -> None:
    """A loaded checkpoint carries the anchors it was trained with, so a caller
    does not have to remember to pass them again for evaluation."""
    anchors = compute_anchors(prepared, seed=1)
    model = CustomYoloModel()
    train_result = model.train(
        data=str(prepared.root),
        imgsz=IMAGE_SIZE,
        epochs=1,
        patience=0,
        batch=2,
        workers=0,
        device=0,
        amp=False,
        deterministic=True,
        seed=1,
        cache=False,
        learning_rate=0.001,
        weight_decay=0.0,
        anchors=anchors,
        project=str(tmp_path / "runs"),
        name="anchor-roundtrip",
    )
    checkpoint = Path(train_result.save_dir) / "weights" / "best.pt"

    loaded = CustomYoloModel(weights=checkpoint)
    result = loaded.val(data=str(prepared.root), split="validation", imgsz=IMAGE_SIZE)
    assert set(result.results_dict) == {
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    }


def test_anchors_are_required_when_none_are_available(prepared: PreparedDataset) -> None:
    model = CustomYoloModel()
    with pytest.raises(AdapterError, match="anchors"):
        model.val(data=str(prepared.root), split="test", imgsz=IMAGE_SIZE)


def test_flatten_head_output_matches_the_runtime_contract_shape() -> None:
    """(1, 5 + nc, anchors) channels-then-anchors, matching onnx_provider's expectation."""
    model = YOLO(num_classes=2, num_anchors=3)
    wrapper = _FlattenHead(model)
    dummy = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)

    output = wrapper(dummy)

    expected_anchors = ANCHORS_PER_SCALE * sum((IMAGE_SIZE // stride) ** 2 for stride in STRIDES)
    assert output.shape == (1, 7, expected_anchors)
