from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import pytest

from shrimp_training import artifacts
from shrimp_training.artifacts import (
    ArtifactError,
    compare_parity,
    evaluate_artifact,
    export_static_onnx,
    validate_onnx_contract,
)
from shrimp_training.config import TrainingProfile


def _profile() -> TrainingProfile:
    return TrainingProfile(
        profile_name="compact-nvidia-6gb",
        image_size=640,
        epochs=100,
        patience=20,
        batch_fallback=(4, 2, 1),
        workers=4,
        device=0,
        amp=True,
        deterministic=True,
        seed=20260730,
        cache=False,
    )


def _matched(precision: float, recall: float) -> dict[str, float]:
    """Build a threshold_matched_metrics block flat across every compared confidence."""
    values = {"precision": precision, "recall": recall}
    return {
        f"{metric}@{confidence:.2f}": values[metric]
        for metric, (_, confidences) in artifacts._CURVE_KEYS.items()
        for confidence in confidences
    }


def _bind_dataset(
    monkeypatch: pytest.MonkeyPatch,
    prepared_root: Path,
    *,
    inventories: tuple[str, str] = ("d" * 64, "d" * 64),
) -> list[Path]:
    calls: list[Path] = []
    snapshots = iter(inventories)

    def validate(path: Path) -> SimpleNamespace:
        calls.append(path)
        return SimpleNamespace(
            root=prepared_root.resolve(),
            manifest_sha256="c" * 64,
            inventory_sha256=next(snapshots),
            split_counts={"train": 3, "validation": 3, "test": 3},
        )

    monkeypatch.setattr(artifacts, "validate_prepared_dataset", validate, raising=False)
    return calls


def _flat_curve(value: float) -> list[list[float]]:
    """Two per-class rows over Ultralytics' 1000-point confidence axis."""
    return [[value] * 1000, [value] * 1000]


class _Box:
    maps: ClassVar[list[float]] = [0.21, 0.34]
    p_curve: ClassVar[list[list[float]]] = _flat_curve(0.5)
    r_curve: ClassVar[list[list[float]]] = _flat_curve(0.4)


class _Validation:
    results_dict: ClassVar[dict[str, float]] = {
        "metrics/precision(B)": 0.5,
        "metrics/recall(B)": 0.4,
        "metrics/mAP50(B)": 0.42,
        "metrics/mAP50-95(B)": 0.275,
        "fitness": 0.3,
    }
    box = _Box()


class _FakeModel:
    def __init__(self, artifact: Path) -> None:
        self.artifact = artifact
        self.val_kwargs: dict[str, Any] = {}
        self.export_kwargs: dict[str, Any] = {}

    def val(self, **kwargs: Any) -> _Validation:
        self.val_kwargs = kwargs
        return _Validation()

    def export(self, **kwargs: Any) -> str:
        self.export_kwargs = kwargs
        exported = self.artifact.with_suffix(".onnx")
        exported.write_bytes(b"onnx")
        return str(exported)


def test_evaluate_writes_normalized_finite_metrics_and_uses_locked_test_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "best.pt"
    artifact.write_bytes(b"pt")
    dataset = tmp_path / "dataset.yaml"
    dataset.write_text("names: {}\n", encoding="utf-8")
    destination = tmp_path / "evaluation.json"
    prepared_root = tmp_path / "prepared"
    calls = _bind_dataset(monkeypatch, prepared_root)
    models: list[_FakeModel] = []

    summary = evaluate_artifact(
        artifact,
        dataset,
        _profile(),
        destination,
        prepared_root=prepared_root,
        model_factory=lambda path: models.append(_FakeModel(path)) or models[-1],
    )

    assert summary.per_class_map50_95 == (0.21, 0.34)
    assert summary.metrics["map50_95"] == 0.275
    assert models[0].val_kwargs["split"] == "test"
    assert models[0].val_kwargs["batch"] == 1
    assert calls == [prepared_root, prepared_root]
    document = json.loads(destination.read_text())
    assert document["artifact_sha256"]
    assert document["dataset_descriptor_sha256"]
    assert document["prepared_manifest_sha256"] == "c" * 64
    assert document["dataset_inventory_sha256"] == "d" * 64
    assert document["test_image_count"] == 3
    assert document["split"] == "test"


def test_evaluate_rejects_metrics_outside_unit_interval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "best.pt"
    artifact.write_bytes(b"pt")
    dataset = tmp_path / "dataset.yaml"
    dataset.write_text("names: {}\n", encoding="utf-8")
    prepared_root = tmp_path / "prepared"
    _bind_dataset(monkeypatch, prepared_root)
    invalid = {**_Validation.results_dict, "metrics/precision(B)": 1.01}
    monkeypatch.setattr(_Validation, "results_dict", invalid)

    models: list[_FakeModel] = []
    with pytest.raises(ArtifactError, match=r"within.*0, 1"):
        evaluate_artifact(
            artifact,
            dataset,
            _profile(),
            tmp_path / "evaluation.json",
            prepared_root=prepared_root,
            model_factory=lambda path: models.append(_FakeModel(path)) or models[-1],
        )


def test_evaluate_accepts_ultralytics_numpy_per_class_maps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "best.pt"
    artifact.write_bytes(b"pt")
    dataset = tmp_path / "dataset.yaml"
    dataset.write_text("names: {}\n", encoding="utf-8")
    prepared_root = tmp_path / "prepared"
    _bind_dataset(monkeypatch, prepared_root)
    monkeypatch.setattr(_Box, "maps", np.array([0.21, 0.34]))

    models: list[_FakeModel] = []
    summary = evaluate_artifact(
        artifact,
        dataset,
        _profile(),
        tmp_path / "evaluation.json",
        prepared_root=prepared_root,
        model_factory=lambda path: models.append(_FakeModel(path)) or models[-1],
    )

    assert summary.per_class_map50_95 == pytest.approx((0.21, 0.34))


def test_evaluate_rejects_dataset_mutation_during_model_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "best.pt"
    artifact.write_bytes(b"pt")
    dataset = tmp_path / "dataset.yaml"
    dataset.write_text("names: {}\n", encoding="utf-8")
    prepared_root = tmp_path / "prepared"
    _bind_dataset(monkeypatch, prepared_root, inventories=("d" * 64, "e" * 64))
    destination = tmp_path / "evaluation.json"
    models: list[_FakeModel] = []

    with pytest.raises(ArtifactError, match="changed during evaluation"):
        evaluate_artifact(
            artifact,
            dataset,
            _profile(),
            destination,
            prepared_root=prepared_root,
            model_factory=lambda path: models.append(_FakeModel(path)) or models[-1],
        )

    assert not destination.exists()


def test_compare_parity_accepts_small_deltas_and_rejects_drift(tmp_path: Path) -> None:
    pt = tmp_path / "pt.json"
    onnx = tmp_path / "onnx.json"
    baseline = {
        "schema_version": "1.1.0",
        "artifact_sha256": "a" * 64,
        "dataset_descriptor_sha256": "c" * 64,
        "prepared_manifest_sha256": "d" * 64,
        "dataset_inventory_sha256": "e" * 64,
        "test_image_count": 3,
        "split": "test",
        "metrics": {"precision": 0.5, "recall": 0.4, "map50": 0.42, "map50_95": 0.275},
        "per_class_map50_95": [0.21, 0.34],
        "threshold_matched_metrics": _matched(0.5, 0.4),
    }
    pt.write_text(json.dumps(baseline), encoding="utf-8")
    close = dict(baseline)
    close["artifact_sha256"] = "b" * 64
    close["metrics"] = {**baseline["metrics"], "map50_95": 0.27}
    onnx.write_text(json.dumps(close), encoding="utf-8")

    report = compare_parity(pt, onnx, tmp_path / "parity.json", tolerance=0.01)
    assert report.passed is True
    parity_document = json.loads((tmp_path / "parity.json").read_text())
    assert (
        parity_document["pytorch_evaluation_sha256"] == hashlib.sha256(pt.read_bytes()).hexdigest()
    )
    assert (
        parity_document["onnx_evaluation_sha256"] == hashlib.sha256(onnx.read_bytes()).hexdigest()
    )
    assert parity_document["prepared_manifest_sha256"] == "d" * 64
    assert parity_document["dataset_inventory_sha256"] == "e" * 64

    close["dataset_descriptor_sha256"] = "d" * 64
    onnx.write_text(json.dumps(close), encoding="utf-8")
    with pytest.raises(ArtifactError, match="different dataset descriptors"):
        compare_parity(pt, onnx, tmp_path / "parity-other-data.json", tolerance=0.01)

    close["dataset_descriptor_sha256"] = "c" * 64
    close["metrics"] = {**baseline["metrics"], "map50_95": 0.20}
    onnx.write_text(json.dumps(close), encoding="utf-8")
    with pytest.raises(ArtifactError, match="parity"):
        compare_parity(pt, onnx, tmp_path / "parity-failed.json", tolerance=0.01)


def test_precision_is_only_compared_where_enough_predictions_survive() -> None:
    """Precision's denominator is the prediction count, which collapses at high confidence.

    On the reference test split roughly 20 predictions survive at confidence 0.50, so one
    differing detection moves precision by about 0.05 -- five times the parity tolerance.
    Recall's denominator is the fixed ground-truth count, so it stays comparable throughout.
    Extending precision into the sparse tail would make the gate measure sampling noise.
    """
    _, precision_confidences = artifacts._CURVE_KEYS["precision"]
    _, recall_confidences = artifacts._CURVE_KEYS["recall"]
    assert max(precision_confidences) <= 0.20
    assert set(precision_confidences) <= set(recall_confidences)
    assert max(recall_confidences) >= 0.50


def test_parity_ignores_reported_operating_point_but_catches_matched_divergence(
    tmp_path: Path,
) -> None:
    """Reported precision/recall come from each artifact's own max-F1 confidence.

    Two faithful artifacts can select different confidences and report very different
    precision/recall while behaving identically, so those reported values must not fail
    the gate. Divergence at a shared confidence is real and must still fail it.
    """
    baseline = {
        "schema_version": "1.1.0",
        "artifact_sha256": "a" * 64,
        "dataset_descriptor_sha256": "c" * 64,
        "prepared_manifest_sha256": "d" * 64,
        "dataset_inventory_sha256": "e" * 64,
        "test_image_count": 3,
        "split": "test",
        "metrics": {"precision": 0.148, "recall": 0.260, "map50": 0.42, "map50_95": 0.275},
        "per_class_map50_95": [0.21, 0.34],
        "threshold_matched_metrics": _matched(0.5, 0.4),
    }
    pt = tmp_path / "pt.json"
    onnx = tmp_path / "onnx.json"
    pt.write_text(json.dumps(baseline), encoding="utf-8")

    # Same behaviour at every shared confidence, but a wildly different reported operating
    # point -- the exact pattern observed between a checkpoint and its own ONNX export.
    shifted = dict(baseline)
    shifted["artifact_sha256"] = "b" * 64
    shifted["metrics"] = {**baseline["metrics"], "precision": 0.162, "recall": 0.234}
    onnx.write_text(json.dumps(shifted), encoding="utf-8")
    assert compare_parity(pt, onnx, tmp_path / "ok.json", tolerance=0.01).passed is True

    # Divergence at a shared confidence is genuine and must fail.
    diverged = dict(baseline)
    diverged["artifact_sha256"] = "b" * 64
    diverged["threshold_matched_metrics"] = {
        **_matched(0.5, 0.4),
        "recall@0.25": 0.44,
    }
    onnx.write_text(json.dumps(diverged), encoding="utf-8")
    with pytest.raises(ArtifactError, match="parity"):
        compare_parity(pt, onnx, tmp_path / "diverged.json", tolerance=0.01)


def test_export_uses_static_runtime_contract_arguments(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"pt")
    destination = tmp_path / "return" / "model.onnx"
    models: list[_FakeModel] = []

    result = export_static_onnx(
        checkpoint,
        _profile(),
        destination,
        model_factory=lambda path: models.append(_FakeModel(path)) or models[-1],
    )

    assert result == destination
    assert result.read_bytes() == b"onnx"
    assert models[0].export_kwargs == {
        "format": "onnx",
        "imgsz": 640,
        "opset": 17,
        "dynamic": False,
        "simplify": False,
        "nms": False,
        "batch": 1,
        "device": "cpu",
    }


class _Node:
    def __init__(self, shape: list[int], node_type: str = "tensor(float)") -> None:
        self.shape = shape
        self.type = node_type


class _Metadata:
    custom_metadata_map: ClassVar[dict[str, str]] = {
        "names": "{0: 'dark_gill', 1: 'white_spot'}",
        "task": "detect",
        "imgsz": "[640, 640]",
    }


class _Session:
    def __init__(self, output_shape: list[int]) -> None:
        self.output_shape = output_shape

    def get_inputs(self) -> list[_Node]:
        return [_Node([1, 3, 640, 640])]

    def get_outputs(self) -> list[_Node]:
        return [_Node(self.output_shape)]

    def get_modelmeta(self) -> _Metadata:
        return _Metadata()


def test_validate_onnx_contract_rejects_transposed_output(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"graph")

    validate_onnx_contract(model, 640, session_factory=lambda _: _Session([1, 6, 8400]))

    with pytest.raises(ArtifactError, match="output shape"):
        validate_onnx_contract(model, 640, session_factory=lambda _: _Session([1, 8400, 6]))
