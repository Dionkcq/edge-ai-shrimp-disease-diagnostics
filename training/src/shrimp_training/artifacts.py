# SPDX-License-Identifier: AGPL-3.0-or-later
"""Evaluation, static ONNX export, contract validation, and backend parity."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol, cast

from shrimp_training.config import TrainingProfile
from shrimp_training.dataset import validate_prepared_dataset

_METRIC_KEYS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "map50": "metrics/mAP50(B)",
    "map50_95": "metrics/mAP50-95(B)",
}
_EXPECTED_NAMES = {0: "dark_gill", 1: "white_spot"}


class ArtifactError(RuntimeError):
    """An evaluated or exported artifact violates a required contract."""


class ValidationBox(Protocol):
    maps: Any


class ValidationResult(Protocol):
    results_dict: Any
    box: ValidationBox


class ArtifactModel(Protocol):
    def val(self, **kwargs: Any) -> ValidationResult: ...

    def export(self, **kwargs: Any) -> str | Path: ...


ArtifactModelFactory = Callable[[Path], ArtifactModel]


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    artifact_sha256: str
    dataset_descriptor_sha256: str
    prepared_manifest_sha256: str
    dataset_inventory_sha256: str
    test_image_count: int
    split: str
    metrics: dict[str, float]
    per_class_map50_95: tuple[float, float]


@dataclass(frozen=True, slots=True)
class ParityReport:
    passed: bool
    tolerance: float
    maximum_delta: float
    deltas: dict[str, float]
    pytorch_evaluation_sha256: str
    onnx_evaluation_sha256: str
    prepared_manifest_sha256: str
    dataset_inventory_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise ArtifactError(f"artifact cannot be read: {path}") from exc
    return digest.hexdigest()


def _model_factory(artifact: Path) -> ArtifactModel:
    module = import_module("ultralytics")
    constructor = getattr(module, "YOLO", None)
    if constructor is None:
        raise ArtifactError("the installed ultralytics package exposes no YOLO constructor")
    return cast(ArtifactModel, constructor(str(artifact)))


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ArtifactError(f"{label} is not numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ArtifactError(f"{label} is not finite")
    return parsed


def _unit_float(value: Any, label: str) -> float:
    parsed = _finite_float(value, label)
    if not 0.0 <= parsed <= 1.0:
        raise ArtifactError(f"{label} must be within [0, 1]")
    return parsed


def _write_json_exclusive(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(document, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError:
        raise
    except OSError as exc:
        raise ArtifactError(f"cannot write artifact record: {path}") from exc


def evaluate_artifact(
    artifact: Path,
    dataset_yaml: Path,
    profile: TrainingProfile,
    destination: Path,
    *,
    prepared_root: Path,
    model_factory: ArtifactModelFactory = _model_factory,
) -> EvaluationSummary:
    """Evaluate one PT or ONNX artifact against the locked test partition."""
    if not artifact.is_file() or not dataset_yaml.is_file():
        raise ArtifactError("evaluation requires an artifact and dataset descriptor")
    before = validate_prepared_dataset(prepared_root)
    result = model_factory(artifact).val(
        data=str(dataset_yaml),
        split="test",
        imgsz=profile.image_size,
        batch=1,
        device="cpu" if artifact.suffix.casefold() == ".onnx" else profile.device,
        workers=profile.workers,
        plots=True,
        save_json=True,
    )
    if not isinstance(result.results_dict, dict):
        raise ArtifactError("Ultralytics returned no metric dictionary")
    metrics = {
        name: _unit_float(result.results_dict.get(source), source)
        for name, source in _METRIC_KEYS.items()
    }
    raw_maps = result.box.maps
    to_list = getattr(raw_maps, "tolist", None)
    if callable(to_list):
        raw_maps = to_list()
    if not isinstance(raw_maps, list | tuple) or len(raw_maps) != 2:
        raise ArtifactError("evaluation must return per-class mAP for exactly two classes")
    per_class = tuple(_unit_float(value, "per-class mAP50-95") for value in raw_maps)
    after = validate_prepared_dataset(prepared_root)
    before_binding = (before.manifest_sha256, before.inventory_sha256, before.split_counts)
    after_binding = (after.manifest_sha256, after.inventory_sha256, after.split_counts)
    if after_binding != before_binding:
        raise ArtifactError("prepared dataset changed during evaluation")
    summary = EvaluationSummary(
        artifact_sha256=_sha256(artifact),
        dataset_descriptor_sha256=_sha256(dataset_yaml),
        prepared_manifest_sha256=before.manifest_sha256,
        dataset_inventory_sha256=before.inventory_sha256,
        test_image_count=before.split_counts["test"],
        split="test",
        metrics=metrics,
        per_class_map50_95=cast(tuple[float, float], per_class),
    )
    _write_json_exclusive(
        destination,
        {
            "schema_version": "1.0.0",
            "artifact_sha256": summary.artifact_sha256,
            "dataset_descriptor_sha256": summary.dataset_descriptor_sha256,
            "prepared_manifest_sha256": summary.prepared_manifest_sha256,
            "dataset_inventory_sha256": summary.dataset_inventory_sha256,
            "test_image_count": summary.test_image_count,
            "split": summary.split,
            "metrics": summary.metrics,
            "per_class_map50_95": list(summary.per_class_map50_95),
        },
    )
    return summary


def export_static_onnx(
    checkpoint: Path,
    profile: TrainingProfile,
    destination: Path,
    *,
    model_factory: ArtifactModelFactory = _model_factory,
) -> Path:
    """Export a static, raw-head ONNX graph matching the MIT runtime decoder."""
    if not checkpoint.is_file():
        raise ArtifactError(f"checkpoint does not exist: {checkpoint}")
    exported = Path(
        model_factory(checkpoint).export(
            format="onnx",
            imgsz=profile.image_size,
            opset=17,
            dynamic=False,
            simplify=False,
            nms=False,
            batch=1,
            device="cpu",
        )
    )
    if not exported.is_file():
        raise ArtifactError(f"Ultralytics did not create its reported ONNX file: {exported}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with exported.open("rb") as source, destination.open("xb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
    except FileExistsError:
        raise
    except OSError as exc:
        destination.unlink(missing_ok=True)
        raise ArtifactError(f"cannot copy exported ONNX artifact to {destination}") from exc
    return destination


class _Node(Protocol):
    shape: Any
    type: str


class _Metadata(Protocol):
    custom_metadata_map: Any


class _Session(Protocol):
    def get_inputs(self) -> list[_Node]: ...

    def get_outputs(self) -> list[_Node]: ...

    def get_modelmeta(self) -> _Metadata: ...


SessionFactory = Callable[[Path], _Session]


def _session_factory(path: Path) -> _Session:
    module = import_module("onnxruntime")
    constructor = getattr(module, "InferenceSession", None)
    if constructor is None:
        raise ArtifactError("onnxruntime exposes no InferenceSession")
    return cast(_Session, constructor(str(path), providers=["CPUExecutionProvider"]))


def _static_shape(raw: Any) -> tuple[int, ...] | None:
    if not isinstance(raw, list | tuple) or any(type(value) is not int for value in raw):
        return None
    return tuple(raw)


def validate_onnx_contract(
    model: Path,
    image_size: int,
    *,
    session_factory: SessionFactory = _session_factory,
) -> None:
    """Reject an ONNX graph that the runtime would decode incorrectly."""
    if not model.is_file():
        raise ArtifactError(f"ONNX artifact does not exist: {model}")
    try:
        session = session_factory(model)
    except Exception as exc:
        raise ArtifactError(f"onnxruntime refused the exported graph: {exc}") from exc
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or _static_shape(inputs[0].shape) != (1, 3, image_size, image_size):
        raise ArtifactError("ONNX input shape is not the pinned static runtime shape")
    if inputs[0].type != "tensor(float)":
        raise ArtifactError("ONNX input type is not float32")
    anchors = sum((image_size // stride) ** 2 for stride in (8, 16, 32))
    expected_output = (1, 4 + len(_EXPECTED_NAMES), anchors)
    if len(outputs) != 1 or _static_shape(outputs[0].shape) != expected_output:
        raise ArtifactError(f"ONNX output shape is not {expected_output}")
    metadata = session.get_modelmeta().custom_metadata_map
    if not isinstance(metadata, dict):
        raise ArtifactError("ONNX metadata is absent")
    try:
        names = ast.literal_eval(metadata.get("names", ""))
        imgsz = ast.literal_eval(metadata.get("imgsz", ""))
    except (SyntaxError, ValueError) as exc:
        raise ArtifactError("ONNX metadata cannot be parsed") from exc
    if names != _EXPECTED_NAMES or metadata.get("task") != "detect":
        raise ArtifactError("ONNX class names or task metadata do not match the runtime")
    if imgsz not in (image_size, [image_size, image_size], (image_size, image_size)):
        raise ArtifactError("ONNX image-size metadata does not match the runtime")


def _digest_value(raw: dict[str, Any], field: str) -> str:
    value = raw.get(field)
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value.casefold())
    ):
        raise ArtifactError(f"evaluation {field} is not a SHA-256 digest")
    return value.casefold()


def _read_evaluation(path: Path) -> EvaluationSummary:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"evaluation record cannot be read: {path}") from exc
    if not isinstance(raw, dict) or raw.get("schema_version") != "1.0.0":
        raise ArtifactError("evaluation record schema is invalid")
    metrics = raw.get("metrics")
    maps = raw.get("per_class_map50_95")
    if not isinstance(metrics, dict) or not isinstance(maps, list) or len(maps) != 2:
        raise ArtifactError("evaluation record metrics are invalid")
    parsed_metrics = {name: _unit_float(metrics.get(name), name) for name in _METRIC_KEYS}
    artifact_digest = _digest_value(raw, "artifact_sha256")
    dataset_digest = _digest_value(raw, "dataset_descriptor_sha256")
    manifest_digest = _digest_value(raw, "prepared_manifest_sha256")
    inventory_digest = _digest_value(raw, "dataset_inventory_sha256")
    test_image_count = raw.get("test_image_count")
    if type(test_image_count) is not int or test_image_count < 1:
        raise ArtifactError("evaluation test_image_count must be a positive integer")
    if raw.get("split") != "test":
        raise ArtifactError("evaluation record is not bound to the test split")
    return EvaluationSummary(
        artifact_sha256=artifact_digest,
        dataset_descriptor_sha256=dataset_digest,
        prepared_manifest_sha256=manifest_digest,
        dataset_inventory_sha256=inventory_digest,
        test_image_count=test_image_count,
        split="test",
        metrics=parsed_metrics,
        per_class_map50_95=cast(
            tuple[float, float],
            tuple(_unit_float(value, "per-class mAP50-95") for value in maps),
        ),
    )


def compare_parity(
    pytorch_evaluation: Path,
    onnx_evaluation: Path,
    destination: Path,
    *,
    tolerance: float,
) -> ParityReport:
    """Require metric and per-class parity between PT and ONNX test evaluations."""
    if not 0.0 <= tolerance <= 1.0:
        raise ArtifactError("parity tolerance must be within [0, 1]")
    pytorch = _read_evaluation(pytorch_evaluation)
    onnx = _read_evaluation(onnx_evaluation)
    if pytorch.dataset_descriptor_sha256 != onnx.dataset_descriptor_sha256:
        raise ArtifactError("PyTorch and ONNX evaluations use different dataset descriptors")
    binding_fields = (
        "prepared_manifest_sha256",
        "dataset_inventory_sha256",
        "test_image_count",
    )
    if any(getattr(pytorch, field) != getattr(onnx, field) for field in binding_fields):
        raise ArtifactError("PyTorch and ONNX evaluations use different prepared dataset bytes")
    deltas = {name: abs(pytorch.metrics[name] - onnx.metrics[name]) for name in _METRIC_KEYS}
    deltas.update(
        {
            f"class_{index}_map50_95": abs(first - second)
            for index, (first, second) in enumerate(
                zip(pytorch.per_class_map50_95, onnx.per_class_map50_95, strict=True)
            )
        }
    )
    maximum = max(deltas.values(), default=0.0)
    report = ParityReport(
        passed=maximum <= tolerance,
        tolerance=tolerance,
        maximum_delta=maximum,
        deltas=deltas,
        pytorch_evaluation_sha256=_sha256(pytorch_evaluation),
        onnx_evaluation_sha256=_sha256(onnx_evaluation),
        prepared_manifest_sha256=pytorch.prepared_manifest_sha256,
        dataset_inventory_sha256=pytorch.dataset_inventory_sha256,
    )
    if not report.passed:
        raise ArtifactError(
            f"PyTorch/ONNX parity failed: maximum metric delta {maximum:.6f} > {tolerance:.6f}"
        )
    _write_json_exclusive(
        destination,
        {"schema_version": "1.0.0", **asdict(report)},
    )
    return report
