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

# Only threshold-independent metrics are compared from the reported metric block.
#
# Ultralytics reports `precision` and `recall` in results_dict at whichever confidence
# maximises F1 for that artifact. That is a point on a curve, chosen per artifact. When the
# F1 curve is flat near its peak, two numerically equivalent models select different
# confidences and report precision/recall that differ far more than any defensible
# tolerance, while their underlying curves match. Observed in practice: a PyTorch checkpoint
# and its own ONNX export selected 0.0921 and 0.1061, giving a 0.0149 recall difference,
# with mAP50 agreeing to 0.0014 and F1 at the two optima agreeing to 0.0025.
#
# Precision and recall are therefore compared at fixed confidences shared by both
# artifacts, so like is compared with like. This does not weaken the gate: an export that
# genuinely detects differently still diverges at matched confidences.
_PARITY_METRIC_KEYS = ("map50", "map50_95")

# Recall is compared across the whole confidence range because its denominator is the fixed
# number of ground-truth instances, so one differing detection always moves it by 1/N.
#
# Precision is compared only where enough predictions survive. Its denominator is the
# prediction count, which collapses as confidence rises, so its measurement quantum grows
# without bound. Measured on the reference test split (799 instances):
#
#   conf   surviving predictions   precision change from one detection
#   0.05                   2,742                               0.0004
#   0.20                     446                               0.0022
#   0.25                     257                               0.0039
#   0.40                      83                               0.0121
#   0.50                      20                               0.0497
#
# Above roughly 0.20 the quantum approaches or exceeds the 0.010 tolerance, so a comparison
# there measures sampling noise rather than export fidelity. The high-confidence region is
# still covered: mAP50 and mAP50-95 integrate the entire precision-recall curve.
_PARITY_RECALL_CONFIDENCES = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
_PARITY_PRECISION_CONFIDENCES = (0.05, 0.10, 0.15, 0.20)
_CURVE_KEYS: dict[str, tuple[str, tuple[float, ...]]] = {
    "precision": ("p_curve", _PARITY_PRECISION_CONFIDENCES),
    "recall": ("r_curve", _PARITY_RECALL_CONFIDENCES),
}


def _matched_metric_name(metric: str, confidence: float) -> str:
    return f"{metric}@{confidence:.2f}"


def _matched_metric_names() -> list[str]:
    return [
        _matched_metric_name(metric, confidence)
        for metric, (_, confidences) in _CURVE_KEYS.items()
        for confidence in confidences
    ]


class ArtifactError(RuntimeError):
    """An evaluated or exported artifact violates a required contract."""


class ValidationBox(Protocol):
    maps: Any
    p_curve: Any
    r_curve: Any


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
    #: Precision and recall sampled at each confidence in _PARITY_CONFIDENCES, so parity
    #: compares both artifacts at the same operating points rather than at each one's own
    #: max-F1 confidence. `metrics` keeps the conventional Ultralytics figures for reporting.
    threshold_matched_metrics: dict[str, float]


@dataclass(frozen=True, slots=True)
class ParityReport:
    passed: bool
    tolerance: float
    maximum_delta: float
    deltas: dict[str, float]
    comparison_confidences: dict[str, tuple[float, ...]]
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


def _as_sequence(raw: Any) -> list[Any]:
    to_list = getattr(raw, "tolist", None)
    values = to_list() if callable(to_list) else raw
    if not isinstance(values, list | tuple) or not values:
        return []
    return list(values)


def _mean_curve(raw: Any, label: str) -> list[float]:
    """Collapse a per-class curve to a single averaged curve over the confidence axis."""
    values = _as_sequence(raw)
    if not values:
        raise ArtifactError(f"{label} is not a curve")
    if isinstance(values[0], list | tuple):
        rows = [_as_sequence(row) for row in values]
        if not all(rows) or len({len(row) for row in rows}) != 1:
            raise ArtifactError(f"{label} has inconsistent per-class curve lengths")
        return [
            sum(_finite_float(value, label) for value in column) / len(column)
            for column in zip(*rows, strict=True)
        ]
    return [_finite_float(value, label) for value in values]


def _sample_curve(curve: list[float], confidence: float, label: str) -> float:
    """Read a curve at a confidence, assuming Ultralytics' uniform 0..1 confidence axis."""
    if len(curve) < 2:
        raise ArtifactError(f"{label} is too short to sample")
    index = round(confidence * (len(curve) - 1))
    return _unit_float(curve[min(len(curve) - 1, max(0, index))], label)


def _threshold_matched_metrics(box: ValidationBox) -> dict[str, float]:
    sampled: dict[str, float] = {}
    for metric, (attribute, confidences) in _CURVE_KEYS.items():
        curve = _mean_curve(getattr(box, attribute, None), f"{metric} curve")
        for confidence in confidences:
            sampled[_matched_metric_name(metric, confidence)] = _sample_curve(
                curve, confidence, f"{metric} at confidence {confidence}"
            )
    return sampled


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
    matched = _threshold_matched_metrics(result.box)
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
        threshold_matched_metrics=matched,
    )
    _write_json_exclusive(
        destination,
        {
            "schema_version": "1.1.0",
            "artifact_sha256": summary.artifact_sha256,
            "dataset_descriptor_sha256": summary.dataset_descriptor_sha256,
            "prepared_manifest_sha256": summary.prepared_manifest_sha256,
            "dataset_inventory_sha256": summary.dataset_inventory_sha256,
            "test_image_count": summary.test_image_count,
            "split": summary.split,
            "metrics": summary.metrics,
            "per_class_map50_95": list(summary.per_class_map50_95),
            "threshold_matched_metrics": summary.threshold_matched_metrics,
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
    if not isinstance(raw, dict) or raw.get("schema_version") != "1.1.0":
        raise ArtifactError("evaluation record schema is invalid")
    metrics = raw.get("metrics")
    maps = raw.get("per_class_map50_95")
    if not isinstance(metrics, dict) or not isinstance(maps, list) or len(maps) != 2:
        raise ArtifactError("evaluation record metrics are invalid")
    parsed_metrics = {name: _unit_float(metrics.get(name), name) for name in _METRIC_KEYS}
    matched = raw.get("threshold_matched_metrics")
    if not isinstance(matched, dict):
        raise ArtifactError("evaluation record threshold_matched_metrics are invalid")
    parsed_matched = {
        name: _unit_float(matched.get(name), name) for name in _matched_metric_names()
    }
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
        threshold_matched_metrics=parsed_matched,
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
    # Threshold-independent metrics compare directly.
    deltas = {name: abs(pytorch.metrics[name] - onnx.metrics[name]) for name in _PARITY_METRIC_KEYS}
    # Precision and recall compare only at shared confidences. Comparing the reported
    # metrics would measure each artifact's own max-F1 threshold choice, not its behaviour.
    deltas.update(
        {
            name: abs(
                pytorch.threshold_matched_metrics[name] - onnx.threshold_matched_metrics[name]
            )
            for name in pytorch.threshold_matched_metrics
        }
    )
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
        comparison_confidences={
            metric: confidences for metric, (_, confidences) in _CURVE_KEYS.items()
        },
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
        {
            "schema_version": "1.1.0",
            **asdict(report),
            "comparison_confidences": {
                metric: list(confidences)
                for metric, confidences in report.comparison_confidences.items()
            },
        },
    )
    return report
