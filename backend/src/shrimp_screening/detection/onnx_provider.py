"""The only module in the runtime that imports onnxruntime.

Everything here is a **load-time** assertion, not a first-inference surprise. A
model that does not match the contract must fail while the process is starting, so
``/readyz`` reports 503 -- not halfway through a demonstration, and never by
producing plausible-looking boxes from a misread tensor.

What is asserted, and why each one has bitten somebody
------------------------------------------------------
* **sha256 is registered.** Weights arrive out of band; an unrecorded artifact is
  precisely the one that must not be executed.
* **One input, ``[1, 3, S, S]``, float32.** A dynamic export would silently accept
  a differently-shaped batch and letterbox against the wrong size.
* **One rank-3 output, ``(1, 4 + nc, anchors)``.** ``(1, anchors, 4 + nc)`` is the
  transposed YOLOv5 layout; ``(1, max_det, 6)`` is an ``nms=True`` export. Both
  decode into garbage that looks like detections.
* **``anchors == sum((S // stride)**2)``** over strides 8/16/32, which pins the
  three-scale detect head.
* **``names`` in the ONNX metadata equals the registry's class names, exactly.**
  A class-order flip between training and registration produces a fully working
  application that confidently reports the wrong marker. This equality check is the
  only thing standing between the project and that outcome.
* **``task == "detect"`` and ``imgsz`` agrees with the input.**
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from shrimp_screening.contracts.enums import OutputLayout, ProviderKind
from shrimp_screening.detection.decode import (
    CUSTOM_YOLO_ANCHORS_PER_SCALE,
    CUSTOM_YOLO_STRIDES,
    decode_custom_yolo_anchor_v1,
    decode_ultralytics_v8,
    expected_anchor_count,
)
from shrimp_screening.detection.letterbox import letterbox_image
from shrimp_screening.detection.protocol import Detection, DetectorMetadata
from shrimp_screening.detection.registry import ModelRegistry, RegisteredModel, sha256_of
from shrimp_screening.settings import INTRA_OP_THREADS


class ModelContractError(RuntimeError):
    """The artifact does not satisfy the declared ONNX output contract."""


def _session_options() -> ort.SessionOptions:
    options = ort.SessionOptions()
    options.intra_op_num_threads = INTRA_OP_THREADS
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # ORT's default spin-wait keeps worker threads hot between requests, which
    # starves asyncio on a two-core machine and shows up as random latency spikes.
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return options


def _parse_names(raw: str) -> dict[int, str]:
    """Parse Ultralytics' ``names`` metadata, which is a Python dict repr."""
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError) as exc:
        raise ModelContractError("the model's 'names' metadata is not parseable") from exc
    if not isinstance(parsed, dict) or not parsed:
        raise ModelContractError("the model's 'names' metadata is not a non-empty mapping")
    names: dict[int, str] = {}
    for key, value in parsed.items():
        if not isinstance(key, int) or not isinstance(value, str):
            raise ModelContractError("the model's 'names' metadata is not {int: str}")
        names[key] = value
    return names


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ModelContractError(message)


def _static_shape(dimensions: list[Any]) -> tuple[int, ...] | None:
    if any(not isinstance(value, int) for value in dimensions):
        return None
    return tuple(int(value) for value in dimensions)


def _validate_session(session: ort.InferenceSession, entry: RegisteredModel) -> str:
    """Assert the whole contract and return the single input name."""
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    _require(len(inputs) == 1, f"expected exactly one input, found {len(inputs)}")
    _require(len(outputs) == 1, f"expected exactly one output, found {len(outputs)}")

    size = entry.input_size
    input_shape = _static_shape(list(inputs[0].shape))
    _require(
        input_shape == (1, 3, size, size),
        f"input shape {inputs[0].shape} is not the pinned static [1, 3, {size}, {size}]",
    )
    _require(
        inputs[0].type == "tensor(float)",
        f"input type {inputs[0].type} is not float32",
    )

    output_shape = _static_shape(list(outputs[0].shape))
    _require(
        output_shape is not None and len(output_shape) == 3,
        f"output shape {outputs[0].shape} is not a static rank-3 tensor",
    )
    assert output_shape is not None  # narrowed by the check above

    if entry.output_layout is OutputLayout.CUSTOM_YOLO_ANCHOR_V1:
        channels = 5 + len(entry.class_names)
        anchor_positions = CUSTOM_YOLO_ANCHORS_PER_SCALE * expected_anchor_count(
            size, CUSTOM_YOLO_STRIDES
        )
        _require(
            output_shape[0] == 1 and output_shape[1] == channels,
            f"output shape {output_shape} is not (1, {channels}, anchors) for "
            "custom_yolo_anchor_v1",
        )
        _require(
            output_shape[2] == anchor_positions,
            f"output declares {output_shape[2]} anchor positions; a 3-scale "
            f"anchor-based head with {CUSTOM_YOLO_ANCHORS_PER_SCALE} anchors/scale at "
            f"{size} emits {anchor_positions}",
        )
    else:
        channels = 4 + len(entry.class_names)
        _require(
            output_shape[0] == 1 and output_shape[1] == channels,
            f"output shape {output_shape} is not (1, {channels}, anchors); a "
            f"(1, anchors, {channels}) tensor is the transposed YOLOv5 layout and "
            "(1, max_det, 6) is an nms=True export",
        )
        anchors_count = expected_anchor_count(size)
        _require(
            output_shape[2] == anchors_count,
            f"output declares {output_shape[2]} anchors; a three-scale detect head at "
            f"{size} emits {anchors_count}",
        )

    metadata = session.get_modelmeta().custom_metadata_map or {}
    _require("names" in metadata, "the model carries no 'names' metadata")
    names = _parse_names(metadata["names"])
    _require(
        names == entry.class_names,
        f"the model's class names {names} do not match the registered "
        f"{entry.class_names}; a class-order flip would mislabel every detection",
    )
    _require(
        metadata.get("task") == "detect",
        f"model task {metadata.get('task')!r} is not 'detect'",
    )
    declared_imgsz = metadata.get("imgsz")
    _require(
        declared_imgsz is not None and _parse_imgsz(declared_imgsz) == (size, size),
        f"model imgsz {declared_imgsz!r} does not agree with the registered {size}",
    )
    return str(inputs[0].name)


def _parse_imgsz(raw: str) -> tuple[int, int] | None:
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return None
    if isinstance(parsed, int):
        return (parsed, parsed)
    if isinstance(parsed, list | tuple) and len(parsed) == 2:
        first, second = parsed
        if isinstance(first, int) and isinstance(second, int):
            return (first, second)
    return None


class OnnxProvider:
    """Runs a registered, contract-checked ONNX detect model on the CPU."""

    def __init__(
        self,
        session: ort.InferenceSession,
        input_name: str,
        entry: RegisteredModel,
        *,
        score_threshold: float,
        iou_threshold: float,
        max_detections: int = 300,
    ) -> None:
        self._session = session
        self._input_name = input_name
        self._entry = entry
        self._score_threshold = score_threshold
        self._iou_threshold = iou_threshold
        self._max_detections = max_detections
        self._metadata = DetectorMetadata(
            available=True,
            provider=ProviderKind.ONNX,
            model_id=entry.model_id,
            version=entry.version,
            output_layout=entry.output_layout,
            class_names=dict(entry.class_names),
            mapping_status=entry.dataset_mapping_status,
            demonstration=False,
        )

    @property
    def metadata(self) -> DetectorMetadata:
        return self._metadata

    def warm_up(self) -> None:
        """Pay the graph-optimization cost at startup, not mid-demonstration.

        The first inference through a freshly created session carries 0.5-2 s of
        one-off work, which would otherwise read as a hang on the first upload.
        """
        size = self._entry.input_size
        zeros = np.zeros((1, 3, size, size), dtype=np.float32)
        self._session.run(None, {self._input_name: zeros})

    def infer(self, image: np.ndarray) -> list[Detection]:
        batch, transform = letterbox_image(image, self._entry.input_size)
        raw = self._session.run(None, {self._input_name: batch})[0]
        if self._entry.output_layout is OutputLayout.CUSTOM_YOLO_ANCHOR_V1:
            return decode_custom_yolo_anchor_v1(
                np.asarray(raw),
                self._entry.class_names,
                transform,
                self._entry.anchors,
                score_threshold=self._score_threshold,
                iou_threshold=self._iou_threshold,
                max_detections=self._max_detections,
            )
        return decode_ultralytics_v8(
            np.asarray(raw),
            self._entry.class_names,
            transform,
            score_threshold=self._score_threshold,
            iou_threshold=self._iou_threshold,
            max_detections=self._max_detections,
        )


def load_onnx_provider(
    model_path: Path,
    registry: ModelRegistry,
    *,
    score_threshold: float,
    iou_threshold: float,
    max_detections: int = 300,
) -> OnnxProvider:
    """Load and fully validate an ONNX artifact, or raise :class:`ModelContractError`."""
    if not model_path.is_file():
        raise ModelContractError(f"no model artifact exists at {model_path}")
    entry = _lookup(model_path, registry)
    try:
        session = ort.InferenceSession(
            str(model_path),
            sess_options=_session_options(),
            providers=["CPUExecutionProvider"],
        )
    except Exception as exc:  # ORT raises a wide, unstable set of exception types
        raise ModelContractError(f"onnxruntime refused to load the artifact: {exc}") from exc
    input_name = _validate_session(session, entry)
    provider = OnnxProvider(
        session,
        input_name,
        entry,
        score_threshold=score_threshold,
        iou_threshold=iou_threshold,
        max_detections=max_detections,
    )
    provider.warm_up()
    return provider


def _lookup(model_path: Path, registry: ModelRegistry) -> RegisteredModel:
    digest = sha256_of(model_path)
    entry = registry.by_sha256(digest)
    if entry.filename != model_path.name:
        raise ModelContractError(
            f"the artifact at {model_path.name} matches the sha256 registered for "
            f"{entry.filename}; refusing a renamed artifact"
        )
    return entry
