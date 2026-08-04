"""Extract the runtime registry entry from an ONNX model's own metadata."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any

import onnx


class MetadataError(RuntimeError):
    """The ONNX file does not describe a supported detection model."""


def _metadata(model: onnx.ModelProto) -> dict[str, str]:
    return {item.key: item.value for item in model.metadata_props}


def _parse_mapping(raw: str, key: str) -> dict[str, str]:
    try:
        value: Any = json.loads(raw)
    except json.JSONDecodeError:
        try:
            value = ast.literal_eval(raw)
        except (SyntaxError, ValueError) as exc:
            raise MetadataError(
                f"ONNX metadata {key!r} is not valid JSON or Python mapping"
            ) from exc
    if not isinstance(value, dict) or not value:
        raise MetadataError(f"ONNX metadata {key!r} must be a non-empty class mapping")
    result: dict[str, str] = {}
    for index, name in value.items():
        try:
            parsed_index = int(index)
        except (TypeError, ValueError) as exc:
            raise MetadataError(f"ONNX metadata {key!r} has a non-integer class index") from exc
        if not isinstance(name, str) or not name.strip():
            raise MetadataError(f"ONNX metadata {key!r} has an empty class name")
        result[str(parsed_index)] = name.strip()
    if sorted(map(int, result)) != list(range(len(result))):
        raise MetadataError(f"ONNX metadata {key!r} must cover class indexes 0..n-1")
    return result


def _static_dimensions(value_info: onnx.ValueInfoProto) -> tuple[int, ...]:
    tensor = value_info.type.tensor_type
    dimensions = tuple(dimension.dim_value for dimension in tensor.shape.dim)
    if not dimensions or any(value <= 0 for value in dimensions):
        raise MetadataError(f"ONNX tensor {value_info.name!r} must have static dimensions")
    return dimensions


def _integer_metadata(values: dict[str, str], key: str, default: int | None = None) -> int:
    raw = values.get(key)
    if raw is None:
        if default is not None:
            return default
        raise MetadataError(f"ONNX metadata is missing {key!r}")
    try:
        return int(raw)
    except ValueError as exc:
        raise MetadataError(f"ONNX metadata {key!r} must be an integer") from exc


def _anchors(values: dict[str, str]) -> list[list[float]]:
    raw = values.get("anchors")
    if raw is None:
        raise MetadataError(
            "the ONNX output is custom anchor-based, but metadata is missing 'anchors'"
        )
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MetadataError("ONNX metadata 'anchors' must be a JSON array") from exc
    if not isinstance(parsed, list) or len(parsed) != 9:
        raise MetadataError("ONNX metadata 'anchors' must contain exactly 9 pairs")
    return parsed


def extract_model_entry(model_path: Path) -> dict[str, Any]:
    """Return the registry entry needed by the backend, without a sidecar file."""
    try:
        model = onnx.load(str(model_path), load_external_data=False)
    except Exception as exc:  # onnx exposes several parser-specific exception types
        raise MetadataError(f"could not parse ONNX file: {exc}") from exc

    values = _metadata(model)
    if values.get("task") != "detect":
        raise MetadataError(f"ONNX metadata task must be 'detect', got {values.get('task')!r}")
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise MetadataError("the ONNX graph must have exactly one input and one output")

    input_shape = _static_dimensions(model.graph.input[0])
    if len(input_shape) != 4 or input_shape[:2] != (1, 3) or input_shape[2] != input_shape[3]:
        raise MetadataError("the ONNX input must have static shape [1, 3, S, S]")
    input_size = input_shape[2]
    class_names = _parse_mapping(values.get("names", ""), "names")
    output_shape = _static_dimensions(model.graph.output[0])
    if len(output_shape) != 3 or output_shape[0] != 1:
        raise MetadataError("the ONNX output must have static rank-3 shape [1, channels, anchors]")

    expected_anchors = sum((input_size // stride) ** 2 for stride in (8, 16, 32))
    if output_shape[2] != expected_anchors:
        raise MetadataError(
            f"the ONNX output has {output_shape[2]} anchor positions; expected "
            f"{expected_anchors} for input size {input_size}"
        )
    if output_shape[1] == 4 + len(class_names):
        output_layout = "ultralytics_v8_detect_v1"
        model_anchors: list[list[float]] = []
        default_toolchain = "ultralytics-auto"
    elif output_shape[1] == 5 + len(class_names):
        output_layout = "custom_yolo_anchor_v1"
        model_anchors = _anchors(values)
        default_toolchain = "custom-pytorch-yolo-auto"
    else:
        raise MetadataError(
            f"output channels {output_shape[1]} do not match {len(class_names)} classes "
            "for a supported YOLO detection layout"
        )

    opsets = [item.version for item in model.opset_import if item.domain in ("", "ai.onnx")]
    if opsets != [17]:
        raise MetadataError(f"ONNX opset must be 17, found {opsets or 'none'}")

    artifact_license = values.get("artifact_license")
    if not artifact_license:
        raise MetadataError("ONNX metadata is missing 'artifact_license'")
    toolchain = values.get("training_toolchain", default_toolchain)
    mapping_status = values.get("dataset_mapping_status", "PROVISIONAL_UNCONFIRMED")
    return {
        "model_id": values.get("model_id", model_path.stem),
        "version": values.get("version", "auto-detected"),
        "filename": model_path.name,
        "input_size": input_size,
        "class_names": class_names,
        "opset": 17,
        "output_layout": output_layout,
        "anchors": model_anchors,
        "dataset_mapping_status": mapping_status,
        "artifact_license": artifact_license,
        "training_toolchain": toolchain,
    }


def main(argv: list[str] | None = None) -> int:
    path = Path((argv or sys.argv[1:])[0])
    print(json.dumps(extract_model_entry(path), separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
