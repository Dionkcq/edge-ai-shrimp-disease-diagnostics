"""Synthetic ONNX detect artifacts, generated into `tmp_path` and never committed.

No trained weights exist in this repository and none ever will, so the load-time
contract in `shrimp_screening.detection.onnx_provider` would otherwise be
unreachable code -- exactly the code whose failure mode is "plausible-looking boxes
from a misread tensor". This module builds real ONNX graphs that onnxruntime
actually loads and runs, so each assertion is proved against a session rather than
against a mock that agrees with whatever the implementation happens to do.

The graph is a genuine function of its input rather than a canned constant::

    images[1, 3, S, S] -> Reshape[1, -1] -> Slice[0 : C * A] -> Reshape[1, C, A] -> Mul

`Mul` scales the four geometry rows by 255 and leaves the score rows alone, which
undoes the `/255.0` in `letterbox_image` for the coordinates and keeps the scores in
`[0, 1]` with a `1/255` quantum. The consequence is the useful part: for an image
whose side equals the input size the letterbox is the identity, so a caller can
write a box and a score into the red channel and assert exactly where the provider
reports them. `plant_detection_image` does that arithmetic.

`S = 64` is deliberate. 8, 16 and 32 all divide it, so `expected_anchor_count` is a
real three-scale count, and it is only 84 anchors -- a session builds and runs in
milliseconds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from shrimp_screening.contracts.enums import DatasetMappingStatus, OutputLayout
from shrimp_screening.detection.decode import expected_anchor_count
from shrimp_screening.detection.registry import ModelRegistry, RegisteredModel, sha256_of

SYNTHETIC_INPUT_SIZE = 64

#: Mirrors the fixture provider's provisional mapping rather than inventing an order.
SYNTHETIC_CLASS_NAMES: dict[int, str] = {0: "dark_gill", 1: "white_spot"}

#: Scores travel through a `uint8` red channel, so they land on a 1/255 lattice.
SCORE_QUANTUM = 1.0 / 255.0

#: The opset and IR version an Ultralytics export targets, and which ORT 1.28 loads.
_OPSET = 17
_IR_VERSION = 10

#: Row indices of the geometry channels in a `(1, 4 + nc, anchors)` detect tensor.
_GEOMETRY_ROWS = 4


def default_metadata(
    *,
    input_size: int = SYNTHETIC_INPUT_SIZE,
    class_names: dict[int, str] | None = None,
) -> dict[str, str]:
    """The `custom_metadata_map` an Ultralytics detect export carries."""
    names = SYNTHETIC_CLASS_NAMES if class_names is None else class_names
    return {
        # Ultralytics writes a Python dict repr, which is what `_parse_names` reads.
        "names": repr(dict(names)),
        "task": "detect",
        "imgsz": f"[{input_size}, {input_size}]",
        "stride": "32",
        "batch": "1",
    }


def write_detect_model(
    path: Path,
    *,
    input_size: int = SYNTHETIC_INPUT_SIZE,
    class_count: int = len(SYNTHETIC_CLASS_NAMES),
    output_shape: tuple[int, ...] | None = None,
    metadata: dict[str, str] | None = None,
    dynamic_batch: bool = False,
    double_input: bool = False,
    extra_input: bool = False,
    extra_output: bool = False,
) -> Path:
    """Write one synthetic detect artifact and return its path.

    Every keyword exists to violate exactly one clause of the load-time contract, so
    a test names the defect it is proving rather than hand-assembling a graph.
    """
    shape = output_shape or (1, _GEOMETRY_ROWS + class_count, expected_anchor_count(input_size))
    window = int(np.prod(shape))
    assert window <= input_size * input_size, "the window must fit inside the red channel"

    batch_dim: int | str = "batch" if dynamic_batch else 1
    input_type = TensorProto.DOUBLE if double_input else TensorProto.FLOAT
    inputs = [
        helper.make_tensor_value_info("images", input_type, [batch_dim, 3, input_size, input_size])
    ]
    if extra_input:
        # A second input is how a graph that expects an auxiliary tensor (a mask, a
        # scale) presents itself; feeding it zeros would be guesswork.
        inputs.append(helper.make_tensor_value_info("aux", TensorProto.FLOAT, [1]))

    initializers = [
        numpy_helper.from_array(np.array([1, -1], dtype=np.int64), "flat_shape"),
        numpy_helper.from_array(np.array([0], dtype=np.int64), "window_start"),
        numpy_helper.from_array(np.array([window], dtype=np.int64), "window_end"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), "window_axis"),
        numpy_helper.from_array(np.array(shape, dtype=np.int64), "output_shape"),
    ]
    nodes = []
    source = "images"
    if double_input:
        nodes.append(helper.make_node("Cast", ["images"], ["cast"], to=TensorProto.FLOAT))
        source = "cast"
    nodes += [
        helper.make_node("Reshape", [source, "flat_shape"], ["flat"]),
        helper.make_node(
            "Slice", ["flat", "window_start", "window_end", "window_axis"], ["window"]
        ),
        helper.make_node("Reshape", ["window", "output_shape"], ["shaped"]),
    ]

    if len(shape) == 3:
        scale = np.ones((1, shape[1], 1), dtype=np.float32)
        scale[0, :_GEOMETRY_ROWS, 0] = 255.0
        initializers.append(numpy_helper.from_array(scale, "channel_scale"))
        nodes.append(helper.make_node("Mul", ["shaped", "channel_scale"], ["output0"]))
    else:
        # A rank-2 output cannot broadcast against a per-channel scale; the point of
        # that variant is the rank assertion, not the arithmetic.
        nodes.append(helper.make_node("Identity", ["shaped"], ["output0"]))

    outputs = [helper.make_tensor_value_info("output0", TensorProto.FLOAT, list(shape))]
    if extra_output:
        nodes.append(helper.make_node("Identity", ["output0"], ["output1"]))
        outputs.append(helper.make_tensor_value_info("output1", TensorProto.FLOAT, list(shape)))

    graph = helper.make_graph(nodes, "synthetic_detect", inputs, outputs, initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET)])
    model.ir_version = _IR_VERSION
    resolved = default_metadata(input_size=input_size) if metadata is None else metadata
    if resolved:
        helper.set_model_props(model, resolved)
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    return path


def registered_model(
    path: Path,
    *,
    input_size: int = SYNTHETIC_INPUT_SIZE,
    class_names: dict[int, str] | None = None,
    filename: str | None = None,
    sha256: str | None = None,
) -> RegisteredModel:
    """A registry entry vouching for `path`, digest included."""
    return RegisteredModel(
        model_id="synthetic-detect",
        version="0.0.0-synthetic",
        filename=path.name if filename is None else filename,
        sha256=sha256_of(path) if sha256 is None else sha256,
        input_size=input_size,
        class_names=dict(SYNTHETIC_CLASS_NAMES if class_names is None else class_names),
        opset=_OPSET,
        output_layout=OutputLayout.ULTRALYTICS_V8_DETECT_V1,
        anchors=(),
        dataset_mapping_status=DatasetMappingStatus.PROVISIONAL_UNCONFIRMED,
        artifact_license="AGPL-3.0-or-later",
        training_toolchain="synthetic-onnx-graph",
    )


def registry_for(
    path: Path,
    *,
    input_size: int = SYNTHETIC_INPUT_SIZE,
    class_names: dict[int, str] | None = None,
    filename: str | None = None,
    sha256: str | None = None,
) -> ModelRegistry:
    """A one-entry registry vouching for `path`."""
    return ModelRegistry(
        models=(
            registered_model(
                path,
                input_size=input_size,
                class_names=class_names,
                filename=filename,
                sha256=sha256,
            ),
        )
    )


def registry_document(path: Path) -> dict[str, object]:
    """The JSON form of `registry_for`, for tests that go through `load_registry`."""
    entry = registered_model(path)
    return {
        "schema_version": "1.0.0",
        "models": [
            {
                "model_id": entry.model_id,
                "version": entry.version,
                "filename": entry.filename,
                "sha256": entry.sha256,
                "input_size": entry.input_size,
                "class_names": {str(i): n for i, n in entry.class_names.items()},
                "opset": entry.opset,
                "output_layout": entry.output_layout.value,
                "anchors": [list(pair) for pair in entry.anchors],
                "dataset_mapping_status": entry.dataset_mapping_status.value,
                "artifact_license": entry.artifact_license,
                "training_toolchain": entry.training_toolchain,
            }
        ],
    }


def plant_detection_image(
    *,
    box_pixels: tuple[float, float, float, float],
    class_index: int,
    score: float,
    input_size: int = SYNTHETIC_INPUT_SIZE,
    class_count: int = len(SYNTHETIC_CLASS_NAMES),
    anchor: int = 0,
) -> np.ndarray:
    """An `(S, S, 3)` uint8 image that the synthetic graph turns into one detection.

    `box_pixels` is `cx, cy, w, h` in letterboxed input pixels, which for a square
    image of side `input_size` is also the original frame. Values must be whole
    numbers in `0..255`: they survive as bytes.
    """
    anchors = expected_anchor_count(input_size)
    image = np.zeros((input_size, input_size, 3), dtype=np.uint8)

    def _write(channel: int, value: int) -> None:
        index = channel * anchors + anchor
        image[index // input_size, index % input_size, 0] = value

    for channel, value in enumerate(box_pixels):
        assert value == int(value) and 0 <= value <= 255, f"{value} does not survive as a byte"
        _write(channel, int(value))
    _write(_GEOMETRY_ROWS + class_index, round(score * 255))
    assert class_index < class_count
    return image
