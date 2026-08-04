from __future__ import annotations

import json
from pathlib import Path

import onnx
import pytest
from onnx import StringStringEntryProto, TensorProto, helper

from scripts.model_metadata import MetadataError, extract_model_entry


def _write_model(path: Path, *, channels: int = 6, metadata: dict[str, str] | None = None) -> None:
    input_info = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, channels, 8400])
    graph = helper.make_graph([], "auto-metadata", [input_info], [output_info])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    for key, value in (metadata or {}).items():
        model.metadata_props.append(StringStringEntryProto(key=key, value=value))
    onnx.save(model, path)


def test_extracts_ultralytics_metadata_without_sidecar(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    _write_model(
        path,
        metadata={
            "task": "detect",
            "names": json.dumps({"0": "dark_gill", "1": "white_spot"}),
            "artifact_license": "AGPL-3.0-or-later",
        },
    )

    entry = extract_model_entry(path)

    assert entry["filename"] == "model.onnx"
    assert entry["input_size"] == 640
    assert entry["class_names"] == {"0": "dark_gill", "1": "white_spot"}
    assert entry["output_layout"] == "ultralytics_v8_detect_v1"
    assert entry["anchors"] == []
    assert entry["opset"] == 17


def test_custom_anchor_model_requires_embedded_anchors(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    _write_model(
        path,
        channels=7,
        metadata={
            "task": "detect",
            "names": json.dumps({"0": "dark_gill", "1": "white_spot"}),
            "artifact_license": "AGPL-3.0-or-later",
        },
    )

    with pytest.raises(MetadataError, match="anchors"):
        extract_model_entry(path)
