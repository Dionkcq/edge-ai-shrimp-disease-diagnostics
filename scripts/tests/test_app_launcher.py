from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from scripts.app_launcher import LauncherError, prepare_runtime


def _manifest(
    *, filename: str = "shrimp-model-v1.onnx", sha256: str | None = None
) -> dict[str, object]:
    entry: dict[str, object] = {
        "model_id": "shrimp-model-v1",
        "version": "1.0.0",
        "filename": filename,
        "input_size": 640,
        "class_names": {"0": "dark_gill", "1": "white_spot"},
        "opset": 17,
        "output_layout": "custom_yolo_anchor_v1",
        "anchors": [[10.0, 10.0]] * 9,
        "dataset_mapping_status": "verified",
        "artifact_license": "AGPL-3.0-or-later",
        "training_toolchain": "custom-pytorch-yolo",
    }
    if sha256 is not None:
        entry["sha256"] = sha256
    return entry


def test_clean_model_folder_selects_safe_unavailable_provider(tmp_path: Path) -> None:
    config = prepare_runtime(tmp_path, tmp_path / "model", tmp_path / ".runtime")

    assert config.provider == "unavailable"
    assert config.model_path is None
    assert config.registry_path is None


def test_raw_model_gets_hashed_and_registry_is_generated(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    model_path = model_dir / "shrimp-model-v1.onnx"
    model_path.write_bytes(b"test-onnx")
    (model_dir / "model-manifest.json").write_text(json.dumps(_manifest()), encoding="utf-8")

    config = prepare_runtime(tmp_path, model_dir, tmp_path / ".runtime")

    assert config.provider == "onnx"
    assert config.model_path == model_path
    assert config.registry_path is not None
    registry = json.loads(config.registry_path.read_text(encoding="utf-8"))
    assert registry["models"][0]["filename"] == model_path.name
    assert len(registry["models"][0]["sha256"]) == 64


def test_official_zip_is_extracted_and_registered(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    bundle = model_dir / "shrimp-model-v1.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("model/model.onnx", b"test-onnx")
        archive.writestr("registry-entry.json", json.dumps(_manifest(filename="model.onnx")))

    config = prepare_runtime(tmp_path, model_dir, tmp_path / ".runtime")

    assert config.provider == "onnx"
    assert config.model_path is not None
    assert config.model_path == tmp_path / ".runtime" / "model.onnx"
    assert config.model_path.read_bytes() == b"test-onnx"
    assert config.registry_path is not None
    assert json.loads(config.registry_path.read_text())["models"][0]["filename"] == "model.onnx"


def test_declared_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "shrimp-model-v1.onnx").write_bytes(b"test-onnx")
    (model_dir / "model-manifest.json").write_text(
        json.dumps(_manifest(sha256="0" * 64)), encoding="utf-8"
    )

    with pytest.raises(LauncherError, match="SHA-256"):
        prepare_runtime(tmp_path, model_dir, tmp_path / ".runtime")


def test_multiple_raw_models_are_rejected(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "one.onnx").write_bytes(b"one")
    (model_dir / "two.onnx").write_bytes(b"two")

    with pytest.raises(LauncherError, match="exactly one ONNX"):
        prepare_runtime(tmp_path, model_dir, tmp_path / ".runtime")
