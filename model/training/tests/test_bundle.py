from __future__ import annotations

import hashlib
import importlib
import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

from shrimp_training import bundle as bundle_module
from shrimp_training.bundle import (
    BundleError,
    build_return_bundle,
    bundle_manifest_sha256,
    verify_return_bundle,
)

ANCHOR_BOXES: list[list[float]] = [[10.0 * (i + 1), 10.0 * (i + 1)] for i in range(9)]


def _json(document: dict[str, Any]) -> bytes:
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _payloads(
    *,
    fabricated_acceptance: bool = False,
    wrong_onnx_digest: bool = False,
    prepared_padding: int = 0,
) -> dict[str, bytes]:
    model = b"onnx"
    evidence = _json({"schema_version": "1.0.0", "overlays": 60})
    acceptance: dict[str, Any] = {
        "schema_version": "1.0.0",
        "mapping_status": "PROVISIONAL_UNCONFIRMED",
        "accepted_mapping": {"0": "dark_gill", "1": "white_spot"},
        "provisional_mapping_acknowledged": True,
        "author_confirmed": False,
        "annotation_convention_acknowledged": True,
        "acknowledgement": "Class semantics and annotation drift reviewed.",
        "evidence_report": "evidence-report.json",
        "evidence_report_sha256": _sha(evidence),
        "overlay_sheets_reviewed": 60,
        "reviewer": "Independent aquatic imaging reviewer",
        "reviewed_on": "2026-07-30",
    }
    prepared = {
        "schema_version": "1.0.0",
        "mapping_acceptance": {
            "status": acceptance["mapping_status"],
            "reviewer": acceptance["reviewer"],
            "reviewed_on": acceptance["reviewed_on"],
            "evidence_report_sha256": acceptance["evidence_report_sha256"],
        },
        "classes": {"0": "dark_gill", "1": "white_spot"},
    }
    if fabricated_acceptance:
        acceptance = {"mapping_status": "PROVISIONAL_UNCONFIRMED"}
        prepared["mapping_acceptance"] = {"status": "PROVISIONAL_UNCONFIRMED"}
    if prepared_padding:
        # Stand in for the per-image records a real prepared manifest carries. Distinct
        # digests, not repeated filler, so the entry compresses like real content and stays
        # subject to the compression-ratio guard. Applied before the digest is taken so every
        # record that binds to the manifest stays consistent.
        prepared["images"] = [
            hashlib.sha256(str(index).encode()).hexdigest() for index in range(prepared_padding)
        ]
    prepared_payload = _json(prepared)
    inventory_digest = "d" * 64
    evaluation_common = {
        "schema_version": "1.1.0",
        "dataset_descriptor_sha256": "a" * 64,
        "prepared_manifest_sha256": _sha(prepared_payload),
        "dataset_inventory_sha256": inventory_digest,
        "test_image_count": 3,
        "split": "test",
        "metrics": {"precision": 0.5, "recall": 0.4, "map50": 0.42, "map50_95": 0.275},
        "per_class_map50_95": [0.21, 0.34],
        "threshold_matched_metrics": {"precision@0.25": 0.5, "recall@0.25": 0.4},
    }
    evaluation_pytorch = _json({**evaluation_common, "artifact_sha256": "b" * 64})
    evaluation_onnx = _json(
        {
            **evaluation_common,
            "artifact_sha256": "f" * 64 if wrong_onnx_digest else _sha(model),
        }
    )
    parity = _json(
        {
            "schema_version": "1.1.0",
            "passed": True,
            "tolerance": 0.01,
            "maximum_delta": 0.0,
            "deltas": {
                "precision": 0.0,
                "recall": 0.0,
                "map50": 0.0,
                "map50_95": 0.0,
                "class_0_map50_95": 0.0,
                "class_1_map50_95": 0.0,
            },
            "pytorch_evaluation_sha256": _sha(evaluation_pytorch),
            "onnx_evaluation_sha256": _sha(evaluation_onnx),
            "prepared_manifest_sha256": _sha(prepared_payload),
            "dataset_inventory_sha256": inventory_digest,
        }
    )
    return {
        "model/model.onnx": model,
        "records/evaluation-pytorch.json": evaluation_pytorch,
        "records/evaluation-onnx.json": evaluation_onnx,
        "records/parity.json": parity,
        "records/prepared-manifest.json": prepared_payload,
        "records/mapping-acceptance.json": _json(acceptance),
        "records/evidence-report.json": evidence,
        "records/training-profile.json": _json({"profile_name": "compact-nvidia-6gb"}),
        "records/preflight.json": _json({"schema_version": "1.0.0", "status": "READY"}),
        "records/environment.json": _json(
            {
                "schema_version": "1.0.0",
                "prepared_manifest_sha256": _sha(prepared_payload),
                "dataset_inventory_sha256": inventory_digest,
            }
        ),
        "records/anchors.json": _json(
            {
                "schema_version": "1.0.0",
                "strides": [8, 16, 32],
                "anchors_per_scale": 3,
                "boxes": ANCHOR_BOXES,
                "seed": 20260730,
                "source_box_count": 100,
            }
        ),
    }


def _inputs(root: Path, **options: bool | int) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {}
    for archive_name, payload in _payloads(**options).items():
        path = root / archive_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        files[archive_name] = path
    return files


def _build(destination: Path, inputs: dict[str, Path]) -> Path:
    return build_return_bundle(
        destination,
        inputs,
        model_id="shrimp-marker-custom-yolo",
        version="1.0.0",
        input_size=640,
        toolchain="custom-pytorch-yolo torch 2.5.1+cu118",
        anchors=ANCHOR_BOXES,
    )


def _repack(
    source_path: Path,
    destination: Path,
    *,
    compression: int = zipfile.ZIP_DEFLATED,
    replacements: dict[str, bytes] | None = None,
) -> None:
    replacements = replacements or {}
    with (
        zipfile.ZipFile(source_path) as source,
        zipfile.ZipFile(destination, "w", compression=compression) as target,
    ):
        for name in source.namelist():
            target.writestr(name, replacements.get(name, source.read(name)))


def test_bundle_is_atomic_checksummed_and_emits_runtime_registry_entry(tmp_path: Path) -> None:
    destination = tmp_path / "return" / "shrimp-model-v1.zip"
    result = _build(destination, _inputs(tmp_path))
    manifest_digest = bundle_manifest_sha256(result)
    manifest = verify_return_bundle(result, expected_manifest_sha256=manifest_digest)

    assert result == destination
    assert manifest["schema_version"] == "1.0.0"
    with zipfile.ZipFile(result) as archive:
        names = set(archive.namelist())
        assert "records/evidence-report.json" in names
        assert not any(name.lower().endswith((".pt", ".pth", ".pkl")) for name in names)
        registry_entry = json.loads(archive.read("registry-entry.json"))
        assert registry_entry["sha256"] == manifest["files"]["model/model.onnx"]["sha256"]

    with pytest.raises(FileExistsError):
        _build(destination, _inputs(tmp_path / "again"))


def test_generated_registry_entry_loads_in_the_runtime_parser(tmp_path: Path) -> None:
    try:
        registry_module = importlib.import_module("shrimp_screening.detection.registry")
    except ModuleNotFoundError:
        pytest.skip("backend package is not installed in the standalone training environment")
    bundle = _build(tmp_path / "valid.zip", _inputs(tmp_path / "inputs"))
    with zipfile.ZipFile(bundle) as archive:
        registry_entry = json.loads(archive.read("registry-entry.json"))
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps({"models": [registry_entry]}), encoding="utf-8")

    loaded = registry_module.load_registry(registry_path).by_model_id("shrimp-marker-custom-yolo")

    assert loaded.filename == "model.onnx"
    assert loaded.output_layout.value == "custom_yolo_anchor_v1"
    assert len(loaded.anchors) == 9


def test_build_rejects_fabricated_acceptance_and_wrong_onnx_evaluation(tmp_path: Path) -> None:
    with pytest.raises(BundleError, match="acceptance"):
        _build(
            tmp_path / "fabricated.zip",
            _inputs(tmp_path / "fabricated", fabricated_acceptance=True),
        )
    with pytest.raises(BundleError, match="ONNX evaluation"):
        _build(
            tmp_path / "wrong-onnx.zip",
            _inputs(tmp_path / "wrong-onnx", wrong_onnx_digest=True),
        )


def test_build_rejects_oversized_input_before_reading_it(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path / "inputs")
    with inputs["model/model.onnx"].open("wb") as handle:
        handle.truncate(200 * 1024 * 1024 + 1)

    with pytest.raises(BundleError, match="per-file size"):
        _build(tmp_path / "oversized.zip", inputs)


def test_dataset_manifest_may_exceed_one_megabyte_but_other_records_may_not(
    tmp_path: Path,
) -> None:
    """Only the prepared manifest scales with the dataset, so only it gets the larger bound.

    A real 1,149-image prepared manifest is about 1,246 KiB. Records that are fixed-size
    regardless of dataset size stay bounded at 1 MiB.
    """
    # Roughly 20,000 records, comfortably past 1 MiB.
    inputs = _inputs(tmp_path / "big-manifest", prepared_padding=20_000)
    assert inputs["records/prepared-manifest.json"].stat().st_size > 1024 * 1024
    # Accepted: the prepared manifest is allowed to grow with the dataset.
    bundle = _build(tmp_path / "big.zip", inputs)
    # And the bundle it produces must still verify, so the same bound applies on both sides.
    verify_return_bundle(bundle, expected_manifest_sha256=bundle_manifest_sha256(bundle))

    # Rejected: a fixed-size record has no reason to be large.
    fixed = _inputs(tmp_path / "big-preflight")
    fixed["records/preflight.json"].write_bytes(b'{"a":"' + b" " * (1024 * 1024 + 1) + b'"}')
    with pytest.raises(BundleError, match="bounded JSON size limit"):
        _build(tmp_path / "big-preflight.zip", fixed)

    assert bundle_module._json_bound("records/prepared-manifest.json") == 32 * 1024 * 1024
    assert bundle_module._json_bound("records/preflight.json") == 1024 * 1024


def test_atomic_publication_never_clobbers_existing_destination(tmp_path: Path) -> None:
    temporary = tmp_path / "temporary.zip"
    destination = tmp_path / "destination.zip"
    temporary.write_bytes(b"candidate")
    destination.write_bytes(b"rival")

    with pytest.raises(FileExistsError):
        bundle_module._publish_no_clobber(temporary, destination)

    assert destination.read_bytes() == b"rival"
    assert temporary.read_bytes() == b"candidate"


def test_verify_bundle_rejects_traversal_tampering_and_unsupported_compression(
    tmp_path: Path,
) -> None:
    traversal = tmp_path / "traversal.zip"
    with zipfile.ZipFile(traversal, "w") as archive:
        archive.writestr("../escape", b"bad")
    with pytest.raises(BundleError, match="unsafe"):
        verify_return_bundle(traversal, expected_manifest_sha256="0" * 64)

    bundle = _build(tmp_path / "valid.zip", _inputs(tmp_path / "inputs"))
    digest = bundle_manifest_sha256(bundle)
    tampered = tmp_path / "tampered.zip"
    _repack(bundle, tampered, replacements={"model/model.onnx": b"different"})
    with pytest.raises(BundleError, match="hash"):
        verify_return_bundle(tampered, expected_manifest_sha256=digest)

    bzip = tmp_path / "bzip.zip"
    _repack(bundle, bzip, compression=zipfile.ZIP_BZIP2)
    with pytest.raises(BundleError, match="compression"):
        verify_return_bundle(bzip, expected_manifest_sha256=digest)


def test_verify_rejects_oversized_manifest_before_reading_payload(tmp_path: Path) -> None:
    bundle = _build(tmp_path / "valid.zip", _inputs(tmp_path / "inputs"))
    oversized = b"{" + b" " * (1024 * 1024) + b"}"
    repacked = tmp_path / "oversized-manifest.zip"
    _repack(bundle, repacked, replacements={"bundle-manifest.json": oversized})

    with pytest.raises(BundleError, match="bounded manifest"):
        verify_return_bundle(repacked, expected_manifest_sha256=_sha(oversized))


def test_verify_bundle_requires_the_expected_external_manifest_digest(tmp_path: Path) -> None:
    bundle = _build(tmp_path / "valid.zip", _inputs(tmp_path / "inputs"))

    with pytest.raises(BundleError, match="external manifest digest"):
        verify_return_bundle(bundle, expected_manifest_sha256="f" * 64)
