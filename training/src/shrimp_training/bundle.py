# SPDX-License-Identifier: AGPL-3.0-or-later
"""Atomic, bounded, path-safe return bundles for private model transfer."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import IO, Any, cast

from shrimp_training.acceptance import AcceptanceError, validate_mapping_documents

_REQUIRED_INPUTS = {
    "model/model.onnx",
    "records/evaluation-pytorch.json",
    "records/evaluation-onnx.json",
    "records/parity.json",
    "records/prepared-manifest.json",
    "records/mapping-acceptance.json",
    "records/evidence-report.json",
    "records/training-profile.json",
    "records/preflight.json",
    "records/environment.json",
    "records/anchors.json",
}
_GENERATED = {"registry-entry.json", "checksums.sha256", "bundle-manifest.json"}
_MAX_ENTRIES = 20
_MAX_FILE_BYTES = 200 * 1024 * 1024
_MAX_TOTAL_BYTES = 256 * 1024 * 1024
_MAX_JSON_BYTES = 1024 * 1024

# Every bundled record is fixed-size except the prepared dataset manifest, which carries one
# entry per canonical image and per duplicate, so it grows with the dataset. Measured on the
# reference prepared dataset: 1,149 images and 746 duplicates produce 1,246 KiB, about
# 1.1 KiB per record, most of it the nested source path repeated per entry. Every other
# record stays far below 1 MiB -- preflight 0.2 KiB, evaluations 1.2 KiB, parity 1.4 KiB,
# evidence report 38 KiB -- so the tight bound is kept for them and only this one record is
# allowed to scale. 32 MiB covers roughly thirty thousand images while remaining trivially
# parseable, and the per-file, total-size, entry-count and compression-ratio limits are
# unchanged.
_MAX_DATASET_MANIFEST_BYTES = 32 * 1024 * 1024
_DATASET_MANIFEST_ENTRY = "records/prepared-manifest.json"
_MAX_COMPRESSION_RATIO = 200
_CHUNK_BYTES = 1024 * 1024
_ALLOWED_COMPRESSION = {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
_ZIP_TIME = (2026, 7, 30, 0, 0, 0)


class BundleError(RuntimeError):
    """A return bundle is incomplete, unsafe, or internally inconsistent."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _stream_digest(handle: IO[bytes]) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    while chunk := handle.read(_CHUNK_BYTES):
        size += len(chunk)
        digest.update(chunk)
    return digest.hexdigest(), size


def _file_digest(path: Path) -> tuple[str, int]:
    try:
        with path.open("rb") as handle:
            return _stream_digest(handle)
    except OSError as exc:
        raise BundleError(f"bundle input became unreadable: {path}") from exc


def _json_bytes(document: dict[str, Any]) -> bytes:
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()


def _json_bound(name: str) -> int:
    """Bounded JSON size for one bundle entry, by name."""
    if name == _DATASET_MANIFEST_ENTRY:
        return _MAX_DATASET_MANIFEST_BYTES
    return _MAX_JSON_BYTES


def _bounded_file_bytes(path: Path, label: str) -> bytes:
    try:
        size = path.stat().st_size
        if size > _json_bound(label):
            raise BundleError(f"{label} exceeds the bounded JSON size limit")
        return path.read_bytes()
    except OSError as exc:
        raise BundleError(f"{label} is unreadable") from exc


def _json_document(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BundleError(f"{label} is not readable JSON") from exc
    if not isinstance(value, dict):
        raise BundleError(f"{label} must be a JSON object")
    return value


def _digest_field(document: dict[str, Any], field: str, label: str) -> str:
    value = document.get(field)
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value.casefold())
    ):
        raise BundleError(f"{label} {field} is not a SHA-256 digest")
    return value.casefold()


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=_ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o600) << 16
    return info


def _acceptance_status(payloads: dict[str, bytes]) -> str:
    acceptance_payload = payloads["records/mapping-acceptance.json"]
    evidence_payload = payloads["records/evidence-report.json"]
    prepared_payload = payloads["records/prepared-manifest.json"]
    acceptance_document = _json_document(acceptance_payload, "mapping acceptance")
    prepared_document = _json_document(prepared_payload, "prepared manifest")
    if acceptance_document.get("evidence_report") != "evidence-report.json":
        raise BundleError("mapping acceptance evidence path must name evidence-report.json")
    try:
        validated = validate_mapping_documents(
            acceptance_document,
            evidence_payload,
            prepared_document,
            evidence_report=Path("records/evidence-report.json"),
        )
    except AcceptanceError as exc:
        raise BundleError(f"mapping acceptance is invalid: {exc}") from exc
    return validated.mapping_status


def _validate_semantics(
    payloads: dict[str, bytes],
    inventory: dict[str, dict[str, str | int]],
) -> str:
    mapping_status = _acceptance_status(payloads)
    prepared_digest = str(inventory["records/prepared-manifest.json"]["sha256"])
    model_digest = str(inventory["model/model.onnx"]["sha256"])
    inventory_digest: str | None = None
    evaluations: dict[str, dict[str, Any]] = {}
    for kind in ("pytorch", "onnx"):
        name = f"records/evaluation-{kind}.json"
        evaluation = _json_document(payloads[name], f"{kind} evaluation")
        if evaluation.get("schema_version") != "1.1.0" or evaluation.get("split") != "test":
            raise BundleError(f"{kind} evaluation is not a versioned test-split record")
        if (
            evaluation.get("test_image_count") is None
            or type(evaluation["test_image_count"]) is not int
        ):
            raise BundleError(f"{kind} evaluation test image count is invalid")
        if evaluation["test_image_count"] < 1:
            raise BundleError(f"{kind} evaluation test image count is invalid")
        if (
            _digest_field(evaluation, "prepared_manifest_sha256", f"{kind} evaluation")
            != prepared_digest
        ):
            raise BundleError(f"{kind} evaluation does not match the prepared manifest")
        current_inventory = _digest_field(
            evaluation, "dataset_inventory_sha256", f"{kind} evaluation"
        )
        if inventory_digest is None:
            inventory_digest = current_inventory
        elif current_inventory != inventory_digest:
            raise BundleError("PyTorch and ONNX evaluations use different dataset inventories")
        evaluations[kind] = evaluation
    if evaluations["pytorch"].get("dataset_descriptor_sha256") != evaluations["onnx"].get(
        "dataset_descriptor_sha256"
    ):
        raise BundleError("PyTorch and ONNX evaluations use different dataset descriptors")
    if evaluations["pytorch"]["test_image_count"] != evaluations["onnx"]["test_image_count"]:
        raise BundleError("PyTorch and ONNX evaluations use different test image counts")
    if _digest_field(evaluations["onnx"], "artifact_sha256", "ONNX evaluation") != model_digest:
        raise BundleError("ONNX evaluation artifact digest does not match bundled ONNX")

    parity = _json_document(payloads["records/parity.json"], "parity record")
    if parity.get("schema_version") != "1.1.0" or parity.get("passed") is not True:
        raise BundleError("parity record is absent or does not pass")
    expected_evaluation_digests = {
        "pytorch_evaluation_sha256": inventory["records/evaluation-pytorch.json"]["sha256"],
        "onnx_evaluation_sha256": inventory["records/evaluation-onnx.json"]["sha256"],
    }
    for field, expected in expected_evaluation_digests.items():
        if _digest_field(parity, field, "parity record") != expected:
            raise BundleError("parity record is not bound to the bundled evaluation records")
    if _digest_field(parity, "prepared_manifest_sha256", "parity record") != prepared_digest:
        raise BundleError("parity record does not match the prepared manifest")
    if _digest_field(parity, "dataset_inventory_sha256", "parity record") != inventory_digest:
        raise BundleError("parity record does not match the evaluated dataset inventory")

    environment = _json_document(payloads["records/environment.json"], "environment record")
    if (
        _digest_field(environment, "prepared_manifest_sha256", "environment record")
        != prepared_digest
    ):
        raise BundleError("environment record does not match the prepared manifest")
    if (
        _digest_field(environment, "dataset_inventory_sha256", "environment record")
        != inventory_digest
    ):
        raise BundleError("environment record does not match the evaluated dataset inventory")

    registry_payload = payloads.get("registry-entry.json")
    if registry_payload is not None:
        registry = _json_document(registry_payload, "registry entry")
        if registry.get("sha256") != model_digest:
            raise BundleError("registry model hash does not match bundled ONNX")
        if registry.get("dataset_mapping_status") != mapping_status:
            raise BundleError("registry mapping status does not match acceptance")
    return mapping_status


def _publish_no_clobber(temporary: Path, destination: Path) -> None:
    """Atomically publish within one filesystem without replacing an existing path."""
    try:
        os.link(temporary, destination)
    except FileExistsError:
        raise FileExistsError(f"refusing to replace return bundle: {destination}") from None
    except OSError as exc:
        raise BundleError(f"could not atomically publish return bundle: {destination}") from exc
    temporary.unlink()


def _validate_source_inputs(files: dict[str, Path]) -> int:
    total_input_size = 0
    for name, path in files.items():
        if not path.is_file() or path.is_symlink():
            raise BundleError(f"bundle input is missing or a symlink: {name}")
        size = path.stat().st_size
        if size > _MAX_FILE_BYTES:
            raise BundleError("a bundle input exceeds the per-file size limit")
        if name != "model/model.onnx" and size > _json_bound(name):
            raise BundleError(f"{name} exceeds the bounded JSON size limit")
        total_input_size += size
    if total_input_size > _MAX_TOTAL_BYTES:
        raise BundleError("bundle exceeds the total uncompressed size limit")
    return total_input_size


def build_return_bundle(
    destination: Path,
    files: dict[str, Path],
    *,
    model_id: str,
    version: str,
    input_size: int,
    toolchain: str,
    anchors: list[list[float]],
) -> Path:
    """Build, verify, and no-clobber publish a deterministic private transfer archive."""
    if destination.exists():
        raise FileExistsError(f"refusing to replace return bundle: {destination}")
    if set(files) != _REQUIRED_INPUTS:
        missing = sorted(_REQUIRED_INPUTS - set(files))
        unknown = sorted(set(files) - _REQUIRED_INPUTS)
        raise BundleError(
            f"bundle inputs differ from contract; missing={missing}, unknown={unknown}"
        )
    if not model_id.strip() or not version.strip() or not toolchain.strip():
        raise BundleError("model_id, version, and toolchain must be non-empty")
    if input_size < 32 or input_size % 32:
        raise BundleError("input_size must be positive and divisible by 32")
    if not anchors or any(len(pair) != 2 for pair in anchors):
        raise BundleError("anchors must be a non-empty list of (width, height) pairs")

    total_input_size = _validate_source_inputs(files)

    payloads = {
        name: _bounded_file_bytes(path, name)
        for name, path in files.items()
        if name != "model/model.onnx"
    }
    inventory: dict[str, dict[str, str | int]] = {}
    for name, path in sorted(files.items()):
        digest, size = _file_digest(path)
        inventory[name] = {"sha256": digest, "size": size}
    mapping_status = _acceptance_status(payloads)
    registry = {
        "model_id": model_id.strip(),
        "version": version.strip(),
        "filename": "model.onnx",
        "sha256": inventory["model/model.onnx"]["sha256"],
        "input_size": input_size,
        "class_names": {"0": "dark_gill", "1": "white_spot"},
        "opset": 17,
        "output_layout": "custom_yolo_anchor_v1",
        "anchors": [[float(w), float(h)] for w, h in anchors],
        "dataset_mapping_status": mapping_status,
        "artifact_license": "AGPL-3.0-or-later",
        "training_toolchain": toolchain.strip(),
    }
    registry_payload = _json_bytes(registry)
    inventory["registry-entry.json"] = {
        "sha256": _sha256(registry_payload),
        "size": len(registry_payload),
    }
    payloads["registry-entry.json"] = registry_payload
    _validate_semantics(payloads, inventory)

    checksum_payload = "".join(
        f"{metadata['sha256']}  {name}\n" for name, metadata in sorted(inventory.items())
    ).encode()
    manifest_payload = _json_bytes(
        {
            "schema_version": "1.0.0",
            "model_id": model_id.strip(),
            "version": version.strip(),
            "files": inventory,
        }
    )
    generated = {
        "registry-entry.json": registry_payload,
        "checksums.sha256": checksum_payload,
        "bundle-manifest.json": manifest_payload,
    }
    if total_input_size + sum(map(len, generated.values())) > _MAX_TOTAL_BYTES:
        raise BundleError("bundle exceeds the total uncompressed size limit")

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}-", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
            for name, path in sorted(files.items()):
                with path.open("rb") as source, archive.open(_zip_info(name), "w") as target:
                    shutil.copyfileobj(source, target, length=_CHUNK_BYTES)
            for name, payload in sorted(generated.items()):
                archive.writestr(_zip_info(name), payload)
        # Windows FlushFileBuffers requires a writable OS handle for os.fsync().
        with temporary.open("r+b") as archive_handle:
            os.fsync(archive_handle.fileno())
        verify_return_bundle(
            temporary,
            expected_manifest_sha256=_sha256(manifest_payload),
        )
        _publish_no_clobber(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _safe_name(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(name) and "\\" not in name and not path.is_absolute() and ".." not in path.parts


def _bounded_zip_payload(archive: zipfile.ZipFile, info: zipfile.ZipInfo, label: str) -> bytes:
    # The declared size is attacker-controlled, so the read is capped independently and the
    # result rechecked. Both use the bound for this entry name, not the declared name.
    bound = _json_bound(info.filename)
    if info.file_size > bound:
        raise BundleError(f"bundle must contain exactly one bounded {label}")
    try:
        with archive.open(info) as handle:
            payload = handle.read(bound + 1)
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise BundleError(f"bundle {label} is unreadable") from exc
    if len(payload) > bound:
        raise BundleError(f"bundle must contain exactly one bounded {label}")
    return payload


def bundle_manifest_sha256(path: Path) -> str:
    """Return the bounded embedded manifest digest for separate-channel comparison."""
    try:
        with zipfile.ZipFile(path) as archive:
            matches = [
                info for info in archive.infolist() if info.filename == "bundle-manifest.json"
            ]
            if len(matches) != 1:
                raise BundleError("bundle must contain exactly one bounded manifest")
            info = matches[0]
            if info.compress_type not in _ALLOWED_COMPRESSION:
                raise BundleError("bundle manifest uses unsupported compression")
            return _sha256(_bounded_zip_payload(archive, info, "manifest"))
    except zipfile.BadZipFile as exc:
        raise BundleError("return bundle is not a valid ZIP archive") from exc


def _validate_archive_infos(infos: list[zipfile.ZipInfo]) -> zipfile.ZipInfo:
    names = [info.filename for info in infos]
    if (
        len(names) > _MAX_ENTRIES
        or len(names) != len(set(names))
        or len(names) != len({name.casefold() for name in names})
    ):
        raise BundleError("bundle has too many or duplicate entries")
    if any(not _safe_name(name) for name in names):
        raise BundleError("bundle contains an unsafe archive path")
    if any(stat.S_ISLNK(info.external_attr >> 16) for info in infos):
        raise BundleError("bundle contains an unsafe symlink")
    if any(info.flag_bits & 0x1 for info in infos):
        raise BundleError("bundle contains an encrypted entry")
    if any(info.compress_type not in _ALLOWED_COMPRESSION for info in infos):
        raise BundleError("bundle contains an unsupported compression method")
    if set(names) != _REQUIRED_INPUTS | _GENERATED:
        raise BundleError("bundle entries do not match the required contract")
    if any(info.file_size > _MAX_FILE_BYTES for info in infos):
        raise BundleError("bundle entry exceeds the size limit")
    if sum(info.file_size for info in infos) > _MAX_TOTAL_BYTES:
        raise BundleError("bundle exceeds the total size limit")
    manifest_info = next(info for info in infos if info.filename == "bundle-manifest.json")
    if manifest_info.file_size > _MAX_JSON_BYTES:
        raise BundleError("bundle must contain exactly one bounded manifest")
    if any(
        info.filename != "model/model.onnx" and info.file_size > _json_bound(info.filename)
        for info in infos
    ):
        raise BundleError("bundle JSON or metadata entry exceeds its bounded size limit")
    if any(
        info.file_size > _MAX_JSON_BYTES
        and info.file_size / max(info.compress_size, 1) > _MAX_COMPRESSION_RATIO
        for info in infos
    ):
        raise BundleError("bundle entry exceeds the compression-ratio limit")
    return manifest_info


def verify_return_bundle(
    path: Path,
    *,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    """Stream-verify a bundle against a separately supplied manifest digest."""
    if len(expected_manifest_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_manifest_sha256
    ):
        raise BundleError("expected external manifest digest must be 64 lowercase hex characters")
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            manifest_info = _validate_archive_infos(infos)

            manifest_payload = _bounded_zip_payload(archive, manifest_info, "manifest")
            if not hmac.compare_digest(_sha256(manifest_payload), expected_manifest_sha256):
                raise BundleError("bundle does not match the expected external manifest digest")
            manifest = _json_document(manifest_payload, "bundle manifest")
            if manifest.get("schema_version") != "1.0.0":
                raise BundleError("bundle manifest schema is invalid")
            inventory_value = manifest.get("files")
            expected_inventory = _REQUIRED_INPUTS | {"registry-entry.json"}
            if not isinstance(inventory_value, dict) or set(inventory_value) != expected_inventory:
                raise BundleError("bundle manifest inventory is invalid")
            inventory = cast(dict[str, Any], inventory_value)

            payloads: dict[str, bytes] = {}
            actual: dict[str, dict[str, str | int]] = {}
            for info in infos:
                name = info.filename
                if name == "bundle-manifest.json":
                    continue
                with archive.open(info) as handle:
                    digest, size = _stream_digest(handle)
                actual[name] = {"sha256": digest, "size": size}
                if name != "model/model.onnx":
                    payloads[name] = _bounded_zip_payload(archive, info, name)
    except zipfile.BadZipFile as exc:
        raise BundleError("return bundle is not a valid ZIP archive") from exc

    for name, metadata in inventory.items():
        if not isinstance(metadata, dict):
            raise BundleError("bundle manifest file metadata is invalid")
        if metadata != actual[name]:
            raise BundleError(f"bundle hash or size mismatch: {name}")
    expected_checksums = "".join(
        f"{metadata['sha256']}  {name}\n" for name, metadata in sorted(inventory.items())
    ).encode()
    if payloads["checksums.sha256"] != expected_checksums:
        raise BundleError("bundle checksum inventory does not match its manifest")
    typed_inventory: dict[str, dict[str, str | int]] = {
        name: {"sha256": str(metadata["sha256"]), "size": int(metadata["size"])}
        for name, metadata in inventory.items()
    }
    _validate_semantics(payloads, typed_inventory)
    return manifest
