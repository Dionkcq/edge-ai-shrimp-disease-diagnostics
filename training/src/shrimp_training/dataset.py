# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fail-closed validation of the prepared dataset consumed by training."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_PARTITIONS = ("train", "validation", "test")
_EXPECTED_CLASSES = {0: "dark_gill", 1: "white_spot"}
_REQUIRED_IMAGE_FIELDS = {
    "specimen_key",
    "sha256",
    "partition",
    "image_output",
    "image_output_sha256",
    "label_output",
    "label_output_sha256",
}


class DatasetContractError(RuntimeError):
    """The prepared dataset cannot be trusted for model training."""


@dataclass(frozen=True, slots=True)
class PreparedDataset:
    root: Path
    manifest: Path
    manifest_sha256: str
    inventory_sha256: str
    image_count: int
    specimen_count: int
    split_counts: dict[str, int]
    class_names: dict[int, str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise DatasetContractError(f"cannot read required dataset file: {path}") from exc
    return digest.hexdigest()


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DatasetContractError(f"{label} must be an object")
    return value


def _inventory_sha256(images: list[Any]) -> str:
    fields = (
        "partition",
        "image_output",
        "image_output_sha256",
        "label_output",
        "label_output_sha256",
    )
    rows = [
        {field: record[field] for field in fields}
        for record in (_object(raw, "image record") for raw in images)
    ]
    rows.sort(key=lambda row: str(row["image_output"]))
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _safe_file(root: Path, raw: Any, label: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise DatasetContractError(f"{label} must be a non-empty relative path")
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise DatasetContractError(f"{label} escapes the prepared dataset root")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise DatasetContractError(f"{label} is missing or escapes the prepared dataset root")
    return path


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise DatasetContractError(f"{label} must be a SHA-256 string")
    digest = value.casefold()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise DatasetContractError(f"{label} must be a 64-character hexadecimal SHA-256")
    return digest


def _classes(value: Any) -> dict[int, str]:
    document = _object(value, "classes")
    try:
        parsed = {int(key): name for key, name in document.items()}
    except (TypeError, ValueError) as exc:
        raise DatasetContractError("classes must use integer-like keys") from exc
    if parsed != _EXPECTED_CLASSES:
        raise DatasetContractError(
            f"classes must be exactly {_EXPECTED_CLASSES}; refusing a class-order change"
        )
    return parsed


def _label_classes(path: Path, index: int) -> set[int]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise DatasetContractError(f"images[{index}] label cannot be read as UTF-8") from exc
    classes: set[int] = set()
    for line_number, line in enumerate(lines, start=1):
        fields = line.split()
        if len(fields) != 5:
            raise DatasetContractError(
                f"images[{index}] label line {line_number} must contain five fields"
            )
        try:
            class_id = int(fields[0])
            x, y, width, height = (float(value) for value in fields[1:])
        except ValueError as exc:
            raise DatasetContractError(
                f"images[{index}] label line {line_number} contains a non-numeric value"
            ) from exc
        if class_id not in _EXPECTED_CLASSES:
            raise DatasetContractError(
                f"images[{index}] label line {line_number} has an unknown class"
            )
        if not all(math.isfinite(value) for value in (x, y, width, height)):
            raise DatasetContractError(
                f"images[{index}] label line {line_number} contains a non-finite coordinate"
            )
        if not all(0.0 <= value <= 1.0 for value in (x, y, width, height)):
            raise DatasetContractError(
                f"images[{index}] label line {line_number} is outside [0, 1]"
            )
        if width <= 0.0 or height <= 0.0:
            raise DatasetContractError(
                f"images[{index}] label line {line_number} has a non-positive box"
            )
        if x - width / 2 < 0.0 or x + width / 2 > 1.0:
            raise DatasetContractError(
                f"images[{index}] label line {line_number} exceeds horizontal bounds"
            )
        if y - height / 2 < 0.0 or y + height / 2 > 1.0:
            raise DatasetContractError(
                f"images[{index}] label line {line_number} exceeds vertical bounds"
            )
        classes.add(class_id)
    return classes


def _validate_image_record(
    root: Path, raw_record: Any, index: int
) -> tuple[str, str, str, set[int]]:
    record = _object(raw_record, f"images[{index}]")
    missing = _REQUIRED_IMAGE_FIELDS - set(record)
    if missing:
        raise DatasetContractError(f"images[{index}] is missing {sorted(missing)[0]}")
    partition = record["partition"]
    if partition not in _PARTITIONS:
        raise DatasetContractError(f"images[{index}].partition is invalid")
    assert isinstance(partition, str)
    specimen = record["specimen_key"]
    if not isinstance(specimen, str) or not specimen.strip():
        raise DatasetContractError(f"images[{index}].specimen_key is invalid")
    image = _safe_file(root, record["image_output"], f"images[{index}].image_output")
    label = _safe_file(root, record["label_output"], f"images[{index}].label_output")
    image_digest = _digest(record["image_output_sha256"], "image_output_sha256")
    source_digest = _digest(record["sha256"], "sha256")
    label_digest = _digest(record["label_output_sha256"], "label_output_sha256")
    if _sha256(image) != image_digest or image_digest != source_digest:
        raise DatasetContractError(f"images[{index}] image hash does not match its manifest")
    if _sha256(label) != label_digest:
        raise DatasetContractError(f"images[{index}] label hash does not match its manifest")
    return partition, specimen, image_digest, _label_classes(label, index)


def _validate_partition_coverage(
    counts: dict[str, int],
    classes: dict[str, set[int]],
    negative_counts: dict[str, int],
) -> None:
    required_classes = set(_EXPECTED_CLASSES)
    for partition in _PARTITIONS:
        if counts[partition] == 0:
            raise DatasetContractError(f"partition {partition} is empty")
        if classes[partition] != required_classes:
            raise DatasetContractError(f"partition {partition} must contain both target classes")
        if negative_counts[partition] == 0:
            raise DatasetContractError(
                f"partition {partition} must contain at least one negative image"
            )


def validate_prepared_dataset(root: Path) -> PreparedDataset:
    """Validate every training input against the preparation manifest."""
    resolved = root.resolve()
    manifest_path = resolved / "manifest.json"
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise DatasetContractError(f"prepared manifest cannot be read: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise DatasetContractError("prepared manifest is not valid JSON") from exc
    manifest = _object(document, "manifest")
    if manifest.get("schema_version") != "1.0.0":
        raise DatasetContractError("prepared manifest schema_version must be 1.0.0")
    class_names = _classes(manifest.get("classes"))
    images = manifest.get("images")
    if not isinstance(images, list) or not images:
        raise DatasetContractError("prepared manifest must contain at least one image")

    observed_counts: dict[str, int] = {"train": 0, "validation": 0, "test": 0}
    observed_classes: dict[str, set[int]] = {partition: set() for partition in _PARTITIONS}
    negative_counts: dict[str, int] = dict.fromkeys(_PARTITIONS, 0)
    specimen_partitions: dict[str, set[str]] = {}
    image_digests: set[str] = set()
    for index, raw_record in enumerate(images):
        partition, specimen, image_digest, label_classes = _validate_image_record(
            resolved, raw_record, index
        )
        if image_digest in image_digests:
            raise DatasetContractError("a canonical image hash appears more than once")
        image_digests.add(image_digest)
        observed_counts[partition] += 1
        observed_classes[partition].update(label_classes)
        if not label_classes:
            negative_counts[partition] += 1
        specimen_partitions.setdefault(specimen, set()).add(partition)

    _validate_partition_coverage(observed_counts, observed_classes, negative_counts)

    crossing = sorted(
        key for key, partitions in specimen_partitions.items() if len(partitions) != 1
    )
    if crossing:
        raise DatasetContractError(f"specimen crosses partitions: {crossing[0]}")
    split = _object(manifest.get("split"), "split")
    declared_counts = _object(split.get("counts"), "split.counts")
    if declared_counts != observed_counts:
        raise DatasetContractError(
            f"split counts {declared_counts} do not match observed {observed_counts}"
        )
    summary = _object(manifest.get("summary"), "summary")
    if summary.get("canonical_images") != len(images):
        raise DatasetContractError("summary canonical_images does not match image records")
    return PreparedDataset(
        root=resolved,
        manifest=manifest_path,
        manifest_sha256=_sha256(manifest_path),
        inventory_sha256=_inventory_sha256(images),
        image_count=len(images),
        specimen_count=len(specimen_partitions),
        split_counts=observed_counts,
        class_names=class_names,
    )


def write_ultralytics_dataset(dataset: PreparedDataset, destination: Path) -> Path:
    """Write the private Ultralytics data descriptor without overwriting evidence."""
    rendered = (
        f"path: {json.dumps(dataset.root.as_posix())}\n"
        "train: images/train\n"
        "val: images/validation\n"
        "test: images/test\n"
        "names:\n"
        "  0: dark_gill\n"
        "  1: white_spot\n"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(rendered)
    except FileExistsError:
        raise
    except OSError as exc:
        raise DatasetContractError(
            f"cannot write Ultralytics dataset descriptor: {destination}"
        ) from exc
    return destination
