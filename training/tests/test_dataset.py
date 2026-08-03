from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from shrimp_training.dataset import DatasetContractError, validate_prepared_dataset


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prepared(root: Path) -> Path:
    records: list[dict[str, object]] = []
    index = 0
    for partition in ("train", "validation", "test"):
        for class_id, source_class in ((0, "BG"), (1, "WSSV"), (None, "HEALTHY")):
            index += 1
            image = root / "images" / partition / f"image-{index}.jpg"
            label = root / "labels" / partition / f"image-{index}.txt"
            image.parent.mkdir(parents=True, exist_ok=True)
            label.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(f"image-{index}".encode())
            label.write_text(
                "" if class_id is None else f"{class_id} 0.5 0.5 0.25 0.25\n",
                encoding="utf-8",
            )
            records.append(
                {
                    "source_path": f"source/image-{index}.jpg",
                    "source_role": "raw" if class_id is None else "annotated",
                    "source_class": source_class,
                    "specimen_key": f"{source_class}:{index}",
                    "sha256": _sha(image),
                    "partition": partition,
                    "image_output": image.relative_to(root).as_posix(),
                    "image_output_sha256": _sha(image),
                    "label_output": label.relative_to(root).as_posix(),
                    "label_output_sha256": _sha(label),
                }
            )
    manifest = {
        "schema_version": "1.0.0",
        "input": {"archive": "private.zip", "sha256": "a" * 64},
        "mapping_acceptance": {
            "status": "PROVISIONAL_UNCONFIRMED",
            "reviewer": "Human reviewer",
            "reviewed_on": "2026-07-30",
            "evidence_report_sha256": "b" * 64,
        },
        "classes": {"0": "dark_gill", "1": "white_spot"},
        "split": {
            "seed": 20260730,
            "counts": {"train": 3, "validation": 3, "test": 3},
        },
        "summary": {
            "canonical_images": 9,
            "duplicate_entries": 0,
            "excluded_augmentations": 0,
        },
        "images": records,
        "duplicates": [],
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _manifest(root: Path) -> tuple[Path, dict[str, Any]]:
    path = root / "manifest.json"
    return path, json.loads(path.read_text(encoding="utf-8"))


def _rewrite_label(root: Path, record: dict[str, Any], text: str) -> None:
    label = root / record["label_output"]
    label.write_text(text, encoding="utf-8")
    record["label_output_sha256"] = _sha(label)


def test_validate_prepared_dataset_accepts_hash_verified_grouped_split(tmp_path: Path) -> None:
    result = validate_prepared_dataset(_prepared(tmp_path / "prepared"))

    assert result.image_count == 9
    assert result.specimen_count == 9
    assert result.split_counts == {"train": 3, "validation": 3, "test": 3}
    assert result.class_names == {0: "dark_gill", 1: "white_spot"}
    assert len(result.inventory_sha256) == 64

    second = validate_prepared_dataset(_prepared(tmp_path / "prepared-copy"))
    assert second.inventory_sha256 == result.inventory_sha256


@pytest.mark.parametrize("defect", ["traversal", "hash", "class_order", "group_overlap"])
def test_validate_prepared_dataset_fails_closed_on_contract_defects(
    tmp_path: Path, defect: str
) -> None:
    root = _prepared(tmp_path / "prepared")
    path, manifest = _manifest(root)
    if defect == "traversal":
        manifest["images"][0]["image_output"] = "../private.jpg"
    elif defect == "hash":
        manifest["images"][0]["image_output_sha256"] = "0" * 64
    elif defect == "class_order":
        manifest["classes"] = {"0": "white_spot", "1": "dark_gill"}
    else:
        manifest["images"][3]["specimen_key"] = manifest["images"][0]["specimen_key"]
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DatasetContractError):
        validate_prepared_dataset(root)


def test_validate_prepared_dataset_rejects_empty_partition(tmp_path: Path) -> None:
    root = _prepared(tmp_path / "prepared")
    path, manifest = _manifest(root)
    manifest["images"] = [record for record in manifest["images"] if record["partition"] != "test"]
    manifest["split"]["counts"]["test"] = 0
    manifest["summary"]["canonical_images"] = 6
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DatasetContractError, match=r"partition.*empty"):
        validate_prepared_dataset(root)


def test_validate_prepared_dataset_rejects_partition_missing_target_class(tmp_path: Path) -> None:
    root = _prepared(tmp_path / "prepared")
    path, manifest = _manifest(root)
    test_class_one = next(
        record
        for record in manifest["images"]
        if record["partition"] == "test" and record["source_class"] == "WSSV"
    )
    _rewrite_label(root, test_class_one, "0 0.5 0.5 0.25 0.25\n")
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DatasetContractError, match=r"test.*both target classes"):
        validate_prepared_dataset(root)


def test_validate_prepared_dataset_rejects_partition_without_negative(tmp_path: Path) -> None:
    root = _prepared(tmp_path / "prepared")
    path, manifest = _manifest(root)
    test_negative = next(
        record
        for record in manifest["images"]
        if record["partition"] == "test" and record["source_class"] == "HEALTHY"
    )
    _rewrite_label(root, test_negative, "0 0.5 0.5 0.25 0.25\n")
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DatasetContractError, match=r"test.*negative"):
        validate_prepared_dataset(root)


def test_validate_prepared_dataset_rejects_malformed_label(tmp_path: Path) -> None:
    root = _prepared(tmp_path / "prepared")
    path, manifest = _manifest(root)
    _rewrite_label(root, manifest["images"][0], "0 not-a-number 0.5 0.25 0.25\n")
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DatasetContractError, match="label"):
        validate_prepared_dataset(root)
