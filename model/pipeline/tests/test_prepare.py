from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from shrimp_pipeline.convert import (
    _GLOBAL_CLASS_NAMES,
    _OVERLAY_CLASS_COLOURS,
    ConversionError,
    _OverlayStyle,
    generate_evidence_report,
    prepare_archive,
)
from shrimp_pipeline.gate import MappingGateError, require_mapping_acceptance
from shrimp_pipeline.manifest import inventory_archive


def _jpeg(colour: tuple[int, int, int], size: tuple[int, int] = (32, 24)) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", size, colour).save(out, format="JPEG")
    return out.getvalue()


def _source_archive(path: Path, *, inconsistent_duplicate: bool = False) -> Path:
    bg = _jpeg((40, 80, 120))
    wssv = _jpeg((120, 80, 40))
    combined = _jpeg((80, 120, 40))
    healthy = _jpeg((90, 90, 90))
    with zipfile.ZipFile(path, "w") as archive:
        files = {
            "Annotated/1. BG/images/BG-1-img-1.jpg": bg,
            "Annotated/1. BG/labels/BG-1-img-1.txt": b"0 0.5 0.5 0.25 0.25\n",
            "Raw/2. BG/images/BG-1-img-1.jpg": bg,
            "Annotated/2. WSSV/images/WSSV-2-img-1.jpg": wssv,
            "Annotated/2. WSSV/labels/WSSV-2-img-1.txt": b"0 0.4 0.4 0.2 0.2\n",
            "Raw/3. WSSV/images/WSSV-2-img-1.jpg": wssv,
            "Annotated/4. WSSV_BG/images/WSSV_BG-3-img-1.jpg": combined,
            "Annotated/4. WSSV_BG/labels/WSSV_BG-3-img-1.txt": (
                b"0 0.3 0.3 0.1 0.1\n1 0.7 0.7 0.1 0.1\n"
            ),
            "Raw/4. WSSV_BG/images/WSSV_BG-3-img-1.jpg": combined,
            "Raw/1. Healthy/images/Healthy-4-img-1.jpg": healthy,
            "Raw/1. Healthy/images/Healthy-4-img-1-augmented.jpg": _jpeg((1, 2, 3)),
        }
        if inconsistent_duplicate:
            files["Raw/1. Healthy/images/Healthy-99-img-1.jpg"] = bg
        for name, payload in files.items():
            archive.writestr(name, payload)
    return path


def _acceptance(path: Path, evidence: Path, **overrides: object) -> Path:
    record: dict[str, object] = {
        "schema_version": "1.0.0",
        "mapping_status": "PROVISIONAL_UNCONFIRMED",
        "accepted_mapping": {"0": "dark_gill", "1": "white_spot"},
        "provisional_mapping_acknowledged": True,
        "author_confirmed": False,
        "annotation_convention_acknowledged": True,
        "acknowledgement": "I reviewed the provisional class order and annotation drift.",
        "evidence_report": str(evidence),
        "evidence_report_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
        "overlay_sheets_reviewed": 60,
        "reviewer": "Independent aquatic imaging reviewer",
        "reviewed_on": "2026-07-30",
    }
    record.update(overrides)
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def test_gate_rejects_placeholder_unknown_invalid_and_unverified_records(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.json"
    evidence.write_text("{}\n", encoding="utf-8")
    bad_values = [
        {"reviewer": "REPLACE_WITH_HUMAN_REVIEWER"},
        {"reviewed_on": "YYYY-MM-DD"},
        {"author_confirmed": 0},
        {"overlay_sheets_reviewed": 59},
        {"acknowledgement": "  "},
        {"mapping_status": "CONFIRMED"},
        {"unexpected": True},
        {"evidence_report_sha256": "0" * 64},
    ]
    for index, override in enumerate(bad_values):
        path = _acceptance(tmp_path / f"bad-{index}.json", evidence, **override)
        with pytest.raises(MappingGateError):
            require_mapping_acceptance(path)


def test_gate_accepts_only_a_strict_record_with_matching_evidence(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.json"
    evidence.write_text('{"overlays": 60}\n', encoding="utf-8")
    accepted = require_mapping_acceptance(_acceptance(tmp_path / "acceptance.json", evidence))
    assert accepted.reviewer == "Independent aquatic imaging reviewer"
    assert accepted.author_confirmed is False


def test_inventory_records_canonical_hash_dedup_and_inconsistency(tmp_path: Path) -> None:
    report = inventory_archive(_source_archive(tmp_path / "source.zip"))
    assert report.image_files == 8
    assert report.unique_image_hashes == 5
    assert report.duplicate_image_entries == 3
    assert report.duplicate_groups == 3
    assert report.inconsistent_duplicate_groups == 0

    inconsistent = inventory_archive(
        _source_archive(tmp_path / "inconsistent.zip", inconsistent_duplicate=True)
    )
    assert inconsistent.inconsistent_duplicate_groups == 1


def test_prepare_is_fail_closed_deduplicated_remapped_and_atomic(tmp_path: Path) -> None:
    archive = _source_archive(tmp_path / "source.zip")
    evidence = tmp_path / "evidence.json"
    evidence.write_text('{"overlays": 60}\n', encoding="utf-8")
    acceptance = _acceptance(tmp_path / "acceptance.json", evidence)
    output = tmp_path / "prepared"

    result = prepare_archive(archive, acceptance, output, seed=7)

    assert result.canonical_images == 4
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["input"]["sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert manifest["summary"]["canonical_images"] == 4
    assert manifest["summary"]["excluded_augmentations"] == 1
    assert len(manifest["duplicates"]) == 3
    records = manifest["images"]
    assert len({item["sha256"] for item in records}) == 4
    assert {item["partition"] for item in records} <= {"train", "validation", "test"}
    by_class = {item["source_class"]: item for item in records}
    assert (output / by_class["HEALTHY"]["label_output"]).read_text(encoding="utf-8") == ""
    assert (output / by_class["BG"]["label_output"]).read_text().startswith("0 ")
    assert (output / by_class["WSSV"]["label_output"]).read_text().startswith("1 ")
    assert {
        line.split()[0]
        for line in (output / by_class["WSSV_BG"]["label_output"]).read_text().splitlines()
    } == {"0", "1"}
    for record in records:
        assert (
            hashlib.sha256((output / record["image_output"]).read_bytes()).hexdigest()
            == record["sha256"]
        )
    specimen_partitions: dict[str, set[str]] = {}
    for record in records:
        specimen_partitions.setdefault(record["specimen_key"], set()).add(record["partition"])
    assert all(len(partitions) == 1 for partitions in specimen_partitions.values())

    with pytest.raises(FileExistsError):
        prepare_archive(archive, acceptance, output, seed=7)
    assert (output / "manifest.json").is_file()


def test_prepare_refuses_before_output_on_bad_labels_gate_or_duplicate_identity(
    tmp_path: Path,
) -> None:
    archive = _source_archive(tmp_path / "source.zip")
    evidence = tmp_path / "evidence.json"
    evidence.write_text("{}\n", encoding="utf-8")
    acceptance = _acceptance(tmp_path / "acceptance.json", evidence, overlay_sheets_reviewed=0)
    output = tmp_path / "prepared"
    with pytest.raises(MappingGateError):
        prepare_archive(archive, acceptance, output)
    assert not output.exists()

    inconsistent = _source_archive(tmp_path / "inconsistent.zip", inconsistent_duplicate=True)
    good = _acceptance(tmp_path / "good.json", evidence)
    with pytest.raises(ConversionError, match="inconsistent duplicate"):
        prepare_archive(inconsistent, good, output)
    assert not output.exists()


def test_evidence_generation_draws_source_boxes_without_accepting_mapping(tmp_path: Path) -> None:
    archive = _source_archive(tmp_path / "source.zip")
    destination = tmp_path / "evidence"
    report_path = generate_evidence_report(archive, destination, minimum_overlays=3, seed=3)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["human_acceptance"] is False
    assert report["mapping_status"] == "PROVISIONAL_UNCONFIRMED"
    assert len(report["overlays"]) == 3
    for overlay in report["overlays"]:
        path = destination / overlay["path"]
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == overlay["sha256"]


def test_overlay_style_scales_annotations_with_image_size() -> None:
    # A fixed-size annotation is unreadable on a full-resolution source photograph, which
    # would make the human mapping gate impossible to satisfy honestly.
    large = _OverlayStyle.for_image(2048, 2048)
    small = _OverlayStyle.for_image(320, 240)
    assert large.font.size > small.font.size
    assert large.outline_width > small.outline_width
    # Text must occupy a visually meaningful share of the shorter edge.
    assert large.font.size >= 2048 * 0.02
    # Tiny fixtures must still render rather than collapse to a zero-sized font.
    assert small.font.size >= 14
    assert small.outline_width >= 2


def _pixels_near(image: Image.Image, colour: tuple[int, int, int], tolerance: int = 40) -> int:
    raw = image.tobytes()
    red, green, blue = colour
    limit = tolerance**2
    return sum(
        1
        for offset in range(0, len(raw), 3)
        if (raw[offset] - red) ** 2 + (raw[offset + 1] - green) ** 2 + (raw[offset + 2] - blue) ** 2
        <= limit
    )


def test_evidence_overlay_labels_are_legible_and_class_distinct(tmp_path: Path) -> None:
    combined = _jpeg((80, 120, 40), size=(640, 640))
    archive = tmp_path / "large.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("Annotated/4. WSSV_BG/images/WSSV_BG-3-img-1.jpg", combined)
        zipped.writestr(
            "Annotated/4. WSSV_BG/labels/WSSV_BG-3-img-1.txt",
            b"0 0.3 0.3 0.1 0.1\n1 0.7 0.7 0.1 0.1\n",
        )
    destination = tmp_path / "evidence"
    report_path = generate_evidence_report(archive, destination, minimum_overlays=1, seed=1)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    overlay = Image.open(destination / report["overlays"][0]["path"]).convert("RGB")

    # Both global classes appear in this image, and each must be drawn in its own visibly
    # distinct colour so an inversion is obvious to the reviewer.
    for class_id in _GLOBAL_CLASS_NAMES:
        painted = _pixels_near(overlay, _OVERLAY_CLASS_COLOURS[class_id])
        assert painted > 200, f"class {class_id} is not visibly marked: {painted}px"

    # Per-box text stays compact so dense annotations do not hide the tissue under review.
    style = _OverlayStyle.for_image(640, 640)
    scratch = ImageDraw.Draw(Image.new("RGB", (640, 640)))
    digit_width = scratch.textbbox((0, 0), "0", font=style.font)[2]
    named_width = scratch.textbbox((0, 0), "0 dark_gill", font=style.font)[2]
    assert digit_width * 3 < named_width
