# SPDX-License-Identifier: AGPL-3.0-or-later
"""Deterministic, fail-closed ShrimpDiseaseImageBD preparation and evidence overlays."""

from __future__ import annotations

import hashlib
import io
import json
import random
import shutil
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from shrimp_pipeline.archive import ArchiveEntry, ArchiveReader
from shrimp_pipeline.gate import require_mapping_acceptance
from shrimp_pipeline.manifest import parse_specimen_key
from shrimp_pipeline.split import grouped_split

_PARTITIONS = ("train", "validation", "test")
_SOURCE_CLASS_MAP: dict[str, dict[int, int]] = {
    "BG": {0: 0},
    "WSSV": {0: 1},
    "WSSV_BG": {0: 0, 1: 1},
}
_GLOBAL_CLASS_NAMES: dict[int, str] = {0: "dark_gill", 1: "white_spot"}

# Overlay rendering exists so a human can falsify the class mapping by eye. Source
# photographs are around 2048 px, so fixed-size annotations are unreadable: the previous
# bitmap default font drew a 6x8 px digit, roughly 0.3% of the image width. Every visual
# dimension below therefore scales with the image.
#
# Boxes cluster tightly on these specimens, so per-box text is limited to the class digit
# and the class colour carries the meaning. Spelling the name beside every box occludes the
# very tissue under review. Amber and blue are used because they stay separable under the
# common red-green colour vision deficiencies, and the digit remains as a non-colour cue.
_OVERLAY_LABEL_TEXT = (255, 255, 255)
_OVERLAY_CLASS_COLOURS: dict[int, tuple[int, int, int]] = {
    0: (245, 158, 11),
    1: (37, 99, 235),
}
_OVERLAY_LEGEND_BACKGROUND = (17, 24, 39)
_OVERLAY_FONT_SCALE = 0.022
_OVERLAY_MINIMUM_FONT_SIZE = 14
_OVERLAY_OUTLINE_SCALE = 0.0025
_OVERLAY_MINIMUM_OUTLINE_WIDTH = 2


class ConversionError(RuntimeError):
    """Source data violates an invariant required for safe conversion."""


@dataclass(frozen=True, slots=True)
class _OverlayStyle:
    """Image-relative drawing metrics so annotations stay legible at any resolution."""

    font: ImageFont.FreeTypeFont | ImageFont.ImageFont
    outline_width: int
    padding: int

    @classmethod
    def for_image(cls, width: int, height: int) -> _OverlayStyle:
        shorter = min(width, height)
        font_size = max(_OVERLAY_MINIMUM_FONT_SIZE, round(shorter * _OVERLAY_FONT_SCALE))
        outline_width = max(_OVERLAY_MINIMUM_OUTLINE_WIDTH, round(shorter * _OVERLAY_OUTLINE_SCALE))
        return cls(
            font=ImageFont.load_default(size=font_size),
            outline_width=outline_width,
            padding=max(2, round(font_size * 0.25)),
        )


def _draw_class_label(
    draw: ImageDraw.ImageDraw,
    style: _OverlayStyle,
    class_id: int,
    origin: tuple[float, float],
    image_size: tuple[int, int],
) -> None:
    """Draw a compact class-digit chip anchored to a box.

    Only the digit is drawn. These specimens carry many boxes within a small area, so
    spelling the class name at every box hides the tissue the reviewer must judge; the
    name is stated once in the legend instead.
    """
    left, top = origin
    image_width, image_height = image_size
    text = str(class_id)
    bounds = draw.textbbox((0, 0), text, font=style.font)
    chip_width = bounds[2] - bounds[0] + style.padding * 2
    chip_height = bounds[3] - bounds[1] + style.padding * 2
    chip_left = min(max(0.0, left), max(0.0, image_width - chip_width))
    # Prefer sitting the chip above the box; drop it inside when there is no headroom.
    chip_top = top - chip_height
    if chip_top < 0.0:
        chip_top = max(0.0, min(top, image_height - chip_height))
    draw.rectangle(
        (chip_left, chip_top, chip_left + chip_width, chip_top + chip_height),
        fill=_OVERLAY_CLASS_COLOURS[class_id],
    )
    draw.text(
        (chip_left + style.padding - bounds[0], chip_top + style.padding - bounds[1]),
        text,
        font=style.font,
        fill=_OVERLAY_LABEL_TEXT,
    )


def _draw_legend(draw: ImageDraw.ImageDraw, style: _OverlayStyle, class_ids: Iterable[int]) -> None:
    """State each present class once, in a corner panel, away from the specimen."""
    entries = [
        (class_id, f"{class_id} = {_GLOBAL_CLASS_NAMES[class_id]}") for class_id in class_ids
    ]
    if not entries:
        return
    heights = [draw.textbbox((0, 0), text, font=style.font)[3] for _, text in entries]
    widths = [draw.textbbox((0, 0), text, font=style.font)[2] for _, text in entries]
    swatch = max(heights)
    row_height = swatch + style.padding
    panel_width = swatch + style.padding * 3 + max(widths)
    panel_height = row_height * len(entries) + style.padding
    draw.rectangle(
        (style.padding, style.padding, style.padding + panel_width, style.padding + panel_height),
        fill=_OVERLAY_LEGEND_BACKGROUND,
    )
    for row, (class_id, text) in enumerate(entries):
        top = style.padding * 2 + row * row_height
        draw.rectangle(
            (style.padding * 2, top, style.padding * 2 + swatch, top + swatch),
            fill=_OVERLAY_CLASS_COLOURS[class_id],
        )
        draw.text(
            (style.padding * 3 + swatch, top),
            text,
            font=style.font,
            fill=_OVERLAY_LABEL_TEXT,
        )


@dataclass(frozen=True, slots=True)
class ConversionResult:
    output: Path
    manifest: Path
    canonical_images: int


@dataclass(frozen=True, slots=True)
class _PlannedImage:
    entry: ArchiveEntry
    sha256: str
    source_class: str
    specimen_key: tuple[str, int]
    label_text: str
    duplicate_paths: tuple[str, ...]
    partition: str
    output_name: str


def _is_augmentation(entry: ArchiveEntry) -> bool:
    lowered = entry.path.casefold()
    return any(token in lowered for token in ("/augment", "-augmented", "-augmentation", "_aug"))


def _parse_yolo(text: str, source_class: str) -> list[tuple[int, float, float, float, float]]:
    remap = _SOURCE_CLASS_MAP.get(source_class)
    if remap is None:
        if source_class == "HEALTHY" and not text.strip():
            return []
        raise ConversionError(f"unsupported source class {source_class!r}")
    boxes: list[tuple[int, float, float, float, float]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        fields = line.split()
        if len(fields) != 5:
            raise ConversionError(f"malformed YOLO label line {line_number}: expected five fields")
        try:
            source_id = int(fields[0])
            x, y, width, height = (float(value) for value in fields[1:])
        except ValueError as exc:
            raise ConversionError(
                f"malformed YOLO label line {line_number}: non-numeric value"
            ) from exc
        if source_id not in remap:
            raise ConversionError(
                f"source class {source_class} label {source_id} has no accepted global remapping"
            )
        if not all(0.0 <= value <= 1.0 for value in (x, y, width, height)):
            raise ConversionError(f"YOLO label line {line_number} is outside [0, 1]")
        if width <= 0.0 or height <= 0.0:
            raise ConversionError(f"YOLO label line {line_number} has a non-positive box")
        if x - width / 2 < 0.0 or x + width / 2 > 1.0:
            raise ConversionError(
                f"YOLO label line {line_number} extends outside the image horizontally"
            )
        if y - height / 2 < 0.0 or y + height / 2 > 1.0:
            raise ConversionError(
                f"YOLO label line {line_number} extends outside the image vertically"
            )
        boxes.append((remap[source_id], x, y, width, height))
    if source_class != "HEALTHY" and not boxes:
        raise ConversionError(
            f"diseased source {source_class} has no boxes; boxes are never fabricated"
        )
    return boxes


def _render_labels(boxes: list[tuple[int, float, float, float, float]]) -> str:
    return "".join(
        f"{class_id} {x:.12g} {y:.12g} {width:.12g} {height:.12g}\n"
        for class_id, x, y, width, height in boxes
    )


def _partition_map(keys: list[tuple[str, int]], seed: int) -> dict[tuple[str, int], str]:
    split = grouped_split(keys, seed=seed)
    mapping: dict[tuple[str, int], str] = {}
    for partition in _PARTITIONS:
        for key in getattr(split, partition):
            if key in mapping:
                raise ConversionError(f"specimen {key} appears in more than one partition")
            mapping[key] = partition
    if set(mapping) != set(keys):
        raise ConversionError("grouped split did not assign every specimen")
    return mapping


def _plan(reader: ArchiveReader, seed: int) -> tuple[list[_PlannedImage], int]:
    labels_by_stem: dict[str, list[ArchiveEntry]] = {}
    for label in reader.labels():
        labels_by_stem.setdefault(Path(label.name).stem.casefold(), []).append(label)
    digest_groups: dict[str, list[ArchiveEntry]] = {}
    excluded = 0
    for image in reader.images():
        if _is_augmentation(image):
            excluded += 1
            continue
        digest = hashlib.sha256(reader.read(image)).hexdigest()
        digest_groups.setdefault(digest, []).append(image)

    provisional: list[tuple[ArchiveEntry, str, str, tuple[str, int], str, tuple[str, ...]]] = []
    output_names: set[str] = set()
    for digest, entries in sorted(digest_groups.items()):
        parsed_keys = [parse_specimen_key(entry.name) for entry in entries]
        specimen_keys = [key for key in parsed_keys if key is not None]
        if len(specimen_keys) != len(entries) or len(set(specimen_keys)) != 1:
            paths = sorted(entry.path for entry in entries)
            raise ConversionError(f"inconsistent duplicate identity for sha256 {digest}: {paths}")
        specimen_key = specimen_keys[0]
        source_class = specimen_key[0]
        canonical = sorted(
            entries, key=lambda entry: ("annotated" not in entry.source.casefold(), entry.path)
        )[0]
        stem = Path(canonical.name).stem.casefold()
        associated = labels_by_stem.get(stem, [])
        if source_class == "HEALTHY":
            if associated:
                raise ConversionError(
                    f"Healthy image {canonical.path} unexpectedly has supplied boxes"
                )
            label_text = ""
        else:
            if len(associated) != 1:
                raise ConversionError(
                    f"diseased image {canonical.path} requires exactly one source label, "
                    f"found {len(associated)}"
                )
            label_text = _render_labels(_parse_yolo(reader.read_text(associated[0]), source_class))
        output_name = canonical.name
        if output_name.casefold() in output_names:
            raise ConversionError(f"canonical output filename collision: {output_name}")
        output_names.add(output_name.casefold())
        duplicates = tuple(sorted(entry.path for entry in entries if entry.path != canonical.path))
        provisional.append((canonical, digest, source_class, specimen_key, label_text, duplicates))

    partition_keys = sorted({item[3] for item in provisional})
    partitions = _partition_map(partition_keys, seed)
    planned = [
        _PlannedImage(
            entry=entry,
            sha256=digest,
            source_class=source_class,
            specimen_key=key,
            label_text=label_text,
            duplicate_paths=duplicates,
            partition=partitions[key],
            output_name=entry.name,
        )
        for entry, digest, source_class, key, label_text, duplicates in provisional
    ]
    return planned, excluded


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_archive(
    archive: Path,
    acceptance: Path,
    output: Path,
    *,
    seed: int = 20260730,
) -> ConversionResult:
    """Prepare one verified archive atomically; never overwrite an existing dataset."""
    if output.exists():
        raise FileExistsError(f"refusing to replace existing prepared dataset: {output}")
    archive_digest = _file_digest(archive)
    with ArchiveReader(archive) as reader:
        planned, excluded = _plan(reader, seed)
        if not planned:
            raise ConversionError("source archive contains no canonical images")

        # This is deliberately the final operation before any output directory is created.
        gate = require_mapping_acceptance(acceptance)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
        try:
            image_records: list[dict[str, object]] = []
            duplicate_records: list[dict[str, str]] = []
            split_summary = dict.fromkeys(_PARTITIONS, 0)
            for item in planned:
                image_relative = Path("images") / item.partition / item.output_name
                label_relative = (
                    Path("labels") / item.partition / f"{Path(item.output_name).stem}.txt"
                )
                image_path = temporary / image_relative
                label_path = temporary / label_relative
                image_path.parent.mkdir(parents=True, exist_ok=True)
                label_path.parent.mkdir(parents=True, exist_ok=True)
                image_path.write_bytes(reader.read(item.entry))
                label_path.write_text(item.label_text, encoding="utf-8")
                if _file_digest(image_path) != item.sha256:
                    raise ConversionError(f"output hash mismatch after writing {image_relative}")
                split_summary[item.partition] += 1
                specimen = f"{item.specimen_key[0]}:{item.specimen_key[1]}"
                image_records.append(
                    {
                        "source_path": item.entry.path,
                        "source_role": "annotated"
                        if "annotated" in item.entry.source.casefold()
                        else "raw",
                        "source_class": item.source_class,
                        "specimen_key": specimen,
                        "sha256": item.sha256,
                        "partition": item.partition,
                        "image_output": image_relative.as_posix(),
                        "image_output_sha256": _file_digest(image_path),
                        "label_output": label_relative.as_posix(),
                        "label_output_sha256": _file_digest(label_path),
                    }
                )
                duplicate_records.extend(
                    {"path": duplicate, "duplicate_of": item.entry.path, "sha256": item.sha256}
                    for duplicate in item.duplicate_paths
                )
            if sum(split_summary.values()) != len(planned):
                raise ConversionError("split summary does not equal canonical image count")
            manifest = {
                "schema_version": "1.0.0",
                "input": {"archive": str(archive), "sha256": archive_digest},
                "mapping_acceptance": {
                    "status": gate.mapping_status,
                    "reviewer": gate.reviewer,
                    "reviewed_on": gate.reviewed_on.isoformat(),
                    "evidence_report_sha256": gate.evidence_report_sha256,
                },
                "classes": {
                    str(class_id): name for class_id, name in sorted(_GLOBAL_CLASS_NAMES.items())
                },
                "split": {"seed": seed, "counts": split_summary},
                "summary": {
                    "canonical_images": len(planned),
                    "duplicate_entries": len(duplicate_records),
                    "excluded_augmentations": excluded,
                },
                "images": image_records,
                "duplicates": duplicate_records,
            }
            manifest_path = temporary / "manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            temporary.replace(output)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    return ConversionResult(
        output=output, manifest=output / "manifest.json", canonical_images=len(planned)
    )


def generate_evidence_report(
    archive: Path,
    output: Path,
    *,
    minimum_overlays: int = 60,
    seed: int = 20260730,
) -> Path:
    """Render supplied source boxes for review without asserting human acceptance."""
    if minimum_overlays < 1:
        raise ValueError("minimum_overlays must be positive")
    if output.exists():
        raise FileExistsError(f"refusing to replace existing evidence directory: {output}")
    with ArchiveReader(archive) as reader:
        images_by_stem = {
            Path(entry.name).stem.casefold(): entry
            for entry in reader.images()
            if not _is_augmentation(entry)
        }
        candidates: list[tuple[ArchiveEntry, ArchiveEntry, str]] = []
        for label in reader.labels():
            candidate_image = images_by_stem.get(Path(label.name).stem.casefold())
            key = parse_specimen_key(label.name)
            if candidate_image is not None and key is not None:
                _parse_yolo(reader.read_text(label), key[0])
                candidates.append((candidate_image, label, key[0]))
        if len(candidates) < minimum_overlays:
            raise ConversionError(
                f"only {len(candidates)} labelled views exist; at least "
                f"{minimum_overlays} are required"
            )
        rng = random.Random(seed)
        by_class: dict[str, list[tuple[ArchiveEntry, ArchiveEntry, str]]] = {}
        for candidate in candidates:
            by_class.setdefault(candidate[2], []).append(candidate)
        for values in by_class.values():
            values.sort(key=lambda item: item[0].path)
            rng.shuffle(values)
        selected: list[tuple[ArchiveEntry, ArchiveEntry, str]] = []
        while len(selected) < minimum_overlays:
            progressed = False
            for source_class in sorted(by_class):
                if by_class[source_class] and len(selected) < minimum_overlays:
                    selected.append(by_class[source_class].pop())
                    progressed = True
            if not progressed:
                break

        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
        try:
            overlays: list[dict[str, object]] = []
            for index, (image_entry, label_entry, source_class) in enumerate(selected, start=1):
                rendered_image = Image.open(io.BytesIO(reader.read(image_entry))).convert("RGB")
                draw = ImageDraw.Draw(rendered_image)
                style = _OverlayStyle.for_image(rendered_image.width, rendered_image.height)
                source_boxes = _parse_yolo(reader.read_text(label_entry), source_class)
                for class_id, x, y, width, height in source_boxes:
                    left = (x - width / 2) * rendered_image.width
                    top = (y - height / 2) * rendered_image.height
                    right = (x + width / 2) * rendered_image.width
                    bottom = (y + height / 2) * rendered_image.height
                    draw.rectangle(
                        (left, top, right, bottom),
                        outline=_OVERLAY_CLASS_COLOURS[class_id],
                        width=style.outline_width,
                    )
                    _draw_class_label(
                        draw,
                        style,
                        class_id,
                        (left, top),
                        (rendered_image.width, rendered_image.height),
                    )
                _draw_legend(draw, style, sorted({box[0] for box in source_boxes}))
                relative = Path("overlays") / f"overlay-{index:03d}.jpg"
                destination = temporary / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                rendered_image.save(destination, format="JPEG", quality=92)
                overlays.append(
                    {
                        "path": relative.as_posix(),
                        "sha256": _file_digest(destination),
                        "source_image": image_entry.path,
                        "source_label": label_entry.path,
                        "source_class": source_class,
                        "source_box_count": len(source_boxes),
                    }
                )
            report = {
                "schema_version": "1.0.0",
                "archive_sha256": _file_digest(archive),
                "mapping_status": "PROVISIONAL_UNCONFIRMED",
                "human_acceptance": False,
                "note": (
                    "These overlays reproduce supplied boxes; they do not constitute "
                    "human acceptance."
                ),
                "overlays": overlays,
            }
            report_path = temporary / "evidence_report.json"
            report_path.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            temporary.replace(output)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    return output / "evidence_report.json"
