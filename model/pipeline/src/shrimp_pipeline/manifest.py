# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from shrimp_pipeline.archive import ArchiveReader

_PATTERN = re.compile(
    r"^(Healthy|BG|WSSV|WSSV_BG)-(\d+)-img-(\d+)(?:-aug(?:mented)?[^.]*)?\.(?:jpg|jpeg|png|txt)$",
    re.IGNORECASE,
)


def parse_specimen_key(name: str) -> tuple[str, int] | None:
    match = _PATTERN.match(Path(name).name)
    if not match:
        return None
    return match.group(1).upper(), int(match.group(2))


@dataclass(frozen=True, slots=True)
class CanonicalHashGroup:
    sha256: str
    canonical_path: str
    duplicate_paths: tuple[str, ...]
    specimen_keys: tuple[str, ...]
    consistent: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ArchiveInventory:
    archive: str
    archive_sha256: str
    image_files: int
    unique_image_hashes: int
    duplicate_image_entries: int
    duplicate_groups: int
    inconsistent_duplicate_groups: int
    label_files: int
    specimen_keys: int
    malformed_label_lines: int
    metadata_entries_skipped: int
    canonical_hash_groups: tuple[CanonicalHashGroup, ...]

    def to_dict(self) -> dict[str, object]:
        document = asdict(self)
        document["canonical_hash_groups"] = [
            group.to_dict() for group in self.canonical_hash_groups
        ]
        return document


def _label_is_valid(text: str) -> bool:
    for line in text.splitlines():
        fields = line.split()
        if len(fields) != 5:
            return False
        try:
            class_id = int(fields[0])
            values = [float(value) for value in fields[1:]]
        except ValueError:
            return False
        if class_id < 0 or any(not 0.0 <= value <= 1.0 for value in values):
            return False
    return True


def inventory_archive(path: Path) -> ArchiveInventory:
    """Hash every image and expose exact duplicate/canonical relationships."""
    digest_entries: dict[str, list[tuple[str, tuple[str, int] | None]]] = {}
    keys: set[tuple[str, int]] = set()
    malformed = 0
    with ArchiveReader(path) as reader:
        images = reader.images()
        labels = reader.labels()
        for entry in images:
            key = parse_specimen_key(entry.name)
            if key is not None:
                keys.add(key)
            digest = hashlib.sha256(reader.read(entry)).hexdigest()
            digest_entries.setdefault(digest, []).append((entry.path, key))
        for entry in labels:
            key = parse_specimen_key(entry.name)
            if key is not None:
                keys.add(key)
            if not _label_is_valid(reader.read_text(entry)):
                malformed += 1

    groups: list[CanonicalHashGroup] = []
    for digest, members in sorted(digest_entries.items()):
        ordered = sorted(path_name for path_name, _ in members)
        key_values = sorted({f"{key[0]}:{key[1]}" for _, key in members if key is not None})
        consistent = len(key_values) == 1 and all(key is not None for _, key in members)
        groups.append(
            CanonicalHashGroup(
                sha256=digest,
                canonical_path=ordered[0],
                duplicate_paths=tuple(ordered[1:]),
                specimen_keys=tuple(key_values),
                consistent=consistent,
            )
        )

    duplicate_groups = sum(len(group.duplicate_paths) > 0 for group in groups)
    duplicate_entries = sum(len(group.duplicate_paths) for group in groups)
    inconsistent = sum(not group.consistent for group in groups if group.duplicate_paths)
    return ArchiveInventory(
        archive=str(path),
        archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        image_files=len(images),
        unique_image_hashes=len(groups),
        duplicate_image_entries=duplicate_entries,
        duplicate_groups=duplicate_groups,
        inconsistent_duplicate_groups=inconsistent,
        label_files=len(labels),
        specimen_keys=len(keys),
        malformed_label_lines=malformed,
        metadata_entries_skipped=0,
        canonical_hash_groups=tuple(groups),
    )
