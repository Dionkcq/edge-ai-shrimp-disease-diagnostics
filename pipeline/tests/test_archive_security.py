# SPDX-License-Identifier: AGPL-3.0-or-later
"""Resource bounds for read-only nested archive discovery."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

from shrimp_pipeline.archive import ArchiveReader


def _zip_member(name: str, payload: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, payload)
    return buffer.getvalue()


def test_two_nested_zip_levels_are_supported(tmp_path: Path) -> None:
    second = _zip_member("shrimp.jpg", b"image")
    first = _zip_member("second.zip", second)
    path = tmp_path / "outer.zip"
    path.write_bytes(_zip_member("first.zip", first))

    with ArchiveReader(path) as reader:
        assert [entry.name for entry in reader.images()] == ["shrimp.jpg"]


def test_a_third_nested_zip_level_is_not_opened(tmp_path: Path) -> None:
    third = _zip_member("hidden.jpg", b"image")
    second = _zip_member("third.zip", third)
    first = _zip_member("second.zip", second)
    path = tmp_path / "outer.zip"
    path.write_bytes(_zip_member("first.zip", first))

    with ArchiveReader(path) as reader:
        assert reader.entries == []
