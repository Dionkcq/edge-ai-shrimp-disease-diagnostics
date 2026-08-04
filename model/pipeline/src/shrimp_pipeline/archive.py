# SPDX-License-Identifier: AGPL-3.0-or-later
"""Read-only access to the source archives, including their nested ZIPs.

The real archives are not flat. `ShrimpDiseaseImageBD_v3.zip` contains two inner
ZIPs (`Annotated Diseased Shrimp Images.zip`, `Raw Images.zip`) alongside a
`Readme.docx` and an `.xlsx` summary, so anything that reads the dataset has to
recurse one level.

They are also full of macOS resource forks: in the annotated inner archive, 1,494
of 2,994 entries live under `__MACOSX/` and shadow every real file with a `._`
sibling. Counting those as images would roughly double every number in the audit,
and feeding one to a decoder produces an error rather than a picture. Filtering is
therefore not cosmetic, and `is_metadata_path` has its own regression test.

Nothing in this module writes to the archives or extracts them to disk.
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Iterator
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Self

#: A nested ZIP larger than this is not recursed into. The real inner archives are
#: ~200 MB and ~290 MB; the bound exists so a malicious or corrupt archive cannot
#: make us buffer something unbounded in memory.
MAX_NESTED_ZIP_BYTES = 1_500_000_000
MAX_NESTED_ZIP_DEPTH = 2

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
LABEL_SUFFIX = ".txt"


def is_metadata_path(name: str) -> bool:
    """True for macOS resource forks, AppleDouble siblings and OS junk files."""
    normalized = name.replace("\\", "/")
    parts = normalized.split("/")
    base = parts[-1]
    return (
        "__MACOSX" in parts
        or "__MACOSX" in normalized
        or base.startswith("._")
        or base in {".DS_Store", "Thumbs.db", "desktop.ini", ""}
    )


@dataclass(frozen=True, slots=True)
class ArchiveEntry:
    """One file inside an archive, identified by its full logical path.

    ``source`` names the containing ZIP (the inner one, when nested), so a report
    can attribute a finding to the archive it came from.
    """

    path: str
    size: int
    source: str

    @property
    def name(self) -> str:
        return self.path.replace("\\", "/").rsplit("/", 1)[-1]

    @property
    def suffix(self) -> str:
        base = self.name
        return base[base.rfind(".") :].lower() if "." in base else ""

    @property
    def is_image(self) -> bool:
        return self.suffix in IMAGE_SUFFIXES

    @property
    def is_label(self) -> bool:
        return self.suffix == LABEL_SUFFIX


class ArchiveReader:
    """Opens an archive and its nested ZIPs for read-only access.

    Used as a context manager. Nested archives are buffered in memory once and kept
    open, because the audit reads thousands of small members and re-opening per read
    would dominate the runtime.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._stack = ExitStack()
        #: Logical entry path -> (owning ZipFile, member name).
        self._members: dict[str, tuple[zipfile.ZipFile, str]] = {}
        self._entries: list[ArchiveEntry] = []

    def __enter__(self) -> Self:
        try:
            outer = self._stack.enter_context(zipfile.ZipFile(self._path))
        except (OSError, zipfile.BadZipFile) as exc:
            self._stack.close()
            raise ArchiveError(f"could not open archive {self._path}: {exc}") from exc
        self._index(outer, label=self._path.name, prefix="", depth=0)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._stack.close()

    def _index(self, archive: zipfile.ZipFile, *, label: str, prefix: str, depth: int) -> None:
        for info in archive.infolist():
            if info.is_dir() or is_metadata_path(info.filename):
                continue
            logical = f"{prefix}{info.filename}"
            if info.filename.lower().endswith(".zip"):
                if depth >= MAX_NESTED_ZIP_DEPTH or info.file_size > MAX_NESTED_ZIP_BYTES:
                    continue
                try:
                    nested = self._stack.enter_context(
                        zipfile.ZipFile(io.BytesIO(archive.read(info)))
                    )
                except (OSError, zipfile.BadZipFile):
                    continue
                self._index(
                    nested,
                    label=info.filename.rsplit("/", 1)[-1],
                    prefix=f"{logical}!/",
                    depth=depth + 1,
                )
                continue
            self._members[logical] = (archive, info.filename)
            self._entries.append(ArchiveEntry(path=logical, size=info.file_size, source=label))

    @property
    def entries(self) -> list[ArchiveEntry]:
        """Every non-metadata file, in archive order."""
        return list(self._entries)

    def images(self) -> list[ArchiveEntry]:
        return [entry for entry in self._entries if entry.is_image]

    def labels(self) -> list[ArchiveEntry]:
        return [entry for entry in self._entries if entry.is_label]

    def read(self, entry: ArchiveEntry) -> bytes:
        """Read one member's bytes. Raises :class:`ArchiveError` if it is unknown."""
        try:
            archive, member = self._members[entry.path]
        except KeyError as exc:
            raise ArchiveError(f"no such entry: {entry.path}") from exc
        return archive.read(member)

    def read_text(self, entry: ArchiveEntry) -> str:
        return self.read(entry).decode("utf-8", errors="replace")


class ArchiveError(RuntimeError):
    """An archive could not be opened or a member could not be read."""


def iter_entries(paths: list[Path]) -> Iterator[tuple[Path, ArchiveEntry]]:
    """Yield every non-metadata entry across several archives."""
    for path in paths:
        with ArchiveReader(path) as reader:
            for entry in reader.entries:
                yield path, entry
