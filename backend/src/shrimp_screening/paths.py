"""Locating `data/`, the directory holding every file the service loads at runtime.

Its contents are deployment data, not Python package data: a farm operator can
update the guidance corpus or a threshold without reinstalling the service. They
live at the repository root, so the root has to be discoverable.

Discovery is explicit and fails loudly. It never silently falls back to the current
working directory, because that would make policy hashes depend on where uvicorn was
started from.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

#: Marker files that only ever exist together at the root of this repository. All
#: three are loaded during startup, so a root missing any of them is a root the
#: service could not have booted from anyway.
_ROOT_MARKERS: tuple[str, ...] = (
    "data/quality_policy_v1.json",
    "data/guidance_v1.json",
    "data/model_registry.json",
)

#: Override for containerised or packaged deployments.
ROOT_ENV_VAR = "SHRIMP_REPO_ROOT"


class DataRootNotFoundError(RuntimeError):
    """Raised when the policy/guidance/model data root cannot be located."""


def _has_markers(candidate: Path) -> bool:
    return all((candidate / marker).is_file() for marker in _ROOT_MARKERS)


@lru_cache(maxsize=1)
def repository_root() -> Path:
    """Return the directory holding `data/`."""
    override = os.environ.get(ROOT_ENV_VAR)
    if override:
        candidate = Path(override).expanduser().resolve()
        if not _has_markers(candidate):
            raise DataRootNotFoundError(
                f"{ROOT_ENV_VAR}={override!r} does not contain {', '.join(_ROOT_MARKERS)}"
            )
        return candidate

    here = Path(__file__).resolve()
    for parent in here.parents:
        if _has_markers(parent):
            return parent
    raise DataRootNotFoundError(
        "Could not locate the data root above "
        f"{here}. Set {ROOT_ENV_VAR} to the directory containing "
        f"{', '.join(_ROOT_MARKERS)}."
    )


def data_dir() -> Path:
    """The one directory every runtime data file is read from."""
    return repository_root() / "data"
