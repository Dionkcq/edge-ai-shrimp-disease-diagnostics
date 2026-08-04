"""Shared fixtures for the backend suite.

The helpers these fixtures are built from live in `tests.support.factories` so that
a test needing one imports it by module path instead of reaching into a conftest.

Every fixture here builds its input in memory. No test reads a real shrimp
photograph, a trained model or anything else that is absent from a clean clone.
"""

from __future__ import annotations

import io
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
from PIL import Image

from shrimp_screening.guidance.store import load_guidance
from shrimp_screening.paths import repository_root
from tests.support.factories import make_image_bytes

#: The repository root as seen from this file: backend/tests/conftest.py -> ../..
_REPO_ROOT = Path(__file__).resolve().parents[2]

_EMPTY_REGISTRY = '{\n  "schema_version": "1.0.0",\n  "models": []\n}\n'


@pytest.fixture(autouse=True)
def pinned_repository_root(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Pin policy/guidance/model discovery to this checkout.

    Without this, `repository_root()` walks up from the installed package. That
    happens to be correct for an editable install and wrong for every other
    packaging mode, which would make the suite pass for the wrong reason.
    """
    monkeypatch.setenv("SHRIMP_REPO_ROOT", str(_REPO_ROOT))
    repository_root.cache_clear()
    yield
    repository_root.cache_clear()


@pytest.fixture(autouse=True)
def no_ambient_provider_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    """A developer's exported `SHRIMP_*` must never change what the suite proves."""
    for name in ("SHRIMP_PROVIDER", "SHRIMP_ENV", "SHRIMP_ONNX_MODEL_PATH"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def cleared_document_caches() -> Iterator[None]:
    """Drop the `lru_cache` on the guidance corpus.

    It is `lru_cache(maxsize=1)` keyed on the path argument, so a test that loads a
    `tmp_path` document evicts the default entry and the *next* test to ask for the
    real one would be handed whatever the previous test wrote.
    """
    load_guidance.cache_clear()
    yield
    load_guidance.cache_clear()


@pytest.fixture
def relocated_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A writable repository root that borrows this checkout's reviewed data.

    `data/` is copied rather than invented, so a test cannot accidentally prove
    something against made-up thresholds or made-up guidance. It is a copy rather
    than a symlink because the model registry is overwritten below: registering an
    artifact is the one thing a test legitimately needs to vary, and it must never
    reach the committed registry.

    `data/raw/` is skipped -- it holds the multi-hundred-megabyte source archives on
    a developer machine and nothing in the runtime reads it.
    """
    root = tmp_path / "relocated-root"
    root.mkdir()
    shutil.copytree(
        _REPO_ROOT / "data",
        root / "data",
        ignore=shutil.ignore_patterns("raw", "processed"),
    )
    (root / "data" / "model_registry.json").write_text(_EMPTY_REGISTRY, encoding="utf-8")

    monkeypatch.setenv("SHRIMP_REPO_ROOT", str(root))
    repository_root.cache_clear()
    yield root
    repository_root.cache_clear()


@pytest.fixture
def jpeg_bytes() -> bytes:
    """A well-formed JPEG that passes intake and the quality gate."""
    return make_image_bytes(image_format="JPEG", quality=92)


@pytest.fixture
def png_bytes() -> bytes:
    return make_image_bytes(image_format="PNG")


@pytest.fixture
def blurry_jpeg_bytes() -> bytes:
    """A flat JPEG: large enough, but with no high-frequency detail at all."""
    image = Image.new("RGB", (1024, 768), (128, 130, 126))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    return buffer.getvalue()
