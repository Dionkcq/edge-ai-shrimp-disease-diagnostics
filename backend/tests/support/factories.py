"""Test helpers that are called directly rather than injected as fixtures.

They live here, not in `conftest.py`, so that a test can import them by their real
module path. Importing from a `conftest` is what forces the parent-relative
`from ..conftest import ...` that the import policy bans, and it couples every
caller to pytest's collection layout.

Everything here builds its input in memory: no test reads a real shrimp
photograph, a trained model or anything else absent from a clean clone.
"""

from __future__ import annotations

import io

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from shrimp_screening.main import create_app
from shrimp_screening.settings import ProviderName, Settings


def build_app(provider: ProviderName = "unavailable", **overrides: object) -> FastAPI:
    """Build an app on an explicit provider, bypassing ambient environment variables.

    Settings are constructed directly rather than through `SHRIMP_*` so a test states
    the configuration it is proving. `env="test"` keeps the audience-environment
    validator (which forbids synthetic output under demo/production) out of the way;
    the tests that exercise that validator set `env` themselves.
    """
    return create_app(Settings(env="test", provider=provider, **overrides))  # type: ignore[arg-type]


def client_for(provider: ProviderName = "unavailable", **overrides: object) -> TestClient:
    """A `TestClient` for `provider`, to be used as a context manager."""
    return TestClient(build_app(provider, **overrides))


def make_image_bytes(
    width: int = 1024,
    height: int = 768,
    *,
    image_format: str = "JPEG",
    mode: str = "RGB",
    seed: int = 7,
    **save_kwargs: object,
) -> bytes:
    """Return an in-memory image with enough texture to clear the quality gate."""
    rng = np.random.default_rng(seed)
    yy, xx = np.indices((height, width))
    base = ((xx * 5 + yy * 3) % 150 + 55).astype(np.float32)
    noise = rng.integers(0, 40, size=(height, width)).astype(np.float32)
    plane = np.clip(base + noise, 0, 255).astype(np.uint8)
    array = np.repeat(plane[:, :, None], 3, axis=2)
    image = Image.fromarray(array, mode="RGB").convert(mode)
    buffer = io.BytesIO()
    image.save(buffer, format=image_format, **save_kwargs)  # type: ignore[arg-type]
    return buffer.getvalue()
