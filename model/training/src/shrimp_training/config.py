# SPDX-License-Identifier: AGPL-3.0-or-later
"""Strict generic training profiles; personal hardware details never belong here."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_FIELDS = {
    "schema_version",
    "profile_name",
    "image_size",
    "epochs",
    "patience",
    "batch_fallback",
    "workers",
    "device",
    "amp",
    "deterministic",
    "seed",
    "cache",
    "learning_rate",
    "weight_decay",
}


class ProfileError(ValueError):
    """A training profile is malformed or contains unreviewed fields."""


@dataclass(frozen=True, slots=True)
class TrainingProfile:
    profile_name: str
    image_size: int
    epochs: int
    patience: int
    batch_fallback: tuple[int, ...]
    workers: int
    device: int
    amp: bool
    deterministic: bool
    seed: int
    cache: bool
    learning_rate: float
    weight_decay: float


def _integer(document: dict[str, Any], field: str, *, minimum: int) -> int:
    value = document.get(field)
    if type(value) is not int or value < minimum:
        raise ProfileError(f"{field} must be an integer >= {minimum}")
    return value


def _boolean(document: dict[str, Any], field: str) -> bool:
    value = document.get(field)
    if type(value) is not bool:
        raise ProfileError(f"{field} must be a boolean")
    return value


def _float(document: dict[str, Any], field: str, *, minimum: float, inclusive: bool) -> float:
    value = document.get(field)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ProfileError(f"{field} must be a number")
    invalid = value < minimum if inclusive else value <= minimum
    if invalid:
        description = "a non-negative number" if inclusive else "a positive number"
        raise ProfileError(f"{field} must be {description}")
    return float(value)


def load_profile(path: Path) -> TrainingProfile:
    """Load one exact-schema training profile."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ProfileError(f"training profile cannot be read: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ProfileError("training profile is not valid JSON") from exc
    if not isinstance(raw, dict):
        raise ProfileError("training profile must be a JSON object")
    unknown = set(raw) - _FIELDS
    missing = _FIELDS - set(raw)
    if unknown or missing:
        detail = (
            f"unknown fields {sorted(unknown)}" if unknown else f"missing fields {sorted(missing)}"
        )
        raise ProfileError(f"training profile has {detail}")
    if raw["schema_version"] != "1.0.0":
        raise ProfileError("schema_version must be 1.0.0")
    name = raw["profile_name"]
    if not isinstance(name, str) or not name.strip():
        raise ProfileError("profile_name must be a non-empty string")
    batches = raw["batch_fallback"]
    if (
        not isinstance(batches, list)
        or not batches
        or any(type(value) is not int or value < 1 for value in batches)
        or batches != sorted(set(batches), reverse=True)
    ):
        raise ProfileError("batch_fallback must be unique positive integers in descending order")
    image_size = _integer(raw, "image_size", minimum=32)
    if image_size % 32:
        raise ProfileError("image_size must be divisible by 32")
    return TrainingProfile(
        profile_name=name.strip(),
        image_size=image_size,
        epochs=_integer(raw, "epochs", minimum=1),
        patience=_integer(raw, "patience", minimum=0),
        batch_fallback=tuple(batches),
        workers=_integer(raw, "workers", minimum=0),
        device=_integer(raw, "device", minimum=0),
        amp=_boolean(raw, "amp"),
        deterministic=_boolean(raw, "deterministic"),
        seed=_integer(raw, "seed", minimum=0),
        cache=_boolean(raw, "cache"),
        learning_rate=_float(raw, "learning_rate", minimum=0, inclusive=False),
        weight_decay=_float(raw, "weight_decay", minimum=0, inclusive=True),
    )
