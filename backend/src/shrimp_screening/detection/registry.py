"""``models/registry.json`` -- the allowlist of model artifacts.

A model file is loadable only if its **sha256 is already recorded here**. That is
the whole point: weights are never committed, so they arrive out of band, and an
out-of-band artifact that nobody has recorded is exactly the thing that must not be
executed. Unknown digest means refuse, not warn.

On a clean checkout the registry contains zero entries, so every ONNX load fails
and ``/readyz`` reports 503. That is the correct state of this repository today.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shrimp_screening.contracts.enums import DatasetMappingStatus, OutputLayout
from shrimp_screening.paths import data_dir

_SHA256_LENGTH = 64
_ENTRY_FIELDS = {
    "model_id",
    "version",
    "filename",
    "sha256",
    "input_size",
    "class_names",
    "opset",
    "output_layout",
    "anchors",
    "dataset_mapping_status",
    "artifact_license",
    "training_toolchain",
}
_SUPPORTED_OPSETS = {17}
_TOOLCHAIN_LICENSES = {
    "ultralytics": {
        "AGPL-3.0-or-later",
        "LicenseRef-Ultralytics-Enterprise",
    },
    "synthetic-onnx-graph": {"AGPL-3.0-or-later"},
    "custom-pytorch-yolo": {"AGPL-3.0-or-later"},
}
#: Anchors per detection scale for CUSTOM_YOLO_ANCHOR_V1 (3 scales x 3 anchors).
#: Mirrors ``training/src/shrimp_training/anchors.py::ANCHORS_PER_SCALE`` -- kept
#: as a plain constant here rather than imported, since the backend never imports
#: the (AGPL, separately locked) training package.
_ANCHORS_PER_SCALE = 3
_ANCHOR_SCALES = 3


class RegistryError(RuntimeError):
    """The registry file is missing, malformed, or does not vouch for an artifact."""


@dataclass(frozen=True, slots=True)
class RegisteredModel:
    """One artifact the project has decided it is willing to execute."""

    model_id: str
    version: str
    filename: str
    sha256: str
    input_size: int
    class_names: dict[int, str]
    opset: int
    output_layout: OutputLayout
    anchors: tuple[tuple[float, float], ...]
    dataset_mapping_status: DatasetMappingStatus
    artifact_license: str
    training_toolchain: str


@dataclass(frozen=True, slots=True)
class ModelRegistry:
    """The parsed registry, indexed for the two lookups that exist."""

    models: tuple[RegisteredModel, ...]

    @property
    def is_empty(self) -> bool:
        return not self.models

    def by_sha256(self, digest: str) -> RegisteredModel:
        for model in self.models:
            if model.sha256 == digest:
                return model
        raise RegistryError(
            f"no model with sha256 {digest} is registered in models/registry.json; "
            "refusing to load an unvouched artifact"
        )

    def by_model_id(self, model_id: str) -> RegisteredModel:
        for model in self.models:
            if model.model_id == model_id:
                return model
        raise RegistryError(f"no model with id {model_id!r} is registered")


def _parse_class_names(raw: Any) -> dict[int, str]:
    if not isinstance(raw, dict) or not raw:
        raise RegistryError("class_names must be a non-empty object keyed by class index")
    parsed: dict[int, str] = {}
    for key, value in raw.items():
        try:
            index = int(key)
        except (TypeError, ValueError) as exc:
            raise RegistryError(f"class_names key {key!r} is not an integer index") from exc
        if not isinstance(value, str) or not value:
            raise RegistryError(f"class_names[{key!r}] must be a non-empty string")
        parsed[index] = value
    if sorted(parsed) != list(range(len(parsed))):
        raise RegistryError("class_names must cover 0..n-1 with no gaps")
    return parsed


def _parse_anchors(raw: Any, output_layout: OutputLayout) -> tuple[tuple[float, float], ...]:
    if not isinstance(raw, list):
        raise RegistryError("anchors must be a list of [width, height] pairs")
    parsed: list[tuple[float, float]] = []
    for pair in raw:
        if (
            not isinstance(pair, list | tuple)
            or len(pair) != 2
            or any(isinstance(value, bool) or not isinstance(value, int | float) for value in pair)
        ):
            raise RegistryError("each anchors entry must be a [width, height] numeric pair")
        width, height = float(pair[0]), float(pair[1])
        if width <= 0 or height <= 0:
            raise RegistryError("anchor width and height must be positive")
        parsed.append((width, height))
    if output_layout is OutputLayout.CUSTOM_YOLO_ANCHOR_V1:
        expected = _ANCHORS_PER_SCALE * _ANCHOR_SCALES
        if len(parsed) != expected:
            raise RegistryError(
                f"{output_layout.value} requires exactly {expected} anchors, got {len(parsed)}"
            )
    elif parsed:
        raise RegistryError(f"{output_layout.value} does not use anchors; anchors must be []")
    return tuple(parsed)


def _parse_model(entry: Any) -> RegisteredModel:
    if not isinstance(entry, dict):
        raise RegistryError("each registry entry must be an object")
    unknown = set(entry) - _ENTRY_FIELDS
    missing = _ENTRY_FIELDS - set(entry)
    if unknown:
        raise RegistryError(f"registry entry has unknown fields: {sorted(unknown)}")
    if missing:
        raise RegistryError(f"registry entry is missing required field {sorted(missing)[0]!r}")
    try:
        strings: dict[str, str] = {}
        for field in (
            "model_id",
            "version",
            "filename",
            "artifact_license",
            "training_toolchain",
        ):
            value = entry[field]
            if not isinstance(value, str) or not value.strip():
                raise RegistryError(f"{field} must be a non-empty string")
            strings[field] = value.strip()
        digest_value = entry["sha256"]
        digest = digest_value.lower() if isinstance(digest_value, str) else ""
        if len(digest) != _SHA256_LENGTH or not all(c in "0123456789abcdef" for c in digest):
            raise RegistryError(f"sha256 {digest!r} is not a 64-character hex digest")
        opset_raw = entry["opset"]
        if type(opset_raw) is not int or opset_raw not in _SUPPORTED_OPSETS:
            raise RegistryError(f"opset is invalid; it must be one of {sorted(_SUPPORTED_OPSETS)}")
        input_size = entry["input_size"]
        if type(input_size) is not int or input_size <= 0 or input_size % 32 != 0:
            raise RegistryError("input_size must be a positive integer divisible by 32")
        toolchain = strings["training_toolchain"]
        toolchain_family = next(
            (family for family in _TOOLCHAIN_LICENSES if toolchain.casefold().startswith(family)),
            None,
        )
        if toolchain_family is None:
            raise RegistryError("training_toolchain is not one of the reviewed toolchain families")
        artifact_license = strings["artifact_license"]
        if artifact_license not in _TOOLCHAIN_LICENSES[toolchain_family]:
            raise RegistryError(
                "artifact_license is inconsistent with the registered training_toolchain"
            )
        output_layout = OutputLayout(entry["output_layout"])
        return RegisteredModel(
            model_id=strings["model_id"],
            version=strings["version"],
            filename=strings["filename"],
            sha256=digest,
            input_size=input_size,
            class_names=_parse_class_names(entry["class_names"]),
            opset=opset_raw,
            output_layout=output_layout,
            anchors=_parse_anchors(entry["anchors"], output_layout),
            dataset_mapping_status=DatasetMappingStatus(entry["dataset_mapping_status"]),
            artifact_license=artifact_license,
            training_toolchain=toolchain,
        )
    except KeyError as exc:
        raise RegistryError(f"registry entry is missing required field {exc.args[0]!r}") from exc
    except ValueError as exc:
        raise RegistryError(f"registry entry is invalid: {exc}") from exc


def load_registry(path: Path | None = None) -> ModelRegistry:
    """Parse ``models/registry.json``. Raises :class:`RegistryError` on any defect."""
    source = path if path is not None else data_dir() / "model_registry.json"
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RegistryError(f"the model registry could not be read: {source}") from exc
    except json.JSONDecodeError as exc:
        raise RegistryError(f"the model registry is not valid JSON: {source}") from exc
    if not isinstance(document, dict) or not isinstance(document.get("models"), list):
        raise RegistryError("the model registry must be an object with a 'models' array")
    models = tuple(_parse_model(entry) for entry in document["models"])
    for field in ("model_id", "filename", "sha256"):
        values = [getattr(model, field) for model in models]
        if len(values) != len(set(values)):
            raise RegistryError(f"model registry contains a duplicate {field}")
    return ModelRegistry(models=models)


def sha256_of(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Streaming sha256 of a file, so a large artifact is never held in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
