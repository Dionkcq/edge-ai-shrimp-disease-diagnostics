"""Reading the versioned policy files and hashing them.

Thresholds live in ``policy/*.json`` rather than in code so a reviewer can see and
change them without a release, and so the exact bytes that produced a result are
identifiable. The response echoes ``policy_id`` plus ``sha256:`` of the file, which
makes a screening result reproducible: same image, same policy hash, same decision.

Every threshold currently shipped is marked ``UNCALIBRATED`` in the file itself.
Nothing in this repository has ever been measured against ground truth, and the
API surfaces that as a ``THRESHOLDS_UNCALIBRATED`` notice.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shrimp_screening.contracts.enums import MarkerRole
from shrimp_screening.paths import data_dir


class PolicyError(RuntimeError):
    """A policy file is missing, malformed or internally inconsistent."""


_QUALITY_FIELDS = {
    "policy_id",
    "status",
    "status_note",
    "minimum_side_px",
    "minimum_side_note",
    "minimum_blur_score",
    "minimum_mean_luminance",
    "maximum_mean_luminance",
    "minimum_rms_contrast",
}
_DECISION_FIELDS = {
    "policy_id",
    "status",
    "status_note",
    "candidate_detection_score",
    "candidate_note",
    "minimum_detection_score",
    "minimum_detection_note",
    "moderate_score",
    "high_score",
    "iou_threshold",
    "iou_note",
    "max_detections",
    "multi_label",
    "multi_label_note",
    "class_roles",
}


def _check_exact_fields(document: dict[str, Any], allowed: set[str], path: Path) -> None:
    unknown = set(document) - allowed
    missing = allowed - set(document)
    if unknown:
        raise PolicyError(f"policy file {path.name} has unknown fields: {sorted(unknown)}")
    if missing:
        raise PolicyError(f"policy file {path.name} is missing required fields: {sorted(missing)}")


def _non_empty_string(document: dict[str, Any], key: str, path: Path) -> str:
    value = _field(document, key, path)
    if not isinstance(value, str) or not value.strip():
        raise PolicyError(f"policy file {path.name} field {key!r} must be a non-empty string")
    return value.strip()


def _finite_number(document: dict[str, Any], key: str, path: Path) -> float:
    value = _field(document, key, path)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PolicyError(f"policy file {path.name} field {key!r} must be a finite number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise PolicyError(f"policy file {path.name} field {key!r} must be a finite number")
    return parsed


def _positive_int(document: dict[str, Any], key: str, path: Path) -> int:
    value = _field(document, key, path)
    if type(value) is not int or value <= 0:
        raise PolicyError(f"policy file {path.name} field {key!r} must be a positive integer")
    return value


def _load_document(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise PolicyError(f"policy file could not be read: {path}") from exc
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PolicyError(f"policy file is not valid JSON: {path}") from exc
    if not isinstance(document, dict):
        raise PolicyError(f"policy file must contain a JSON object: {path}")
    return document, "sha256:" + hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True, slots=True)
class QualityPolicy:
    """Thresholds for the deterministic, model-free retake gate."""

    policy_id: str
    policy_hash: str
    minimum_side_px: int
    minimum_blur_score: float
    minimum_mean_luminance: float
    maximum_mean_luminance: float
    minimum_rms_contrast: float
    status: str

    def __post_init__(self) -> None:
        if not self.policy_id.strip() or not self.status.strip():
            raise PolicyError("policy_id and status must be non-empty")
        numeric = (
            self.minimum_blur_score,
            self.minimum_mean_luminance,
            self.maximum_mean_luminance,
            self.minimum_rms_contrast,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise PolicyError("quality thresholds must be finite")
        if not 0.0 <= self.minimum_mean_luminance < self.maximum_mean_luminance <= 255.0:
            raise PolicyError(
                "minimum_mean_luminance and maximum_mean_luminance must satisfy "
                "0 <= min < max <= 255"
            )
        if self.minimum_side_px < 1:
            raise PolicyError("minimum_side_px must be positive")
        if self.minimum_blur_score < 0.0 or self.minimum_rms_contrast < 0.0:
            raise PolicyError("blur and contrast thresholds must be non-negative")


@dataclass(frozen=True, slots=True)
class DecisionPolicy:
    """Thresholds and the class-name-to-role map that turn boxes into a decision."""

    policy_id: str
    policy_hash: str
    #: The decode/NMS floor. Must stay *below* ``minimum_detection_score``: the gap
    #: between the two is what makes ``LOW_CONFIDENCE`` reachable rather than dead
    #: code. See ``candidate_note`` in the policy file.
    candidate_detection_score: float
    minimum_detection_score: float
    moderate_score: float
    high_score: float
    iou_threshold: float
    max_detections: int
    #: Keyed by the model's **own class name**, never by class index.
    class_roles: dict[str, MarkerRole]
    status: str

    def __post_init__(self) -> None:
        if not self.policy_id.strip() or not self.status.strip():
            raise PolicyError("policy_id and status must be non-empty")
        scores = (
            self.candidate_detection_score,
            self.minimum_detection_score,
            self.moderate_score,
            self.high_score,
            self.iou_threshold,
        )
        if not all(math.isfinite(value) for value in scores):
            raise PolicyError("decision thresholds must be finite")
        ordered = (
            0.0
            <= self.candidate_detection_score
            <= self.minimum_detection_score
            <= self.moderate_score
            <= self.high_score
            <= 1.0
        )
        if not ordered:
            raise PolicyError(
                "scores must satisfy 0 <= candidate_detection_score <= "
                "minimum_detection_score <= moderate_score <= high_score <= 1"
            )
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise PolicyError("iou_threshold must lie in [0, 1]")
        if self.max_detections <= 0:
            raise PolicyError("max_detections must be positive")
        if not self.class_roles or any(not name.strip() for name in self.class_roles):
            raise PolicyError("class_roles must name at least one screening-relevant class")

    def role_for(self, class_name: str) -> MarkerRole | None:
        """The screening role of a class name, or ``None`` if the policy ignores it."""
        return self.class_roles.get(class_name)


def _field(document: dict[str, Any], key: str, path: Path) -> Any:
    if key not in document:
        raise PolicyError(f"policy file {path.name} is missing required field {key!r}")
    return document[key]


def load_quality_policy(path: Path | None = None) -> QualityPolicy:
    source = path if path is not None else data_dir() / "quality_policy_v1.json"
    document, digest = _load_document(source)
    _check_exact_fields(document, _QUALITY_FIELDS, source)
    return QualityPolicy(
        policy_id=_non_empty_string(document, "policy_id", source),
        policy_hash=digest,
        minimum_side_px=_positive_int(document, "minimum_side_px", source),
        minimum_blur_score=_finite_number(document, "minimum_blur_score", source),
        minimum_mean_luminance=_finite_number(document, "minimum_mean_luminance", source),
        maximum_mean_luminance=_finite_number(document, "maximum_mean_luminance", source),
        minimum_rms_contrast=_finite_number(document, "minimum_rms_contrast", source),
        status=_non_empty_string(document, "status", source),
    )


def load_decision_policy(path: Path | None = None) -> DecisionPolicy:
    source = path if path is not None else data_dir() / "decision_policy_v1.json"
    document, digest = _load_document(source)
    _check_exact_fields(document, _DECISION_FIELDS, source)
    raw_multi_label = _field(document, "multi_label", source)
    if type(raw_multi_label) is not bool:
        raise PolicyError(f"policy file {source.name} field 'multi_label' must be a JSON boolean")
    if raw_multi_label:
        raise PolicyError(f"policy file {source.name} field 'multi_label' must remain false")
    raw_roles = _field(document, "class_roles", source)
    if not isinstance(raw_roles, dict) or not raw_roles:
        raise PolicyError("class_roles must be a non-empty object mapping class name to role")
    try:
        roles = {
            name: MarkerRole(value)
            for name, value in raw_roles.items()
            if isinstance(name, str) and name.strip() and isinstance(value, str)
        }
        if len(roles) != len(raw_roles):
            raise PolicyError("class_roles keys and values must be non-empty strings")
    except ValueError as exc:
        raise PolicyError(f"policy file {source.name} has an unusable class role: {exc}") from exc
    return DecisionPolicy(
        policy_id=_non_empty_string(document, "policy_id", source),
        policy_hash=digest,
        candidate_detection_score=_finite_number(document, "candidate_detection_score", source),
        minimum_detection_score=_finite_number(document, "minimum_detection_score", source),
        moderate_score=_finite_number(document, "moderate_score", source),
        high_score=_finite_number(document, "high_score", source),
        iou_threshold=_finite_number(document, "iou_threshold", source),
        max_detections=_positive_int(document, "max_detections", source),
        class_roles=roles,
        status=_non_empty_string(document, "status", source),
    )
