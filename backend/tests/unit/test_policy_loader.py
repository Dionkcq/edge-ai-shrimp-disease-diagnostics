from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from shrimp_screening.policy.loader import PolicyError, load_decision_policy, load_quality_policy


def _write(path: Path, document: dict[str, Any]) -> Path:
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _quality() -> dict[str, Any]:
    return {
        "policy_id": "quality-test",
        "status": "UNCALIBRATED",
        "status_note": "test",
        "minimum_side_px": 640,
        "minimum_side_note": "test",
        "minimum_blur_score": 18.0,
        "minimum_mean_luminance": 25.0,
        "maximum_mean_luminance": 235.0,
        "minimum_rms_contrast": 12.0,
    }


def _decision() -> dict[str, Any]:
    return {
        "policy_id": "decision-test",
        "status": "UNCALIBRATED",
        "status_note": "test",
        "candidate_detection_score": 0.15,
        "candidate_note": "test",
        "minimum_detection_score": 0.35,
        "minimum_detection_note": "test",
        "moderate_score": 0.55,
        "high_score": 0.8,
        "iou_threshold": 0.45,
        "iou_note": "test",
        "max_detections": 300,
        "multi_label": False,
        "multi_label_note": "test",
        "class_roles": {"dark_gill": "GILL_DARKENING", "white_spot": "WHITE_SPOT"},
    }


@pytest.mark.parametrize(
    ("factory", "loader"),
    [(_quality, load_quality_policy), (_decision, load_decision_policy)],
)
def test_policy_documents_reject_unknown_fields(
    tmp_path: Path,
    factory: Callable[[], dict[str, Any]],
    loader: Callable[[Path | None], object],
) -> None:
    document = factory()
    document["typo_threshold"] = 0.5
    with pytest.raises(PolicyError, match="unknown fields"):
        loader(_write(tmp_path / "policy.json", document))


@pytest.mark.parametrize(
    ("factory", "loader"),
    [(_quality, load_quality_policy), (_decision, load_decision_policy)],
)
@pytest.mark.parametrize(("field", "value"), [("policy_id", ""), ("status", "  ")])
def test_policy_documents_require_non_empty_identity_fields(
    tmp_path: Path,
    factory: Callable[[], dict[str, Any]],
    loader: Callable[[Path | None], object],
    field: str,
    value: object,
) -> None:
    document = factory()
    document[field] = value
    with pytest.raises(PolicyError, match=field):
        loader(_write(tmp_path / "policy.json", document))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("minimum_side_px", True),
        ("minimum_blur_score", False),
        ("minimum_mean_luminance", float("nan")),
        ("maximum_mean_luminance", float("inf")),
        ("minimum_rms_contrast", float("-inf")),
    ],
)
def test_quality_policy_rejects_boolean_or_non_finite_numbers(
    tmp_path: Path, field: str, value: object
) -> None:
    document = _quality()
    document[field] = value
    with pytest.raises(PolicyError, match=field):
        load_quality_policy(_write(tmp_path / "quality.json", document))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("candidate_detection_score", True),
        ("minimum_detection_score", float("nan")),
        ("moderate_score", float("inf")),
        ("high_score", float("-inf")),
        ("iou_threshold", False),
        ("max_detections", True),
        ("max_detections", 0),
        ("multi_label", 0),
        ("multi_label", True),
    ],
)
def test_decision_policy_rejects_unsafe_numeric_and_multilabel_values(
    tmp_path: Path, field: str, value: object
) -> None:
    document = _decision()
    document[field] = value
    with pytest.raises(PolicyError, match=field):
        load_decision_policy(_write(tmp_path / "decision.json", document))
