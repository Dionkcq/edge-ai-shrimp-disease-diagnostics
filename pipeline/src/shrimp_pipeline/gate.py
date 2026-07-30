# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

_EXPECTED_MAPPING = {"0": "dark_gill", "1": "white_spot"}
_PLACEHOLDERS = {"replace_with_human_reviewer", "reviewer", "tbd", "unknown", "n/a"}
_REQUIRED_FIELDS = {
    "schema_version",
    "mapping_status",
    "accepted_mapping",
    "provisional_mapping_acknowledged",
    "author_confirmed",
    "annotation_convention_acknowledged",
    "acknowledgement",
    "evidence_report",
    "evidence_report_sha256",
    "overlay_sheets_reviewed",
    "reviewer",
    "reviewed_on",
}


class MappingGateError(RuntimeError):
    """The human mapping acceptance is absent, invalid, or not evidence-backed."""


@dataclass(frozen=True, slots=True)
class MappingAcceptance:
    schema_version: str
    mapping_status: str
    accepted_mapping: dict[str, str]
    provisional_mapping_acknowledged: bool
    author_confirmed: bool
    annotation_convention_acknowledged: bool
    acknowledgement: str
    evidence_report: Path
    evidence_report_sha256: str
    overlay_sheets_reviewed: int
    reviewer: str
    reviewed_on: date


def _strict_bool(data: dict[str, Any], field: str) -> bool:
    value = data[field]
    if type(value) is not bool:
        raise MappingGateError(f"{field} must be a JSON boolean")
    return value


def _non_empty_string(data: dict[str, Any], field: str) -> str:
    value = data[field]
    if not isinstance(value, str) or not value.strip():
        raise MappingGateError(f"{field} must be a non-empty string")
    return value.strip()


def _resolve_evidence(acceptance_path: Path, raw: str) -> Path:
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate
    from_working_directory = candidate.resolve()
    if from_working_directory.is_file():
        return from_working_directory
    return (acceptance_path.parent / candidate).resolve()


def require_mapping_acceptance(path: Path) -> MappingAcceptance:
    """Load the exact provisional mapping acceptance schema and verify its evidence."""
    if not path.is_file():
        raise MappingGateError(
            f"mapping acceptance is absent: {path}; run the audit and review overlays first"
        )
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise MappingGateError("mapping acceptance is unreadable") from exc
    if not isinstance(raw, dict):
        raise MappingGateError("mapping acceptance is unreadable: expected a JSON object")
    data: dict[str, Any] = raw
    unknown = set(data) - _REQUIRED_FIELDS
    missing = _REQUIRED_FIELDS - set(data)
    if unknown:
        raise MappingGateError(f"mapping acceptance has unknown fields: {sorted(unknown)}")
    if missing:
        raise MappingGateError(f"mapping acceptance is missing required fields: {sorted(missing)}")

    schema_version = _non_empty_string(data, "schema_version")
    if schema_version != "1.0.0":
        raise MappingGateError("schema_version must be '1.0.0'")
    status = _non_empty_string(data, "mapping_status")
    if status != "PROVISIONAL_UNCONFIRMED":
        raise MappingGateError("mapping_status must explicitly remain PROVISIONAL_UNCONFIRMED")
    if data["accepted_mapping"] != _EXPECTED_MAPPING:
        raise MappingGateError(
            "accepted_mapping must explicitly record 0=dark_gill and 1=white_spot"
        )

    provisional = _strict_bool(data, "provisional_mapping_acknowledged")
    annotation = _strict_bool(data, "annotation_convention_acknowledged")
    author_confirmed = _strict_bool(data, "author_confirmed")
    if not provisional:
        raise MappingGateError("provisional mapping acknowledgement is required")
    if not annotation:
        raise MappingGateError("annotation convention drift acknowledgement is required")
    if author_confirmed:
        raise MappingGateError("author_confirmed must be false while mapping_status is provisional")

    acknowledgement = _non_empty_string(data, "acknowledgement")
    reviewer = _non_empty_string(data, "reviewer")
    if reviewer.casefold().replace(" ", "_") in _PLACEHOLDERS or "replace" in reviewer.casefold():
        raise MappingGateError("reviewer must identify a real non-placeholder human reviewer")
    reviewed_on_text = _non_empty_string(data, "reviewed_on")
    try:
        reviewed_on = date.fromisoformat(reviewed_on_text)
    except ValueError as exc:
        raise MappingGateError(
            "reviewed_on must be an ISO 8601 calendar date (YYYY-MM-DD)"
        ) from exc
    if reviewed_on.isoformat() != reviewed_on_text:
        raise MappingGateError("reviewed_on must use canonical ISO 8601 YYYY-MM-DD form")

    overlays = data["overlay_sheets_reviewed"]
    if type(overlays) is not int or overlays < 60:
        raise MappingGateError("overlay_sheets_reviewed must be an integer of at least 60")
    report_raw = _non_empty_string(data, "evidence_report")
    report = _resolve_evidence(path, report_raw)
    if not report.is_file():
        raise MappingGateError(f"evidence_report does not exist: {report}")
    recorded_digest = _non_empty_string(data, "evidence_report_sha256").lower()
    if len(recorded_digest) != 64 or any(
        character not in "0123456789abcdef" for character in recorded_digest
    ):
        raise MappingGateError("evidence_report_sha256 must be a 64-character hexadecimal digest")
    actual_digest = hashlib.sha256(report.read_bytes()).hexdigest()
    if actual_digest != recorded_digest:
        raise MappingGateError("evidence_report_sha256 does not match the evidence report bytes")

    return MappingAcceptance(
        schema_version=schema_version,
        mapping_status=status,
        accepted_mapping=dict(_EXPECTED_MAPPING),
        provisional_mapping_acknowledged=provisional,
        author_confirmed=author_confirmed,
        annotation_convention_acknowledged=annotation,
        acknowledgement=acknowledgement,
        evidence_report=report,
        evidence_report_sha256=recorded_digest,
        overlay_sheets_reviewed=overlays,
        reviewer=reviewer,
        reviewed_on=reviewed_on,
    )
