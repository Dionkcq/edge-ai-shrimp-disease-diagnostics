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


class AcceptanceError(RuntimeError):
    """The mapping acceptance is absent, invalid, or not bound to its evidence."""


@dataclass(frozen=True, slots=True)
class MappingAcceptance:
    mapping_status: str
    evidence_report: Path
    evidence_report_sha256: str
    overlay_sheets_reviewed: int
    reviewer: str
    reviewed_on: date


def _object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AcceptanceError(f"{name} must be a JSON object")
    return value


def _string(data: dict[str, Any], field: str) -> str:
    value = data[field]
    if not isinstance(value, str) or not value.strip():
        raise AcceptanceError(f"{field} must be a non-empty string")
    return value.strip()


def _bool(data: dict[str, Any], field: str) -> bool:
    value = data[field]
    if type(value) is not bool:
        raise AcceptanceError(f"{field} must be a JSON boolean")
    return value


def _validate_date(value: str) -> date:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise AcceptanceError("reviewed_on must be an ISO 8601 calendar date") from exc
    if parsed.isoformat() != value:
        raise AcceptanceError("reviewed_on must use canonical YYYY-MM-DD form")
    return parsed


def _evidence_digest(data: dict[str, Any], evidence_bytes: bytes) -> str:
    recorded = _string(data, "evidence_report_sha256").lower()
    if len(recorded) != 64 or any(character not in "0123456789abcdef" for character in recorded):
        raise AcceptanceError("evidence_report_sha256 must be a hexadecimal SHA-256 digest")
    if hashlib.sha256(evidence_bytes).hexdigest() != recorded:
        raise AcceptanceError("evidence_report_sha256 does not match the evidence report bytes")
    return recorded


def validate_mapping_documents(
    acceptance_document: object,
    evidence_bytes: bytes,
    prepared_document: object,
    *,
    evidence_report: Path | None = None,
) -> MappingAcceptance:
    """Validate the strict acceptance schema and bind it to evidence and preparation."""
    data = _object(acceptance_document, "mapping acceptance")
    unknown = set(data) - _REQUIRED_FIELDS
    missing = _REQUIRED_FIELDS - set(data)
    if unknown:
        raise AcceptanceError(f"mapping acceptance has unknown fields: {sorted(unknown)}")
    if missing:
        raise AcceptanceError(f"mapping acceptance is missing required fields: {sorted(missing)}")

    if _string(data, "schema_version") != "1.0.0":
        raise AcceptanceError("schema_version must be 1.0.0")
    status = _string(data, "mapping_status")
    if status != "PROVISIONAL_UNCONFIRMED":
        raise AcceptanceError("mapping_status must remain PROVISIONAL_UNCONFIRMED")
    if data["accepted_mapping"] != _EXPECTED_MAPPING:
        raise AcceptanceError("accepted_mapping must be 0=dark_gill and 1=white_spot")
    if not _bool(data, "provisional_mapping_acknowledged"):
        raise AcceptanceError("provisional mapping acknowledgement is required")
    if _bool(data, "author_confirmed"):
        raise AcceptanceError("author_confirmed must be false while the mapping is provisional")
    if not _bool(data, "annotation_convention_acknowledged"):
        raise AcceptanceError("annotation convention acknowledgement is required")
    _string(data, "acknowledgement")

    reviewer = _string(data, "reviewer")
    reviewer_key = reviewer.casefold().replace(" ", "_")
    if reviewer_key in _PLACEHOLDERS or "replace" in reviewer.casefold():
        raise AcceptanceError("reviewer must identify a non-placeholder human reviewer")
    reviewed_on_text = _string(data, "reviewed_on")
    reviewed_on = _validate_date(reviewed_on_text)

    overlays = data["overlay_sheets_reviewed"]
    if type(overlays) is not int or overlays < 60:
        raise AcceptanceError("overlay_sheets_reviewed must be an integer of at least 60")
    report_name = _string(data, "evidence_report")
    recorded_digest = _evidence_digest(data, evidence_bytes)

    prepared = _object(prepared_document, "prepared manifest")
    expected_binding = {
        "status": status,
        "reviewer": reviewer,
        "reviewed_on": reviewed_on_text,
        "evidence_report_sha256": recorded_digest,
    }
    if prepared.get("mapping_acceptance") != expected_binding:
        raise AcceptanceError("prepared manifest mapping acceptance does not match acceptance")

    return MappingAcceptance(
        mapping_status=status,
        evidence_report=evidence_report or Path(report_name),
        evidence_report_sha256=recorded_digest,
        overlay_sheets_reviewed=overlays,
        reviewer=reviewer,
        reviewed_on=reviewed_on,
    )


def _read_json(path: Path, name: str) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AcceptanceError(f"{name} is unreadable") from exc


def validate_mapping_acceptance(
    acceptance_path: Path, prepared_manifest_path: Path
) -> MappingAcceptance:
    """Load and validate an acceptance file, its evidence, and the prepared manifest."""
    acceptance_document = _read_json(acceptance_path, "mapping acceptance")
    data = _object(acceptance_document, "mapping acceptance")
    raw_report = data.get("evidence_report")
    if not isinstance(raw_report, str) or not raw_report.strip():
        raise AcceptanceError("evidence_report must be a non-empty string")
    candidate = Path(raw_report)
    if not candidate.is_absolute():
        working_directory_candidate = candidate.resolve()
        candidate = (
            working_directory_candidate
            if working_directory_candidate.is_file()
            else (acceptance_path.parent / candidate).resolve()
        )
    if not candidate.is_file() or candidate.is_symlink():
        raise AcceptanceError(f"evidence_report does not exist as a regular file: {candidate}")
    try:
        evidence_bytes = candidate.read_bytes()
    except OSError as exc:
        raise AcceptanceError("evidence_report is unreadable") from exc
    prepared_document = _read_json(prepared_manifest_path, "prepared manifest")
    return validate_mapping_documents(
        acceptance_document,
        evidence_bytes,
        prepared_document,
        evidence_report=candidate,
    )
