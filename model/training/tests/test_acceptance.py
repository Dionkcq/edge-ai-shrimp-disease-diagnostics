from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from shrimp_training.acceptance import (
    AcceptanceError,
    validate_mapping_acceptance,
    validate_mapping_documents,
)


def _evidence() -> bytes:
    return b'{"schema_version":"1.0.0","overlays":60}\n'


def _acceptance(evidence_name: str, evidence: bytes) -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "mapping_status": "PROVISIONAL_UNCONFIRMED",
        "accepted_mapping": {"0": "dark_gill", "1": "white_spot"},
        "provisional_mapping_acknowledged": True,
        "author_confirmed": False,
        "annotation_convention_acknowledged": True,
        "acknowledgement": "Combined-folder class semantics and annotation drift reviewed.",
        "evidence_report": evidence_name,
        "evidence_report_sha256": hashlib.sha256(evidence).hexdigest(),
        "overlay_sheets_reviewed": 60,
        "reviewer": "Independent aquatic imaging reviewer",
        "reviewed_on": "2026-07-30",
    }


def _prepared(acceptance: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "mapping_acceptance": {
            "status": acceptance["mapping_status"],
            "reviewer": acceptance["reviewer"],
            "reviewed_on": acceptance["reviewed_on"],
            "evidence_report_sha256": acceptance["evidence_report_sha256"],
        },
    }


def test_acceptance_binds_strict_schema_evidence_and_prepared_manifest(tmp_path: Path) -> None:
    evidence = _evidence()
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_bytes(evidence)
    document = _acceptance(evidence_path.name, evidence)
    acceptance_path = tmp_path / "acceptance.json"
    acceptance_path.write_text(json.dumps(document), encoding="utf-8")
    prepared_path = tmp_path / "manifest.json"
    prepared_path.write_text(json.dumps(_prepared(document)), encoding="utf-8")

    result = validate_mapping_acceptance(acceptance_path, prepared_path)

    assert result.mapping_status == "PROVISIONAL_UNCONFIRMED"
    assert result.evidence_report == evidence_path.resolve()
    assert result.evidence_report_sha256 == hashlib.sha256(evidence).hexdigest()
    assert result.overlay_sheets_reviewed == 60


def test_acceptance_rejects_fabricated_minimal_record() -> None:
    with pytest.raises(AcceptanceError, match="missing required fields"):
        validate_mapping_documents(
            {"mapping_status": "PROVISIONAL_UNCONFIRMED"},
            _evidence(),
            {"mapping_acceptance": {"status": "PROVISIONAL_UNCONFIRMED"}},
        )


def test_acceptance_rejects_unrelated_prepared_manifest() -> None:
    evidence = _evidence()
    acceptance = _acceptance("evidence.json", evidence)
    prepared = _prepared(acceptance)
    prepared["mapping_acceptance"]["reviewer"] = "Different reviewer"  # type: ignore[index]

    with pytest.raises(AcceptanceError, match=r"prepared manifest.*acceptance"):
        validate_mapping_documents(acceptance, evidence, prepared)


def test_acceptance_rejects_tampered_evidence() -> None:
    evidence = _evidence()
    acceptance = _acceptance("evidence.json", evidence)

    with pytest.raises(AcceptanceError, match="evidence report bytes"):
        validate_mapping_documents(acceptance, evidence + b"tampered", _prepared(acceptance))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("overlay_sheets_reviewed", 59),
        ("reviewer", "TBD"),
        ("author_confirmed", True),
        ("accepted_mapping", {"0": "white_spot", "1": "dark_gill"}),
        ("provisional_mapping_acknowledged", False),
        ("annotation_convention_acknowledged", False),
    ],
)
def test_acceptance_rejects_invalid_safety_fields(field: str, value: object) -> None:
    evidence = _evidence()
    acceptance = _acceptance("evidence.json", evidence)
    acceptance[field] = value

    with pytest.raises(AcceptanceError):
        validate_mapping_documents(acceptance, evidence, _prepared(acceptance))
