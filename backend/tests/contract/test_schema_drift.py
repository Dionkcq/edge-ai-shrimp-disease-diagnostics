"""The committed JSON Schema is the interface the frontend will be generated from.

If it can drift from the Pydantic models, then the models are no longer the single
source of truth and a silent contract break becomes possible.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shrimp_screening.contracts.export_schema import SCHEMA_ARTIFACTS, render_schema
from shrimp_screening.contracts.screening import SCHEMA_VERSION
from shrimp_screening.paths import repository_root


@pytest.mark.parametrize("artifact", SCHEMA_ARTIFACTS, ids=lambda a: a.relative_path)
def test_committed_schema_byte_equals_a_fresh_generation(artifact: object) -> None:
    committed = repository_root() / artifact.relative_path  # type: ignore[attr-defined]
    assert committed.is_file(), (
        f"{artifact.relative_path} is missing; run `make schema`"  # type: ignore[attr-defined]
    )
    assert committed.read_text(encoding="utf-8") == render_schema(artifact)  # type: ignore[arg-type]


def test_screening_schema_forbids_unknown_properties() -> None:
    schema = json.loads((repository_root() / "contracts/screening_result.schema.json").read_text())
    assert schema["additionalProperties"] is False
    for name, definition in schema.get("$defs", {}).items():
        if definition.get("type") == "object":
            assert definition.get("additionalProperties") is False, f"$defs.{name} is open"


def test_schema_pins_the_decision_enumeration() -> None:
    schema = json.loads((repository_root() / "contracts/screening_result.schema.json").read_text())
    decision = schema["$defs"]["Decision"]
    assert sorted(decision["enum"]) == sorted(
        [
            "GILL_DARKENING_MARKER_DETECTED",
            "MULTIPLE_TARGET_MARKERS_DETECTED",
            "NO_TARGET_MARKER_DETECTED",
            "UNABLE_TO_ASSESS",
            "WHITE_SPOT_MARKER_DETECTED",
        ]
    )


def test_contract_document_records_the_current_schema_version() -> None:
    contract_doc = (repository_root() / "contracts/CONTRACT.md").read_text(encoding="utf-8")
    assert SCHEMA_VERSION in contract_doc


def test_schema_file_is_small_enough_to_review_by_hand() -> None:
    for artifact in SCHEMA_ARTIFACTS:
        path = repository_root() / artifact.relative_path
        assert path.stat().st_size < 64_000, f"{artifact.relative_path} has become unreviewable"


def test_no_generated_schema_leaks_a_local_filesystem_path() -> None:
    root = str(Path.home())
    for artifact in SCHEMA_ARTIFACTS:
        text = (repository_root() / artifact.relative_path).read_text(encoding="utf-8")
        assert root not in text
