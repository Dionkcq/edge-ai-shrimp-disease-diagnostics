from __future__ import annotations

from pathlib import Path

import pytest

from shrimp_pipeline.gate import MappingGateError, require_mapping_acceptance
from shrimp_pipeline.manifest import parse_specimen_key
from shrimp_pipeline.split import grouped_split


def test_specimen_key_uses_folder_and_number() -> None:
    assert parse_specimen_key("WSSV-12-img-3.jpg") == ("WSSV", 12)
    assert parse_specimen_key("BG-12-img-1.txt") == ("BG", 12)
    assert parse_specimen_key("junk.jpg") is None


def test_mapping_gate_refuses_when_the_acceptance_record_is_absent(tmp_path: Path) -> None:
    with pytest.raises(MappingGateError, match="absent"):
        require_mapping_acceptance(tmp_path / "missing.json")


def test_mapping_gate_refuses_unreadable_json(tmp_path: Path) -> None:
    path = tmp_path / "acceptance.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(MappingGateError, match="unreadable"):
        require_mapping_acceptance(path)


def test_the_repository_ships_no_signed_acceptance_file() -> None:
    root = Path(__file__).resolve().parents[3]
    assert not (root / "datasets" / "mapping_acceptance.json").exists(), (
        "datasets/mapping_acceptance.json must not exist in the repository: signing it "
        "is a reviewer's decision, not a checked-in default"
    )
    assert (root / "datasets" / "mapping_acceptance.example.json").is_file()


def test_grouped_split_has_no_specimen_leakage() -> None:
    keys = [("BG", i) for i in range(20)] + [("WSSV", i) for i in range(20)]
    result = grouped_split(keys, seed=42)
    assert not (set(result.train) & set(result.validation))
    assert not (set(result.train) & set(result.test))
    assert not (set(result.validation) & set(result.test))
    assert set(result.train + result.validation + result.test) == set(keys)
