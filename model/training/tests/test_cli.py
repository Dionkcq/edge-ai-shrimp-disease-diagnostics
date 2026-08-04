from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest

from shrimp_training import cli
from shrimp_training.acceptance import AcceptanceError
from shrimp_training.cli import build_parser


def test_run_all_parser_requires_explicit_private_inputs_and_version() -> None:
    args = build_parser().parse_args(
        [
            "run-all",
            "--dataset-root",
            "private/prepared",
            "--mapping-acceptance",
            "private/acceptance.json",
            "--profile",
            "training/configs/compact-nvidia-6gb.json",
            "--work-dir",
            "private/run-v1",
            "--bundle",
            "private/return-v1.zip",
            "--version",
            "1.0.0",
        ]
    )

    assert args.dataset_root == Path("private/prepared")
    assert args.mapping_acceptance == Path("private/acceptance.json")
    assert args.parity_tolerance == 0.01
    assert args.model_id == "shrimp-marker-custom-yolo"


def test_preflight_and_verify_bundle_commands_parse_paths() -> None:
    preflight = build_parser().parse_args(["preflight", "--output", "private/preflight.json"])
    verify = build_parser().parse_args(
        [
            "verify-bundle",
            "private/return.zip",
            "--expected-manifest-sha256",
            "a" * 64,
        ]
    )

    assert preflight.output == Path("private/preflight.json")
    assert verify.bundle == Path("private/return.zip")
    assert verify.expected_manifest_sha256 == "a" * 64


def test_cli_preserves_windows_paths_with_spaces_unicode_and_metacharacters() -> None:
    unusual = Path("private/space & caret ^ percent % bang !/模型 return.zip")
    args = build_parser().parse_args(
        [
            "verify-bundle",
            str(unusual),
            "--expected-manifest-sha256",
            "b" * 64,
        ]
    )

    assert args.bundle == unusual


def test_run_all_rejects_acceptance_before_preflight_or_work_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "prepared" / "manifest.json"
    work_dir = tmp_path / "work"
    args = argparse.Namespace(
        dataset_root=manifest.parent,
        mapping_acceptance=tmp_path / "fabricated.json",
        work_dir=work_dir,
    )

    def reject_acceptance(*_: object) -> None:
        raise AcceptanceError("invalid acceptance")

    def reject_preflight() -> None:
        raise AssertionError("preflight must not run")

    monkeypatch.setattr(
        cli,
        "validate_prepared_dataset",
        lambda _: SimpleNamespace(manifest=manifest),
    )
    monkeypatch.setattr(cli, "validate_mapping_acceptance", reject_acceptance, raising=False)
    monkeypatch.setattr(cli, "run_preflight", reject_preflight)

    with pytest.raises(AcceptanceError, match="invalid acceptance"):
        cli._run_all(args)

    assert not work_dir.exists()
