"""Tests for the repository-policy checker.

A CI check nobody has tested is a CI check that passes for the wrong reason — which
is exactly what happened to the `find`-based predecessor of this script. So each
rule is exercised against a throwaway Git repository built in `tmp_path`, and the
real repository is asserted to be clean.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - import bootstrap
    sys.path.insert(0, str(REPO_ROOT))

from scripts.check_repository_policy import (  # noqa: E402
    REQUIRED_FILES,
    REQUIRED_JSON,
    check_repository,
    tracked_files,
)


def _git(root: Path, *args: str) -> None:
    subprocess.run(  # noqa: S603
        ["git", "-C", str(root), *args],  # noqa: S607
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def sandbox(tmp_path: Path) -> Path:
    """A minimal repository that satisfies every rule, ready to be broken."""
    _git(tmp_path, "init", "-q")
    for name in REQUIRED_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if name in REQUIRED_JSON:
            path.write_text("{}", encoding="utf-8")
        else:
            path.write_text(f"# {name}\n", encoding="utf-8")
    for name in REQUIRED_JSON:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    (tmp_path / "models" / "registry.json").write_text(json.dumps({"models": []}), encoding="utf-8")
    _git(tmp_path, "add", "-A")
    return tmp_path


def test_a_clean_sandbox_reports_no_violations(sandbox: Path) -> None:
    assert check_repository(sandbox) == []


def test_the_real_repository_is_clean() -> None:
    """The check must pass here, not only on a CI runner."""
    assert check_repository(REPO_ROOT) == []


def test_ignored_local_data_is_not_reported(sandbox: Path) -> None:
    """The predecessor failed on exactly this: gitignored archives present locally.

    `datasets/raw/*.zip` is 700 MB on the author's machine and is supposed to be
    there. A check that fails because of it trains people to ignore the check.
    """
    (sandbox / ".gitignore").write_text("datasets/raw/\n", encoding="utf-8")
    raw = sandbox / "datasets" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "archive.zip").write_bytes(b"\x00" * (3 * 1024 * 1024))
    _git(sandbox, "add", "-A")
    assert check_repository(sandbox) == []


@pytest.mark.parametrize(
    ("relative_path", "expected_fragment"),
    [
        ("datasets/raw/leaked.txt", "raw data is tracked"),
        ("datasets/processed/leaked.txt", "raw data is tracked"),
        ("artifacts/audit.json", "raw data is tracked"),
        ("models/best.pt", "binary artifact"),
        ("models/best.onnx", "binary artifact"),
        ("models/best.safetensors", "binary artifact"),
        ("datasets/archive.zip", "binary artifact"),
        ("datasets/mapping_acceptance.json", "forbidden file"),
        ("docs/implementation-plan.html", "forbidden file"),
        (".DS_Store", "OS metadata"),
        ("datasets/._AppleDouble", "OS metadata"),
        (".env.production", "environment file"),
    ],
)
def test_each_forbidden_path_is_detected(
    sandbox: Path, relative_path: str, expected_fragment: str
) -> None:
    path = sandbox / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")
    _git(sandbox, "add", "-Af")
    findings = check_repository(sandbox)
    assert any(expected_fragment in f for f in findings), findings


def test_an_untracked_but_unignored_violation_is_still_caught(sandbox: Path) -> None:
    """The mistake should surface before it is staged, while it is cheap to undo."""
    (sandbox / "models").mkdir(parents=True, exist_ok=True)
    (sandbox / "models" / "sneaky.onnx").write_text("x", encoding="utf-8")
    findings = check_repository(sandbox)
    assert any("binary artifact" in f for f in findings), findings


@pytest.mark.parametrize("name", ["LICENSING.md", "docs/KNOWN_GAPS.md", "models/MODEL_CARD.md"])
def test_a_missing_required_document_is_reported(sandbox: Path, name: str) -> None:
    _git(sandbox, "rm", "-q", "--cached", name)
    (sandbox / name).unlink()
    findings = check_repository(sandbox)
    assert any(name in f for f in findings), findings


def test_malformed_required_json_is_reported(sandbox: Path) -> None:
    (sandbox / "policy" / "decision_policy_v1.json").write_text("{oops", encoding="utf-8")
    findings = check_repository(sandbox)
    assert any("invalid JSON" in f for f in findings), findings


def test_an_oversized_tracked_file_is_reported(sandbox: Path) -> None:
    (sandbox / "big.md").write_text("x" * (3 * 1024 * 1024), encoding="utf-8")
    _git(sandbox, "add", "-A")
    findings = check_repository(sandbox)
    assert any("exceeds" in f for f in findings), findings


def test_an_oversized_split_file_is_reported(sandbox: Path) -> None:
    splits = sandbox / "datasets" / "splits"
    splits.mkdir(parents=True, exist_ok=True)
    (splits / "specimen_split_v1.json").write_text(
        json.dumps({"pad": "x" * (1024 * 1024 + 64)}), encoding="utf-8"
    )
    _git(sandbox, "add", "-A")
    findings = check_repository(sandbox)
    assert any("split file exceeds" in f for f in findings), findings


def test_a_registry_claiming_a_model_is_reported(sandbox: Path) -> None:
    """No weights exist. The registry must not say otherwise."""
    (sandbox / "models" / "registry.json").write_text(
        json.dumps({"models": [{"id": "imaginary", "sha256": "0" * 64}]}), encoding="utf-8"
    )
    _git(sandbox, "add", "-A")
    findings = check_repository(sandbox)
    assert any("must be empty" in f for f in findings), findings


def test_the_real_registry_declares_no_weights() -> None:
    document = json.loads((REPO_ROOT / "models" / "registry.json").read_text("utf-8"))
    assert document["models"] == []


def test_tracked_files_is_sorted_and_deduplicated(sandbox: Path) -> None:
    names = tracked_files(sandbox)
    assert names == sorted(set(names))


def test_the_script_exits_non_zero_on_a_dirty_repository(sandbox: Path) -> None:
    (sandbox / "models" / "best.pt").write_text("x", encoding="utf-8")
    _git(sandbox, "add", "-Af")
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "check_repository_policy.py"),
            "--root",
            str(sandbox),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "binary artifact" in result.stderr


def test_the_script_exits_zero_on_the_real_repository() -> None:
    result = subprocess.run(  # noqa: S603
        [sys.executable, str(REPO_ROOT / "scripts" / "check_repository_policy.py")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
