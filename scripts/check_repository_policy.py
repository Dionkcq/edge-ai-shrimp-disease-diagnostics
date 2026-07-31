#!/usr/bin/env python3
"""Repository hygiene checks, evaluated over **tracked files only**.

Why this replaced the previous inline `find` in CI
--------------------------------------------------
The original check walked the working tree. That made it wrong in both directions:

* It failed on any developer machine that had the real archives in `datasets/raw/`
  (493 MB + 210 MB), which are correctly gitignored and are *supposed* to be there.
  It also matched `.venv/` -- site-packages ships `.pth` files, bundled `.onnx`
  test models and >20 MB shared objects.
* It passed in CI purely because CI has none of those files, so it was asserting
  something about the runner rather than about the repository.

A check that fails locally and passes in CI trains people to ignore it. Everything
here is therefore evaluated against `git ls-files`: the set of files a clone would
actually receive. Run it locally and in CI and it gives the same answer.

Usage:
    python scripts/check_repository_policy.py [--root PATH]
Exit status is 0 when the repository is clean and 1 with a report otherwise.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

#: Documentation a reader of this repository is entitled to find.
REQUIRED_FILES: tuple[str, ...] = (
    "README.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "LICENSE",
    "LICENSING.md",
    "datasets/DATASET_REGISTRY.md",
    "docs/LIMITATIONS.md",
    "docs/KNOWN_GAPS.md",
    "models/MODEL_CARD.md",
    "models/registry.json",
    "training/README.md",
    "training/LICENSE.AGPL",
    "training/pyproject.toml",
    "training/uv.lock",
    "training/configs/compact-nvidia-6gb.json",
    "contracts/screening_result.schema.json",
    "contracts/CONTRACT.md",
    "policy/quality_policy_v1.json",
    "policy/decision_policy_v1.json",
    "guidance/guidance_v1.json",
    "guidance/SOURCES.md",
)

#: JSON that must parse, because a malformed one of these breaks startup.
REQUIRED_JSON: tuple[str, ...] = (
    "datasets/source-notes/dataset_manifest.json",
    "models/registry.json",
    "policy/quality_policy_v1.json",
    "policy/decision_policy_v1.json",
    "guidance/guidance_v1.json",
    "contracts/screening_result.schema.json",
    "datasets/mapping_acceptance.example.json",
    "training/configs/compact-nvidia-6gb.json",
)

#: Path prefixes that must never be tracked.
FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "datasets/raw/",
    "datasets/processed/",
    "artifacts/",
    "runs/",
    "private/",
    "training/runs/",
    "training/returns/",
    "training/work/",
)

#: Model weights and other binary artifacts. `.onnx` is included deliberately: the
#: synthetic model used by the provider tests is built into `tmp_path` at test time
#: precisely so that this ban can stay absolute.
FORBIDDEN_SUFFIXES: tuple[str, ...] = (
    ".pt",
    ".pth",
    ".onnx",
    ".ckpt",
    ".safetensors",
    ".tflite",
    ".engine",
    ".pb",
    ".h5",
    ".weights",
    ".zip",
    ".tar",
    ".tar.gz",
    ".tgz",
    ".npz",
    ".parquet",
)

#: Exact tracked paths that must not exist. The acceptance record is a human
#: decision about unconfirmed dataset semantics; a committed one would let the
#: conversion gate pass without anybody having reviewed anything.
FORBIDDEN_EXACT: tuple[str, ...] = (
    "datasets/mapping_acceptance.json",
    "docs/implementation-plan.html",
    ".env",
    "HANDOFF.md",
)

#: Filenames that leak a local environment or an OS artifact.
FORBIDDEN_NAMES: tuple[str, ...] = (
    ".DS_Store",
    "Thumbs.db",
    "desktop.ini",
)

#: Nothing tracked should be this large; the repository is text plus small JSON.
MAX_TRACKED_BYTES = 2 * 1024 * 1024

#: The committed split file is the reproducibility anchor and must stay small.
MAX_SPLIT_BYTES = 1024 * 1024


def tracked_files(root: Path) -> list[str]:
    """Every path a commit would include, as repository-relative POSIX strings.

    `--cached` is the index; `--others --exclude-standard` adds files that are
    present and *not* gitignored. The union is the honest answer to "what would end
    up in the repository", and it is what makes this check give the same result in
    both places it runs:

    * In CI, after a checkout, there are no untracked files, so this is the tracked set.
    * On a developer machine it also flags a forbidden file that has been created but
      not yet staged -- which is the moment the mistake is cheap to fix -- while
      still ignoring `datasets/raw/`, `.venv/` and `artifacts/`, all of which are
      gitignored and are supposed to be present locally.
    """
    result = subprocess.run(  # noqa: S603
        ["git", "-C", str(root), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],  # noqa: S607
        capture_output=True,
        check=True,
        text=True,
    )
    return sorted({name for name in result.stdout.split("\0") if name})


def _check_required(root: Path, findings: list[str]) -> None:
    tracked = set(tracked_files(root))
    for name in REQUIRED_FILES:
        if name not in tracked:
            findings.append(
                f"required file is absent or gitignored, so a clone would not receive it: {name}"
            )
        elif not (root / name).is_file():
            findings.append(f"required file is tracked but missing on disk: {name}")


def _check_forbidden(paths: Iterable[str], findings: list[str]) -> None:
    for name in paths:
        lowered = name.lower()
        base = name.rsplit("/", 1)[-1]
        if name in FORBIDDEN_EXACT:
            findings.append(f"forbidden file is tracked: {name}")
        if base in FORBIDDEN_NAMES or base.startswith("._"):
            findings.append(f"OS metadata file is tracked: {name}")
        if name.startswith(".env.") and not name.endswith(".example"):
            findings.append(f"environment file is tracked: {name}")
        for prefix in FORBIDDEN_PREFIXES:
            if name.startswith(prefix):
                findings.append(f"generated or raw data is tracked: {name}")
        for suffix in FORBIDDEN_SUFFIXES:
            if lowered.endswith(suffix):
                findings.append(f"binary artifact is tracked: {name}")


def _check_sizes(root: Path, paths: Iterable[str], findings: list[str]) -> None:
    for name in paths:
        path = root / name
        if not path.is_file():
            continue
        size = path.stat().st_size
        if size > MAX_TRACKED_BYTES:
            findings.append(f"tracked file exceeds {MAX_TRACKED_BYTES} bytes: {name} ({size})")
        if name.startswith("datasets/splits/") and size > MAX_SPLIT_BYTES:
            findings.append(f"split file exceeds {MAX_SPLIT_BYTES} bytes: {name} ({size})")


def _check_json(root: Path, findings: list[str]) -> None:
    for name in REQUIRED_JSON:
        path = root / name
        if not path.is_file():
            continue  # absence is reported by the required-files check
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            findings.append(f"invalid JSON: {name}: {exc}")


def _check_registry_has_no_weights(root: Path, findings: list[str]) -> None:
    """No trained model exists. The registry must not claim otherwise."""
    path = root / "models" / "registry.json"
    if not path.is_file():
        return
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return  # reported by the JSON check
    models = document.get("models")
    if not isinstance(models, list):
        findings.append("models/registry.json must contain a 'models' array")
    elif models:
        findings.append(
            f"models/registry.json declares {len(models)} model(s); no trained weights "
            "exist in this repository, so it must be empty"
        )


def check_repository(root: Path) -> list[str]:
    """Return every policy violation found. Empty means clean."""
    findings: list[str] = []
    paths = tracked_files(root)
    _check_required(root, findings)
    _check_forbidden(paths, findings)
    _check_sizes(root, paths, findings)
    _check_json(root, findings)
    _check_registry_has_no_weights(root, findings)
    return findings


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)

    findings = check_repository(args.root.resolve())
    if findings:
        print("Repository policy violations:", file=sys.stderr)
        for finding in findings:
            print(f"  - {finding}", file=sys.stderr)
        return 1
    print(f"Repository policy: OK ({len(tracked_files(args.root.resolve()))} tracked files).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
