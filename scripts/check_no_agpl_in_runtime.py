#!/usr/bin/env python3
"""Enforce the licence boundary between the MIT runtime and the AGPL training tooling.

The repository root is MIT. Ultralytics -- the intended training and export
toolchain -- is AGPL-3.0. Those two facts cannot both hold for a file that imports
ultralytics, so `pipeline/` is quarantined and the served application must never
reach it. This script is the enforcement; `LICENSING.md` is the explanation.

Four independent checks, because each catches something the others cannot:

1. **Declared dependencies.** `backend/pyproject.toml` must not name a forbidden
   distribution. Catches the obvious mistake.
2. **The resolved lockfile.** A forbidden distribution must not appear anywhere in
   `uv.lock`. Catches a *transitive* pull-in, which a reader of pyproject cannot see.
3. **Static imports.** No module under `backend/src` may import a forbidden
   top-level package, or `shrimp_pipeline`. Catches an import that a lockfile
   check cannot, because the package might be installed for other reasons.
4. **Runtime import graph.** Importing the app must not populate `sys.modules` with
   a forbidden package. This is the only check that accounts for a lazy or
   conditional import inside a dependency.

Usage:
    python scripts/check_no_agpl_in_runtime.py [--root PATH]
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path

#: Distributions the runtime must never resolve. `ultralytics` and `torch` are the
#: AGPL exposure; `opencv-python` is excluded for size and wheel-availability
#: reasons (four numpy functions replace it) and is asserted here so it cannot
#: return through a transitive dependency unnoticed.
FORBIDDEN_DISTRIBUTIONS: frozenset[str] = frozenset(
    {
        "ultralytics",
        "ultralytics-thop",
        "torch",
        "torchvision",
        "torchaudio",
        "opencv-python",
        "opencv-python-headless",
        "opencv-contrib-python",
    }
)

#: Top-level module names the runtime must never import.
FORBIDDEN_MODULES: frozenset[str] = frozenset(
    {"ultralytics", "torch", "torchvision", "torchaudio", "cv2", "shrimp_pipeline"}
)

#: The application entry point, imported to observe what it actually drags in.
RUNTIME_ENTRY_MODULE = "shrimp_screening.main"


def _normalize(name: str) -> str:
    return name.lower().replace("_", "-")


def check_declared_dependencies(root: Path) -> list[str]:
    """Check 1: nothing forbidden is named in the runtime package metadata."""
    path = root / "backend" / "pyproject.toml"
    document = tomllib.loads(path.read_text(encoding="utf-8"))
    declared: list[str] = list(document.get("project", {}).get("dependencies", []))
    for extra in document.get("project", {}).get("optional-dependencies", {}).values():
        declared.extend(extra)

    findings: list[str] = []
    for requirement in declared:
        # Strip everything after the first character that cannot be part of a name.
        name = requirement
        for separator in ("[", "<", ">", "=", "!", "~", ";", " "):
            name = name.split(separator, 1)[0]
        if _normalize(name.strip()) in FORBIDDEN_DISTRIBUTIONS:
            findings.append(f"backend/pyproject.toml declares a forbidden dependency: {name}")
    return findings


def check_lockfile(root: Path) -> list[str]:
    """Check 2: nothing forbidden appears in the resolved workspace lockfile."""
    path = root / "uv.lock"
    if not path.is_file():
        return [f"lockfile is missing: {path}"]
    document = tomllib.loads(path.read_text(encoding="utf-8"))
    findings: list[str] = []
    for package in document.get("package", []):
        name = _normalize(str(package.get("name", "")))
        if name in FORBIDDEN_DISTRIBUTIONS:
            findings.append(
                f"uv.lock resolves a forbidden distribution: {package.get('name')} "
                f"{package.get('version', '')}".rstrip()
            )
    return findings


def _forbidden_imports_in(source: str) -> set[str]:
    tree = ast.parse(source)
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".", 1)[0]
                if top in FORBIDDEN_MODULES:
                    found.add(top)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            top = node.module.split(".", 1)[0]
            if top in FORBIDDEN_MODULES:
                found.add(top)
    return found


def check_static_imports(root: Path) -> list[str]:
    """Check 3: no runtime module imports a forbidden top-level package."""
    source_root = root / "backend" / "src"
    findings: list[str] = []
    for path in sorted(source_root.rglob("*.py")):
        try:
            offenders = _forbidden_imports_in(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError) as exc:
            findings.append(f"could not parse {path.relative_to(root)}: {exc}")
            continue
        for name in sorted(offenders):
            findings.append(f"{path.relative_to(root)} imports forbidden module {name!r}")
    return findings


def check_runtime_import_graph(root: Path) -> list[str]:
    """Check 4: importing the app must not load a forbidden package.

    Runs in a subprocess so the result is not contaminated by anything this
    process (or pytest, or a plugin) has already imported.
    """
    program = (
        "import sys, json\n"
        f"import {RUNTIME_ENTRY_MODULE} as entry\n"
        "entry.create_app\n"
        f"forbidden = sorted({sorted(FORBIDDEN_MODULES)!r})\n"
        "loaded = sorted(name for name in forbidden if name in sys.modules)\n"
        "print(json.dumps(loaded))\n"
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        cwd=str(root),
        check=False,
    )
    if result.returncode != 0:
        return [f"could not import {RUNTIME_ENTRY_MODULE}: {result.stderr.strip()}"]
    try:
        loaded: list[str] = json.loads(result.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        return [f"could not read the import-graph probe output: {exc}"]
    if loaded:
        return [
            f"importing {RUNTIME_ENTRY_MODULE} loaded forbidden module(s): " + ", ".join(loaded)
        ]
    return []


def check_all(root: Path) -> list[str]:
    """Run every boundary check and return the combined findings."""
    return [
        *check_declared_dependencies(root),
        *check_lockfile(root),
        *check_static_imports(root),
        *check_runtime_import_graph(root),
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)

    findings = check_all(args.root.resolve())
    if findings:
        print("Runtime licence/dependency boundary violations:", file=sys.stderr)
        for finding in findings:
            print(f"  - {finding}", file=sys.stderr)
        return 1
    print(
        "Runtime boundary: OK. No AGPL or excluded distribution is declared, "
        "resolved, statically imported or loaded at runtime."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
