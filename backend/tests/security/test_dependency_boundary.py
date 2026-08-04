"""The runtime must not reach the AGPL training toolchain.

`model/training/` and `model/pipeline/` use Ultralytics/PyTorch, which are AGPL-3.0, so the
served application must never resolve or import them. The checks live in
`tests.support.agpl_boundary`; this file runs them, so the boundary breaks in a
developer's terminal rather than silently.

The four checks are independent on purpose — see that module's docstring. The one
worth understanding is the last: a static import scan cannot see a lazy import
performed inside a third-party package at call time, so the import graph of a real
process is observed as well.
"""

from __future__ import annotations

import ast
from pathlib import Path

from tests.support.agpl_boundary import (
    FORBIDDEN_DISTRIBUTIONS,
    FORBIDDEN_MODULES,
    check_declared_dependencies,
    check_lockfile,
    check_runtime_import_graph,
    check_static_imports,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_backend_declares_no_forbidden_dependency() -> None:
    assert check_declared_dependencies(REPO_ROOT) == []


def test_the_lockfile_resolves_no_forbidden_distribution() -> None:
    """Catches a *transitive* pull-in, which reading pyproject.toml cannot."""
    assert check_lockfile(REPO_ROOT) == []


def test_no_runtime_module_imports_torch_ultralytics_or_cv2() -> None:
    assert check_static_imports(REPO_ROOT) == []


def test_no_runtime_module_imports_the_pipeline_package() -> None:
    """The data/training tree is AGPL and must stay unreachable from the service."""
    findings = check_static_imports(REPO_ROOT)
    assert not [f for f in findings if "shrimp_pipeline" in f]


def test_importing_the_app_does_not_load_a_forbidden_module() -> None:
    """Observed in a fresh subprocess, so pytest's own imports cannot mask it."""
    assert check_runtime_import_graph(REPO_ROOT) == []


def test_the_forbidden_lists_still_name_the_things_that_matter() -> None:
    """A guard against the lists being quietly emptied to make the gate pass."""
    assert {"ultralytics", "torch"} <= FORBIDDEN_DISTRIBUTIONS
    assert {"ultralytics", "torch", "cv2", "shrimp_pipeline"} <= FORBIDDEN_MODULES


def test_the_pipeline_tree_carries_its_own_agpl_licence() -> None:
    licence = REPO_ROOT / "model" / "pipeline" / "LICENSE.AGPL"
    assert licence.is_file(), "model/pipeline is AGPL and must carry the licence text"
    text = licence.read_text(encoding="utf-8")
    assert "GNU AFFERO GENERAL PUBLIC LICENSE" in text
    # Section 13 is what distinguishes the AGPL from the GPL; a GPL text pasted here
    # by mistake would not carry it.
    assert "Remote Network Interaction" in text


def test_the_pipeline_package_does_not_import_the_backend() -> None:
    """The boundary runs both ways: the trainer must not depend on the service."""
    offenders: list[str] = []
    for path in sorted((REPO_ROOT / "model" / "pipeline" / "src").rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if "shrimp_screening" in source:
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, f"pipeline must not reference the backend: {offenders}"


def test_the_domain_does_not_import_the_server() -> None:
    """Layering: `shrimp_server` may import the domain, never the reverse.

    This is what keeps the domain usable without an HTTP stack -- a script, a
    notebook or a future CLI can call `decode_image` or `decide` without dragging
    in FastAPI -- and it is easy to undo by reflex, since the routes and the domain
    sit in the same distribution.

    The import graph is parsed rather than grepped: several domain modules name
    `shrimp_server` in a docstring to point the reader at the HTTP half, and a
    substring check would call that a violation.
    """
    offenders: list[str] = []
    for path in sorted((REPO_ROOT / "backend" / "src" / "shrimp_screening").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                imported = [node.module]
            else:
                continue
            if any(name.split(".", 1)[0] == "shrimp_server" for name in imported):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    assert not offenders, f"the domain must not import the server: {offenders}"
