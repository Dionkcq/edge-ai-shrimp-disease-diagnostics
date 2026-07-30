"""The runtime must not reach the AGPL training toolchain.

`scripts/check_no_agpl_in_runtime.py` is the CI gate; this file makes the same
assertions part of the test suite, so the boundary breaks in a developer's terminal
rather than only in a pull request.

The four checks are independent on purpose — see that script's docstring. The one
worth understanding is the last: a static import scan cannot see a lazy import
performed inside a third-party package at call time, so the import graph of a real
process is observed as well.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - import bootstrap
    sys.path.insert(0, str(REPO_ROOT))

from scripts.check_no_agpl_in_runtime import (  # noqa: E402
    FORBIDDEN_DISTRIBUTIONS,
    FORBIDDEN_MODULES,
    check_declared_dependencies,
    check_lockfile,
    check_runtime_import_graph,
    check_static_imports,
)


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
    licence = REPO_ROOT / "pipeline" / "LICENSE.AGPL"
    assert licence.is_file(), "pipeline/ is AGPL and must carry the licence text"
    text = licence.read_text(encoding="utf-8")
    assert "GNU AFFERO GENERAL PUBLIC LICENSE" in text
    # Section 13 is what distinguishes the AGPL from the GPL; a GPL text pasted here
    # by mistake would not carry it.
    assert "Remote Network Interaction" in text


def test_the_pipeline_package_does_not_import_the_backend() -> None:
    """The boundary runs both ways: the trainer must not depend on the service."""
    offenders: list[str] = []
    for path in sorted((REPO_ROOT / "pipeline" / "src").rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if "shrimp_screening" in source:
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, f"pipeline must not reference the backend: {offenders}"


def test_licensing_md_documents_the_boundary() -> None:
    text = (REPO_ROOT / "LICENSING.md").read_text(encoding="utf-8")
    for expected in ("AGPL-3.0-or-later", "pipeline/", "backend/", "CC BY 4.0"):
        assert expected in text, f"LICENSING.md must explain {expected}"
