"""The §4 permission boundary, as a test.

"Read the diagram as a permission boundary. The orchestration layer may CALL
the deterministic core; it may never WRITE into it."

PLAN.md §1 extends that to the build layer: services/api/ never imports
pipelines/. Artefacts cross the line as frozen files plus a manifest, never as
a live call. CI greps for it too; this makes it fail locally as well, because a
boundary you only discover in CI is a boundary you have already crossed.

It has been crossed once: the /space/manifest endpoint imported
pipelines.common.read_manifest to save writing four lines. That is exactly how
these boundaries erode — never by a decision, always by a shortcut.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
API = REPO / "services" / "api"
PIPELINES = REPO / "pipelines"


def _imports(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except SyntaxError:
        return set()
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
    return out


def _py(root: Path) -> list[Path]:
    return [p for p in root.rglob("*.py") if "__pycache__" not in p.parts]


def test_api_never_imports_pipelines():
    """The build layer is one-way. Enforced by AST, not by grep, so an import
    hidden inside a function body is caught too — which is where the real one
    was."""
    offenders = {
        str(p.relative_to(REPO)): sorted(m for m in _imports(p)
                                         if m == "pipelines" or m.startswith("pipelines."))
        for p in _py(API)
    }
    offenders = {k: v for k, v in offenders.items() if v}
    assert offenders == {}, (
        f"services/api imports pipelines — the §4 boundary is broken: {offenders}"
    )


def test_pipelines_may_import_the_api_but_only_for_read_only_artefacts():
    """The other direction is permitted: a build script may read the catalogue
    through core.artifacts. It must not reach into the agent, the tools or the
    critic — a pipeline that could drive the agent would let build-time code
    write into a run."""
    forbidden = ("services.api.agent", "services.api.main")
    offenders = {
        str(p.relative_to(REPO)): sorted(m for m in _imports(p)
                                         if m.startswith(forbidden))
        for p in _py(PIPELINES)
    }
    offenders = {k: v for k, v in offenders.items() if v}
    assert offenders == {}, f"pipelines reaching into the agent layer: {offenders}"


def test_artifacts_module_is_the_only_door_and_it_is_read_only():
    """core/artifacts.py is the sanctioned crossing. If a write path ever
    appears in it, the §4 diagram has stopped being true."""
    src = (API / "core" / "artifacts.py").read_text()
    for banned in ("write_parquet", "to_parquet", ".write_text(", "open(", "shutil"):
        assert banned not in src, (
            f"core/artifacts.py contains {banned!r} — the read-only door now writes"
        )


def test_agent_layer_does_not_import_the_web_or_the_service_entrypoint():
    """The agent is callable from the API, never the reverse."""
    offenders = {
        str(p.relative_to(REPO)): sorted(m for m in _imports(p) if m == "services.api.main")
        for p in _py(API / "agent")
    }
    offenders = {k: v for k, v in offenders.items() if v}
    assert offenders == {}, f"agent imports the service entrypoint: {offenders}"
