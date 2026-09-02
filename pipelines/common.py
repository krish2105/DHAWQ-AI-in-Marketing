"""Shared pipeline utilities.

Everything under pipelines/ is BUILD-TIME ONLY and is never imported by
services/api/. Artefacts cross that boundary as frozen files plus a manifest,
never as a live call. See PLAN.md §1.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
MANIFEST_DIR = REPO_ROOT / "pipelines" / "manifests"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


@contextmanager
def step(label: str) -> Iterator[None]:
    """Timed, labelled stage. Pipelines are long; silent ones are unusable."""
    print(f"  ▸ {label} ... ", end="", flush=True)
    t0 = time.perf_counter()
    try:
        yield
    except Exception:
        print(f"FAILED after {time.perf_counter() - t0:.1f}s")
        raise
    print(f"{time.perf_counter() - t0:.1f}s")


class AssertionFailure(RuntimeError):
    """A named build assertion failed.

    Distinct from a bare AssertionError so a rule violation is never confused
    with an incidental bug, and so `python -O` can never strip it.
    """


def require(condition: bool, code: str, message: str) -> None:
    """Assert a named build invariant (A1..A7 in PLAN.md §7).

    Deliberately not `assert` — these must survive optimisation flags. A silent
    leak assertion is worse than no leak assertion.
    """
    if not condition:
        raise AssertionFailure(f"[{code}] {message}")


def _json_default(o: Any) -> Any:
    if isinstance(o, (date, datetime)):
        return o.isoformat()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, set):
        return sorted(o)
    raise TypeError(f"not JSON-serialisable: {type(o)}")


def write_manifest(name: str, payload: dict[str, Any]) -> Path:
    """Write a pipeline manifest. Every pipeline writes exactly one."""
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    path = MANIFEST_DIR / f"{name}.json"
    full = {
        "pipeline": name,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_rev": git_rev(),
        "python": sys.version.split()[0],
        **payload,
    }
    path.write_text(json.dumps(full, indent=2, default=_json_default) + "\n")
    return path


def read_manifest(name: str) -> dict[str, Any]:
    return json.loads((MANIFEST_DIR / f"{name}.json").read_text())
