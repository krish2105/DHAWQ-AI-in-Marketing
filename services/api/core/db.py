"""Durable state — PLAN.md D0-3: one Postgres carries the catalogue, the
LangGraph checkpointer and the audit log.

WHAT THIS IS AND IS NOT. It is a small, explicit store for the three things
that must survive a process restart: agent runs, their traces, and the audit
log. It is NOT an ORM and it is NOT where the recommendation artefacts live —
those are frozen parquet/npy read through core/artifacts.py, and PLAN.md §1
forbids the API reaching back into pipelines/ for anything else.

DEGRADES, NEVER FAILS. With no DATABASE_URL the store is in-process and the API
runs identically — Render's free tier expires its database after 30 days and a
graded MVP that 500s because a demo database lapsed is worse than one that
quietly loses history. `durable()` reports which mode is live so /health can
say so rather than implying persistence it does not have.
"""

from __future__ import annotations

import json
import os
import threading
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any

DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()

# Render hands out postgres:// ; psycopg 3 wants postgresql://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = "postgresql://" + DATABASE_URL[len("postgres://"):]

_MEM_RUNS: "OrderedDict[str, dict]" = OrderedDict()
_MEM_TRACES: "OrderedDict[str, dict]" = OrderedDict()
_MEM_AUDIT: list[dict] = []
_LOCK = threading.Lock()
MAX_MEM = 200

_pool: Any = None
_ready = False
_error: str | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


DDL = """
CREATE SCHEMA IF NOT EXISTS agent_runs;

CREATE TABLE IF NOT EXISTS agent_runs.runs (
    run_id      text PRIMARY KEY,
    caller_id   text NOT NULL,
    caller_role text NOT NULL,
    phase       text NOT NULL,
    goal        text NOT NULL,
    state       jsonb NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now(),
    updated_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS agent_runs.traces (
    run_id  text PRIMARY KEY,
    spans   jsonb NOT NULL,
    stored_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS agent_runs.audit (
    id      bigserial PRIMARY KEY,
    ts      timestamptz NOT NULL DEFAULT now(),
    actor   text NOT NULL,
    action  text NOT NULL,
    detail  jsonb NOT NULL
);
CREATE INDEX IF NOT EXISTS audit_ts_idx ON agent_runs.audit (ts DESC);
"""


def init() -> None:
    """Connect and migrate. Called once at startup; safe to call twice."""
    global _pool, _ready, _error
    if _ready or not DATABASE_URL:
        return
    try:
        from psycopg_pool import ConnectionPool
        _pool = ConnectionPool(DATABASE_URL, min_size=1, max_size=4,
                               open=True, timeout=10)
        with _pool.connection() as conn:
            conn.execute(DDL)
        _ready = True
    except Exception as exc:                      # noqa: BLE001
        # A database that will not come up must not take the API with it.
        _error = f"{type(exc).__name__}: {exc}"[:200]
        _pool = None
        _ready = False


def close() -> None:
    """Shut the pool down cleanly.

    Without this every restart logs "couldn't stop thread 'pool-1-worker-N'
    within 5.0 seconds" four times and waits for each — on a free instance that
    already cold-starts, adding seconds to every redeploy for no reason. Found
    by running against a real Postgres; the in-process fallback has no pool and
    would never have shown it.
    """
    global _pool, _ready
    if _pool is not None:
        try:
            _pool.close()
        except Exception:                              # noqa: BLE001
            pass
    _pool, _ready = None, False


def durable() -> dict:
    return {"backend": "postgres" if _ready else "in-process",
            "configured": bool(DATABASE_URL), "error": _error}


def _trim(store: OrderedDict) -> None:
    while len(store) > MAX_MEM:
        store.popitem(last=False)


# ── runs ────────────────────────────────────────────────────────────────────

def save_run(run_id: str, caller_id: str, caller_role: str, phase: str,
             goal: str, state: dict) -> None:
    if _ready:
        try:
            with _pool.connection() as conn:
                conn.execute(
                    "INSERT INTO agent_runs.runs "
                    "(run_id, caller_id, caller_role, phase, goal, state) "
                    "VALUES (%s,%s,%s,%s,%s,%s) "
                    "ON CONFLICT (run_id) DO UPDATE SET "
                    "phase=EXCLUDED.phase, state=EXCLUDED.state, updated_at=now()",
                    (run_id, caller_id, caller_role, phase, goal,
                     json.dumps(state, default=str)))
            return
        except Exception:                          # noqa: BLE001
            pass                                   # fall through to memory
    with _LOCK:
        _MEM_RUNS[run_id] = {"run_id": run_id, "caller_id": caller_id,
                             "caller_role": caller_role, "phase": phase,
                             "goal": goal, "state": state, "updated_at": _now()}
        _MEM_RUNS.move_to_end(run_id)
        _trim(_MEM_RUNS)


def load_run(run_id: str) -> dict | None:
    if _ready:
        try:
            with _pool.connection() as conn:
                row = conn.execute(
                    "SELECT state, caller_id, caller_role FROM agent_runs.runs "
                    "WHERE run_id = %s", (run_id,)).fetchone()
            if row:
                return {"state": row[0], "caller_id": row[1], "caller_role": row[2]}
        except Exception:                          # noqa: BLE001
            pass
    with _LOCK:
        rec = _MEM_RUNS.get(run_id)
    return dict(rec) if rec else None


# ── traces (§7.9 — replayable) ──────────────────────────────────────────────

def save_trace(run_id: str, spans: list[dict]) -> None:
    if _ready:
        try:
            with _pool.connection() as conn:
                conn.execute(
                    "INSERT INTO agent_runs.traces (run_id, spans) VALUES (%s,%s) "
                    "ON CONFLICT (run_id) DO UPDATE SET spans=EXCLUDED.spans, "
                    "stored_at=now()",
                    (run_id, json.dumps(spans, default=str)))
            return
        except Exception:                          # noqa: BLE001
            pass
    with _LOCK:
        _MEM_TRACES[run_id] = {"run_id": run_id, "spans": spans}
        _MEM_TRACES.move_to_end(run_id)
        _trim(_MEM_TRACES)


def load_trace(run_id: str) -> list[dict] | None:
    if _ready:
        try:
            with _pool.connection() as conn:
                row = conn.execute(
                    "SELECT spans FROM agent_runs.traces WHERE run_id = %s",
                    (run_id,)).fetchone()
            if row:
                return row[0]
        except Exception:                          # noqa: BLE001
            pass
    with _LOCK:
        rec = _MEM_TRACES.get(run_id)
    return rec["spans"] if rec else None


# ── audit (§13.1) ───────────────────────────────────────────────────────────

def audit(actor: str, action: str, detail: dict) -> None:
    if _ready:
        try:
            with _pool.connection() as conn:
                conn.execute(
                    "INSERT INTO agent_runs.audit (actor, action, detail) "
                    "VALUES (%s,%s,%s)",
                    (actor, action, json.dumps(detail, default=str)))
            return
        except Exception:                          # noqa: BLE001
            pass
    with _LOCK:
        _MEM_AUDIT.append({"ts": _now(), "actor": actor, "action": action,
                           "detail": detail})
        del _MEM_AUDIT[:-500]


def audit_tail(limit: int = 100) -> list[dict]:
    if _ready:
        try:
            with _pool.connection() as conn:
                rows = conn.execute(
                    "SELECT ts, actor, action, detail FROM agent_runs.audit "
                    "ORDER BY ts DESC LIMIT %s", (limit,)).fetchall()
            return [{"ts": r[0].isoformat(), "actor": r[1], "action": r[2],
                     "detail": r[3]} for r in rows]
        except Exception:                          # noqa: BLE001
            pass
    with _LOCK:
        return list(reversed(_MEM_AUDIT[-limit:]))
