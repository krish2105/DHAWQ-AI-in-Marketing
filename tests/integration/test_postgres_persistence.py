"""Persistence against a REAL Postgres, not the in-process fallback.

WHY THIS EXISTS AS A SEPARATE FILE. Every other test runs on the fallback,
which has no pool, no SQL, no migration and no restart. It therefore cannot see
the three defects that were actually there:

  1. psycopg_pool was missing from requirements.txt. db.init() imports it,
     raises ImportError, catches its own exception and degrades — so
     DATABASE_URL would have been set, /health would have said "in-process",
     and nobody would have known why.
  2. The pool was never closed, so every restart waited 5s per worker.
  3. Runs were saved at SUBMIT only. A run reloaded after a restart claimed
     phase="planning" when it had gated long before.

None of the three is visible without a server. Skips cleanly when there isn't
one, and CI provides one.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="no DATABASE_URL; run `docker compose up -d postgres` to exercise this")


@pytest.fixture(scope="module")
def store():
    from services.api.core import db
    db.init()
    if db.durable()["backend"] != "postgres":
        pytest.fail(f"DATABASE_URL is set but the store degraded: {db.durable()}")
    yield db
    db.close()


def test_migration_is_idempotent(store):
    store.init()
    store.init()
    assert store.durable()["backend"] == "postgres"


def test_a_run_round_trips_and_upserts(store):
    store.save_run("t_run", "u", "merchandiser", "GATED", "goal", {"phase": "GATED"})
    assert store.load_run("t_run")["state"]["phase"] == "GATED"
    store.save_run("t_run", "u", "merchandiser", "DONE", "goal", {"phase": "DONE"})
    assert store.load_run("t_run")["state"]["phase"] == "DONE"


def test_a_trace_keeps_its_nesting_through_storage(store):
    store.save_trace("t_run", [
        {"span_id": "a", "parent_id": None, "name": "agent.run"},
        {"span_id": "b", "parent_id": "a", "name": "supervisor"},
    ])
    spans = store.load_trace("t_run")
    assert {s["span_id"] for s in spans} == {"a", "b"}
    assert next(s for s in spans if s["span_id"] == "b")["parent_id"] == "a"


def test_audit_rows_persist_and_come_back_newest_first(store):
    store.audit("u", "gate.resolved", {"run_id": "t_run"})
    rows = store.audit_tail(10)
    assert any(r["action"] == "gate.resolved" for r in rows)


def test_a_finished_run_survives_a_restart_with_its_trace(store):
    """The end-to-end claim, exercised the only way that proves it: complete a
    run, throw away every in-process cache, and read it back."""
    from fastapi.testclient import TestClient

    import services.api.main as M
    from services.api.agent import trace as tr
    from services.api.core.security import DEMO_PASSWORD

    # INSECURE_COOKIES is read at import time, so setting the env var here is
    # too late — the cookie would go out secure=True over the test client's
    # http and every authenticated request would look anonymous. Patch the
    # module attribute instead.
    M.INSECURE_COOKIES = True
    with TestClient(M.app) as c:
        assert c.get("/health").json()["persistence"]["backend"] == "postgres"
        c.post("/auth/login", json={"email": "merchandiser@dhawq.demo",
                                    "password": DEMO_PASSWORD})
        r = c.post("/agent/runs", json={
            "brief": "Build a 12-slot homepage slate for lapsed customers "
                     "respecting the long-tail quota."})
        run_id = r.json()["run_id"]
        c.get(f"/agent/runs/{run_id}/events")

        M._RUNS.clear()
        M.AUDIT.clear()
        tr._TRACES.clear()

        state = c.get(f"/agent/runs/{run_id}").json()
        assert state["phase"] != "planning", (
            "stored state is stale — the run was saved at submit and never again")
        assert state["candidate_slates"], "slate did not survive"

        t = c.get(f"/agent/runs/{run_id}/trace").json()
        assert t["source"] == "stored"
        assert t["spans"] and t["spans"][0]["children"]
        events = [e for sp in t["spans"][0]["children"]
                  for e in sp.get("events", []) if e["name"] == "reasoning"]
        assert events, "reasoning events did not survive storage"
