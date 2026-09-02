"""State model tests — ARCHITECTURE.md §7.2."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from services.api.agent.state import (
    Budget, BudgetExhausted, Claim, Evidence, MerchandisingRun, Rejection, ToolCall,
)
from services.api.core.rbac import Role, Scope


# ── the load-bearing validator ───────────────────────────────────────────────

def test_a_claim_without_evidence_is_not_a_claim():
    """Fails at the TYPE boundary, not in a prompt. The prompt is a request;
    this is a rule."""
    with pytest.raises(ValidationError, match="not a claim"):
        Claim(text="revenue is up 12%", evidence_ids=[])


def test_a_grounded_claim_is_accepted():
    assert Claim(text="x", evidence_ids=["ev_1"]).evidence_ids == ["ev_1"]


# ── content addressing ───────────────────────────────────────────────────────

def test_identical_evidence_deduplicates_by_construction():
    a = Evidence.create("C", "POL-LT-01", "min 20% long tail")
    b = Evidence.create("C", "POL-LT-01", "min 20% long tail")
    assert a.evidence_id == b.evidence_id


def test_different_content_gets_a_different_id():
    a = Evidence.create("C", "POL-LT-01", "min 20%")
    b = Evidence.create("C", "POL-LT-01", "min 25%")
    assert a.evidence_id != b.evidence_id


# ── budget ───────────────────────────────────────────────────────────────────

def test_budget_is_enforced_not_advisory():
    b = Budget(max_steps=2)
    b = b.charge(steps=1).charge(steps=1)
    with pytest.raises(BudgetExhausted):
        b.charge(steps=1)


def test_budget_preflight_does_not_consume():
    b = Budget(max_steps=5)
    assert b.would_exceed(steps=1) is None
    assert b.steps_used == 0, "would_exceed must not mutate"


def test_budget_is_frozen_so_state_merging_cannot_corrupt_it():
    b = Budget()
    with pytest.raises(ValidationError):
        b.max_steps = 999


def test_per_tool_call_cap_is_enforced():
    """§7.8 tool thrash: same tool, tiny arg variations."""
    b = Budget(max_calls_per_tool=2)
    b = b.charge(tool="recommend").charge(tool="recommend")
    with pytest.raises(BudgetExhausted) as exc:
        b.charge(tool="recommend")
    assert exc.value.breach.kind == "tool_calls"


# ── run-level views ──────────────────────────────────────────────────────────

def _run(**kw) -> MerchandisingRun:
    return MerchandisingRun(run_id="r1", goal="g", caller_id="u1",
                            caller_role=Role.MERCHANDISER,
                            granted_scopes=frozenset({Scope.RECS_READ}), **kw)


def test_unresolved_claims_feed_the_ungrounded_gate():
    ev = Evidence.create("C", "s", "c")
    run = _run(evidence=[ev],
               claims=[Claim(text="ok", evidence_ids=[ev.evidence_id]),
                       Claim(text="bad", evidence_ids=["ev_missing"])])
    assert len(run.unresolved_claims()) == 1
    assert run.evidence_coverage() == 0.5


def test_evidence_coverage_of_a_claimless_run_is_one_not_zero():
    """No claims means nothing ungrounded. Returning 0.0 here would make every
    empty run trip the coverage gate."""
    assert _run().evidence_coverage() == 1.0


def test_tool_call_dedupe_key_ignores_ordering_but_not_values():
    a = ToolCall(call_id="1", tool="recommend", args={"k": 10, "m": "x"},
                 scope_required=Scope.RECS_READ)
    b = ToolCall(call_id="2", tool="recommend", args={"m": "x", "k": 10},
                 scope_required=Scope.RECS_READ)
    c = ToolCall(call_id="3", tool="recommend", args={"k": 11, "m": "x"},
                 scope_required=Scope.RECS_READ)
    assert a.dedupe_key == b.dedupe_key
    assert a.dedupe_key != c.dedupe_key


def test_rejections_are_frozen_so_the_audit_trail_cannot_be_edited():
    r = Rejection(slate_id="s1", criterion=3, reason="quota", evaluated_by="code")
    with pytest.raises(ValidationError):
        r.reason = "actually it was fine"
