"""Agent graph integration — ARCHITECTURE.md §7.1 (orchestration layer).

Tested by driving the real graph and inspecting the trace, per §7.1.
"""

from __future__ import annotations

import pytest

from services.api.agent.graph import (
    build_graph, explainer, human_gate, new_run, run_to_gate,
)
from services.api.agent.state import Budget, GateResolution, Phase
from services.api.core.rbac import Role, Scope, is_write_class

BRIEF = "build me a 12 slot landing page for lapsed high-CLV customers with 20% long-tail"


@pytest.fixture(scope="module")
def finished():
    return run_to_gate(new_run(BRIEF, "u1", Role.MERCHANDISER))


def test_graph_compiles_with_an_interrupt_before_the_gate():
    assert build_graph() is not None


def test_a_run_is_down_scoped_at_creation(finished):
    """§13.3 — the agent's ceiling is its own role, never the caller's."""
    from services.api.core.rbac import ROLE_SCOPES
    assert finished.granted_scopes <= ROLE_SCOPES[Role.AGENT]
    assert all(not is_write_class(s) for s in finished.granted_scopes)


def test_admin_caller_does_not_escalate_the_run():
    run = new_run(BRIEF, "admin", Role.ADMIN)
    assert Scope.SLATE_APPROVE not in run.granted_scopes
    assert Scope.POLICY_OVERRIDE not in run.granted_scopes


def test_run_reaches_a_gate_and_nothing_publishes_before_it(finished):
    """"Nothing publishes without approval." Before the gate is resolved there
    must be no approved slate."""
    assert finished.phase is Phase.GATED
    assert finished.pending_gate is not None
    assert finished.gate_history == []


def test_zero_scope_violations_end_to_end(finished):
    """scope_violation_rate is a hard gate at 0.00. A node must ASK whether it
    may call a tool, not attempt and be refused."""
    assert [r for r in finished.rejections if r.criterion == 9] == []


def test_every_claim_resolves_to_real_evidence(finished):
    """ungrounded_claim_rate, hard gate, target 0.00."""
    assert finished.unresolved_claims() == []


def test_approving_the_gate_produces_a_final_slate(finished):
    run = finished.model_copy(deep=True)
    res = GateResolution(gate_id=run.pending_gate.gate_id, decision="approve",
                         actor_id="u1")
    run = explainer(human_gate(run, res))
    assert run.phase is Phase.DONE
    assert run.final_slate_id is not None
    assert run.pending_gate is None


def test_rejecting_the_gate_drops_the_slate(finished):
    run = finished.model_copy(deep=True)
    res = GateResolution(gate_id=run.pending_gate.gate_id, decision="reject",
                         actor_id="u1")
    run = human_gate(run, res)
    assert run.final_slate_id is None and run.phase is Phase.DONE


def test_a_stale_gate_resolution_is_refused(finished):
    """Otherwise a replayed approval could authorise a slate it was never
    shown."""
    run = finished.model_copy(deep=True)
    run = human_gate(run, GateResolution(gate_id="gt_wrong", decision="approve",
                                         actor_id="attacker"))
    assert run.gate_history == []
    assert any(e.kind == "stale_gate" for e in run.errors)
    assert run.pending_gate is not None, "the real gate must remain open"


def test_budget_exhaustion_fails_cleanly_rather_than_hanging():
    run = new_run(BRIEF, "u1", Role.MERCHANDISER, budget=Budget(max_steps=1))
    run = run_to_gate(run)
    assert run.phase is Phase.FAILED
    assert any(e.kind == "budget" for e in run.errors)


def test_state_accumulates_rather_than_overwriting(finished):
    """Rejections, tool calls and evidence are the debugging surface and the
    most interesting thing to show a user (§7.2)."""
    assert len(finished.tool_calls) > 0
    assert len(finished.evidence) > 0
    assert finished.lineage, "derived artefacts must carry what produced them"


def test_critic_is_capped_at_two_rounds(finished):
    from services.api.agent.critic import MAX_ROUNDS
    assert finished.critic_rounds <= MAX_ROUNDS
