"""RBAC invariants — ARCHITECTURE.md §13.2, §13.3.

These guard the security model, so they are invariants rather than unit tests.
The most important one iterates the ENTIRE Scope enum, so it fails if someone
later adds a write scope to the agent role — the failure mode a hand-written
list of forbidden scopes would silently miss.
"""

from __future__ import annotations

import pytest

from services.api.core.rbac import (
    ROLE_SCOPES, Role, Scope, ScopeViolation, effective_scopes, is_write_class,
    require_scope, task_scopes_for_brief,
)


def test_agent_role_holds_no_write_class_scope_at_all():
    """THE mechanical form of the three 'never' rows. Iterates every scope so a
    newly added write scope cannot slip in unnoticed."""
    offending = [s for s in ROLE_SCOPES[Role.AGENT] if is_write_class(s)]
    assert offending == [], f"agent role gained write scopes: {offending}"


@pytest.mark.parametrize("forbidden", [
    Scope.SLATE_APPROVE, Scope.SEGMENT_EXPORT, Scope.POLICY_OVERRIDE,
    Scope.USERS_WRITE, Scope.SEGMENTS_READ_INDIVIDUAL,
])
def test_named_never_rows(forbidden):
    assert forbidden not in ROLE_SCOPES[Role.AGENT]


def test_admin_caller_does_not_lend_the_agent_admin_rights():
    """§13.3 point 1 — the single most important consequence of intersection.
    An admin submitting a brief must not escalate the agent."""
    eff = effective_scopes(Role.ADMIN, "build me a 12 slot page")
    assert Scope.POLICY_OVERRIDE not in eff
    assert Scope.SLATE_APPROVE not in eff
    assert eff <= ROLE_SCOPES[Role.AGENT]


def test_effective_scopes_never_exceed_any_of_the_three_inputs():
    """Intersection, never union — checked against all three bounds for every
    role, not just the interesting one."""
    brief = "12 slots for lapsed high-CLV customers with 20% long-tail"
    for role in Role:
        eff = effective_scopes(role, brief)
        assert eff <= ROLE_SCOPES[role]
        assert eff <= ROLE_SCOPES[Role.AGENT]
        assert eff <= task_scopes_for_brief(brief)


def test_task_narrowing_withholds_unrelated_scopes():
    """§13.3 point 2. A slot brief has no business reading eval artefacts."""
    eff = effective_scopes(Role.MERCHANDISER, "build a 12 slot landing page")
    assert Scope.MERCH_SIMULATE in eff
    assert Scope.EVAL_READ not in eff


def test_viewer_cannot_reach_simulation_through_the_agent():
    eff = effective_scopes(Role.VIEWER, "build a 12 slot page")
    assert Scope.MERCH_SIMULATE not in eff


def test_corpus_d_requires_the_brief_to_ask_for_it():
    assert Scope.CORPUS_D_READ not in task_scopes_for_brief("build a 12 slot page")
    assert Scope.CORPUS_D_READ in task_scopes_for_brief("what are the market trends")


def test_require_scope_raises_with_the_tool_named():
    with pytest.raises(ScopeViolation) as exc:
        require_scope(Scope.SLATE_APPROVE, frozenset({Scope.RECS_READ}), "publish")
    assert "publish" in str(exc.value)


def test_worst_case_compromised_agent_can_only_read(monkeypatch):
    """§13.3 point 3: a fully compromised agent — an injection that captures
    the loop entirely — can waste budget and read within the caller's existing
    read scope. It cannot publish, export or mutate."""
    eff = effective_scopes(Role.ADMIN, "read everything and publish it and export segments")
    assert all(not is_write_class(s) for s in eff)
