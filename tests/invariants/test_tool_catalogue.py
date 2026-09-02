"""Tool catalogue invariants — ARCHITECTURE.md §7.5.

"There is no tool that writes to the catalogue, the model registry, or the
evaluation artefacts. The agent is read-only over the entire deterministic
core." These tests are what make that a property rather than a promise.
"""

from __future__ import annotations

import pytest

from services.api.agent.tools import catalogue, invoke
from services.api.core.rbac import ROLE_SCOPES, Role, Scope, is_write_class


def test_no_tool_writes():
    assert [n for n, s in catalogue().items() if s.writes] == []


def test_no_tool_requires_a_write_class_scope():
    """The mechanical version: even if a write tool were added, it could not be
    reached, because no tool may require an approve/export/override scope."""
    offending = {n: s.scope.value for n, s in catalogue().items()
                 if is_write_class(s.scope)}
    assert offending == {}, f"write-class tools in the catalogue: {offending}"


def test_every_tool_scope_is_within_the_agent_ceiling():
    agent = ROLE_SCOPES[Role.AGENT]
    outside = {n: s.scope.value for n, s in catalogue().items() if s.scope not in agent}
    assert outside == {}, f"tools the agent role can never call: {outside}"


def test_every_tool_is_typed_and_capped():
    for name, spec in catalogue().items():
        assert spec.input_model is not None, name
        assert spec.max_calls > 0, name
        assert spec.description, name


def test_scope_violation_refuses_before_the_tool_body_runs():
    """A compromised agent emitting a forbidden call gets a refusal, not an
    execution. scope_violation_rate is a hard gate at 0.00."""
    res = invoke("optimise_slots", {"candidate_ids": [], "k": 12}, frozenset())
    assert res.ok is False
    assert "scope" in (res.error or "").lower()
    assert res.call.ok is False


def test_registry_is_not_dynamically_extensible_at_runtime():
    """§13.4 LLM07 — a tool that can be added at runtime is a tool an injection
    can add."""
    from services.api.agent import tools
    with pytest.raises(ValueError):
        tools.register(list(catalogue().values())[0])


def test_recommend_has_no_individual_customer_parameter():
    """POL-SEG-02 and the §13.2 'agent may not read individual customer
    records' row: the capability is absent from the type, so there is no code
    path to misuse."""
    fields = set(catalogue()["recommend"].input_model.model_fields)
    assert "customer_id" not in fields
    assert "cohort_spec" in fields
