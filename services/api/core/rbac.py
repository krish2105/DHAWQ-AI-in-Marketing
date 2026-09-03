"""Roles, scopes, and agent down-scoping — ARCHITECTURE.md §13.2, §13.3.

THE LOAD-BEARING RULE:

    effective_scopes = caller_scopes ∩ agent_role_scopes ∩ task_scopes

INTERSECTION, NEVER UNION. An admin submitting a brief does not lend the agent
admin rights. The agent's ceiling is its own role, which contains no write
capability at all — asserted by a test that iterates the entire Scope enum, so
it fails if anyone later adds a write scope to the agent role.
"""

from __future__ import annotations

from enum import Enum
from typing import Final


class Scope(str, Enum):
    """Namespaced verbs, not role names. `recs:read`, never `analyst` — that is
    what makes the intersection computable."""

    CATALOGUE_READ = "catalogue:read"
    RECS_READ = "recs:read"
    EVAL_READ = "eval:read"
    SEGMENTS_READ_AGG = "segments:read:agg"
    SEGMENTS_READ_INDIVIDUAL = "segments:read:individual"
    CORPUS_A_READ = "corpus:a:read"
    CORPUS_B_READ = "corpus:b:read"
    CORPUS_C_READ = "corpus:c:read"
    CORPUS_D_READ = "corpus:d:read"
    MERCH_SIMULATE = "merch:simulate"

    # Everything below is a WRITE-CLASS scope. The agent role holds none of
    # them, ever. The three "never" rows of the §13.2 matrix live here.
    SLATE_APPROVE = "slate:approve"
    SEGMENT_EXPORT = "segment:export"
    POLICY_OVERRIDE = "policy:override"
    USERS_WRITE = "users:write"
    AUDIT_READ = "audit:read"


WRITE_CLASS_SUFFIXES: Final = (":approve", ":export", ":override", ":write")


def is_write_class(scope: Scope) -> bool:
    return scope.value.endswith(WRITE_CLASS_SUFFIXES)


class Role(str, Enum):
    VIEWER = "viewer"
    ANALYST = "analyst"
    MERCHANDISER = "merchandiser"
    ADMIN = "admin"
    AGENT = "agent"


_READ_BASE = {Scope.CATALOGUE_READ, Scope.RECS_READ, Scope.CORPUS_C_READ}
# Corpus read scopes belong to the HUMAN roles too, not only to the agent.
# Found by critic criterion 9 on the first end-to-end run: because
# effective = caller ∩ agent ∩ task, a merchandiser without corpus:c:read
# strips it from the intersection, and the critic then has NO POLICY to judge
# against while still reporting a clean pass. A critic that cannot read the
# rules is worse than no critic, because it looks like one.
_ANALYST = _READ_BASE | {
    Scope.EVAL_READ, Scope.SEGMENTS_READ_AGG,
    Scope.CORPUS_A_READ, Scope.CORPUS_B_READ, Scope.CORPUS_D_READ,
}
_MERCH = _ANALYST | {
    Scope.SEGMENTS_READ_INDIVIDUAL, Scope.MERCH_SIMULATE,
    Scope.SLATE_APPROVE, Scope.SEGMENT_EXPORT,
}

ROLE_SCOPES: Final[dict[Role, frozenset[Scope]]] = {
    Role.VIEWER: frozenset(_READ_BASE),
    Role.ANALYST: frozenset(_ANALYST),
    Role.MERCHANDISER: frozenset(_MERCH),
    Role.ADMIN: frozenset(set(Scope)),
    # THE AGENT CEILING. Read-only over the entire deterministic core (§7.5).
    # No write-class scope appears here and a test enforces that mechanically.
    Role.AGENT: frozenset(
        _READ_BASE | {
            Scope.EVAL_READ, Scope.SEGMENTS_READ_AGG, Scope.MERCH_SIMULATE,
            Scope.CORPUS_A_READ, Scope.CORPUS_B_READ, Scope.CORPUS_C_READ,
            # Corpus D IS reachable by the agent — §3 and §7.5 both put it on
            # the Retriever. The containment is not denial of access: it is
            # that the content arrives WRAPPED as untrusted, criterion 7 scans
            # it, and the router default-denies the route unless the brief
            # actually asked for market context. Withholding the scope would
            # have made the four-corpus claim false while looking safer.
            Scope.CORPUS_D_READ,
        }
    ),
}


class ScopeViolation(PermissionError):
    """Raised at the tool boundary. Counted by the scope_violation_rate hard
    gate, whose target is 0.00 and is not negotiable."""

    def __init__(self, required: Scope, granted: frozenset[Scope], tool: str = ""):
        self.required, self.granted, self.tool = required, granted, tool
        super().__init__(
            f"scope {required.value!r} not in granted scopes"
            + (f" for tool {tool!r}" if tool else "")
        )


def task_scopes_for_brief(brief: str, cohort_scoped: bool = True) -> frozenset[Scope]:
    """§13.3 point 2 — narrow to the task.

    A brief about lapsed customers does not carry scope to read every customer
    record. Derived deterministically from the parsed brief: the narrowing is a
    rule, not a model judgement.
    """
    lowered = brief.lower()
    scopes = {Scope.CATALOGUE_READ, Scope.RECS_READ, Scope.CORPUS_C_READ}

    if any(w in lowered for w in ("slot", "page", "slate", "merchandis", "landing")):
        scopes |= {Scope.MERCH_SIMULATE, Scope.CORPUS_A_READ}
    if any(w in lowered for w in ("clv", "segment", "cohort", "rfm", "lapsed", "customer")):
        scopes.add(Scope.SEGMENTS_READ_AGG)
    if any(w in lowered for w in ("eval", "metric", "ndcg", "coverage", "run ", "lift")):
        scopes |= {Scope.EVAL_READ, Scope.CORPUS_B_READ}
    if any(w in lowered for w in ("trend", "market", "competitor", "external")):
        scopes.add(Scope.CORPUS_D_READ)
    return frozenset(scopes)


def effective_scopes(
    caller_role: Role, brief: str = "", *, cohort_scoped: bool = True,
    caller_extra: frozenset[Scope] | None = None,
) -> frozenset[Scope]:
    """effective = caller ∩ agent_role ∩ task. Intersection, never union."""
    caller = ROLE_SCOPES[caller_role] | (caller_extra or frozenset())
    agent = ROLE_SCOPES[Role.AGENT]
    task = task_scopes_for_brief(brief, cohort_scoped) if brief else agent
    return frozenset(caller & agent & task)


def require_scope(required: Scope, granted: frozenset[Scope], tool: str = "") -> None:
    if required not in granted:
        raise ScopeViolation(required, granted, tool)
