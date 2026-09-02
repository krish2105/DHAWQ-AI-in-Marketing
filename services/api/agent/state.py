"""Agent state — a Pydantic model, never a dict (ARCHITECTURE.md §7.2).

"Typos in dict keys are the most common silent agent bug, and the most
expensive to find. Make invalid states unrepresentable rather than validating
them in a node."

The `must_be_grounded` validator is load-bearing: a claim without evidence
fails at the TYPE BOUNDARY, not in a prompt. The prompt is a request; the
validator is a rule.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from services.api.core.rbac import Role, Scope


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ─────────────────────────────────────────────────────────────────────────────
# Budget — in the initial state model, never retrofitted (§7.2)
# ─────────────────────────────────────────────────────────────────────────────

class BudgetBreach(BaseModel):
    kind: Literal["steps", "tokens", "wall_clock", "tool_calls", "critic_rounds"]
    limit: float
    attempted: float


class BudgetExhausted(RuntimeError):
    def __init__(self, breach: BudgetBreach):
        self.breach = breach
        super().__init__(f"budget exhausted: {breach.kind} "
                         f"(limit {breach.limit}, attempted {breach.attempted})")


class Budget(BaseModel):
    """Limits and consumption in ONE object, so a node can ask "can I afford
    this?" without reaching for a second field.

    Frozen, and `charge()` returns a new instance — budget accounting then
    survives LangGraph's state merging without a mutable-shared-object bug.
    """

    model_config = ConfigDict(frozen=True)

    max_steps: int = 24
    max_tokens: int = 250_000
    max_wall_clock_s: float = 120.0
    max_calls_per_tool: int = 4
    max_critic_rounds: int = 2
    max_retrieval_fanouts: int = 2

    steps_used: int = 0
    tokens_used: int = 0
    critic_rounds_used: int = 0
    tool_call_counts: dict[str, int] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=_utcnow)

    def elapsed_s(self) -> float:
        return (_utcnow() - self.started_at).total_seconds()

    def would_exceed(self, *, steps: int = 0, tokens: int = 0,
                     tool: str | None = None) -> BudgetBreach | None:
        """Pre-flight check. Nodes call this BEFORE an expensive action, not
        after — a budget discovered post-hoc has already been spent."""
        if self.steps_used + steps > self.max_steps:
            return BudgetBreach(kind="steps", limit=self.max_steps,
                                attempted=self.steps_used + steps)
        if self.tokens_used + tokens > self.max_tokens:
            return BudgetBreach(kind="tokens", limit=self.max_tokens,
                                attempted=self.tokens_used + tokens)
        if (el := self.elapsed_s()) > self.max_wall_clock_s:
            return BudgetBreach(kind="wall_clock", limit=self.max_wall_clock_s,
                                attempted=el)
        if tool is not None:
            n = self.tool_call_counts.get(tool, 0) + 1
            if n > self.max_calls_per_tool:
                return BudgetBreach(kind="tool_calls", limit=self.max_calls_per_tool,
                                    attempted=n)
        return None

    def charge(self, *, steps: int = 0, tokens: int = 0, tool: str | None = None,
               critic_round: bool = False) -> "Budget":
        if (breach := self.would_exceed(steps=steps, tokens=tokens, tool=tool)):
            raise BudgetExhausted(breach)
        counts = dict(self.tool_call_counts)
        if tool:
            counts[tool] = counts.get(tool, 0) + 1
        return self.model_copy(update={
            "steps_used": self.steps_used + steps,
            "tokens_used": self.tokens_used + tokens,
            "critic_rounds_used": self.critic_rounds_used + int(critic_round),
            "tool_call_counts": counts,
        })

    def remaining(self) -> dict[str, float]:
        return {
            "steps": self.max_steps - self.steps_used,
            "tokens": self.max_tokens - self.tokens_used,
            "wall_clock_s": max(0.0, self.max_wall_clock_s - self.elapsed_s()),
            "critic_rounds": self.max_critic_rounds - self.critic_rounds_used,
        }


# ─────────────────────────────────────────────────────────────────────────────
# The evidence spine
# ─────────────────────────────────────────────────────────────────────────────

class Finding(BaseModel):
    """A detected injection attempt or other anomaly in retrieved content.
    Counted, not merely defended against (§8.5)."""
    kind: Literal["injection", "contradiction", "malformed", "policy_conflict"]
    detail: str
    evidence_id: str | None = None
    pattern: str | None = None
    detected_at: datetime = Field(default_factory=_utcnow)


class Evidence(BaseModel):
    model_config = ConfigDict(frozen=True)

    evidence_id: str
    corpus: Literal["A", "B", "C", "D"]
    source_ref: str
    content: str
    trust: Literal["trusted", "untrusted"] = "trusted"
    retrieved_at: datetime = Field(default_factory=_utcnow)
    produced_by: list[str] = Field(default_factory=list)
    injection_findings: list[Finding] = Field(default_factory=list)

    @staticmethod
    def make_id(corpus: str, source_ref: str, content: str) -> str:
        """Content-addressed, not a counter. The same policy paragraph
        retrieved twice deduplicates by construction, and citation validity is
        checkable without a lookup table."""
        h = hashlib.sha256(f"{corpus}|{source_ref}|{content}".encode()).hexdigest()
        return f"ev_{h[:16]}"

    @classmethod
    def create(cls, corpus, source_ref, content, **kw) -> "Evidence":
        return cls(evidence_id=cls.make_id(corpus, source_ref, content),
                   corpus=corpus, source_ref=source_ref, content=content, **kw)


class Claim(BaseModel):
    """A claim without evidence is not a claim."""

    claim_id: str = ""
    text: str
    evidence_ids: list[str]
    kind: Literal["factual", "projected", "policy"] = "factual"

    @field_validator("evidence_ids")
    @classmethod
    def must_be_grounded(cls, v: list[str]) -> list[str]:
        # THE validator. Worth more than any amount of prompt engineering
        # asking a model to cite its sources: the prompt is a request, this is
        # a rule, and it fires at the type boundary before a node can act on it.
        if not v:
            raise ValueError("A claim without evidence is not a claim.")
        return v

    @model_validator(mode="after")
    def _assign_id(self) -> "Claim":
        if not self.claim_id:
            h = hashlib.sha256(self.text.encode()).hexdigest()[:12]
            object.__setattr__(self, "claim_id", f"cl_{h}")
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Work products
# ─────────────────────────────────────────────────────────────────────────────

class SlotAssignment(BaseModel):
    model_config = ConfigDict(frozen=True)
    position: int = Field(ge=1)
    article_id: str
    score: float
    is_long_tail: bool = False


class Slate(BaseModel):
    slate_id: str
    slots: list[SlotAssignment]
    k_requested: int
    cohort_spec: dict[str, Any] = Field(default_factory=dict)
    optimiser_report: dict[str, Any] = Field(default_factory=dict)
    produced_by: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)

    @property
    def article_ids(self) -> list[str]:
        return [s.article_id for s in sorted(self.slots, key=lambda s: s.position)]

    @property
    def long_tail_share(self) -> float:
        return (sum(1 for s in self.slots if s.is_long_tail) / len(self.slots)
                if self.slots else 0.0)


class Rejection(BaseModel):
    """Persisted AND surfaced in the UI. The rejection panel is a first-class
    surface (§12.6) — "a system that shows what it refused is more credible
    than one that only shows what it produced"."""

    model_config = ConfigDict(frozen=True)

    slate_id: str | None
    criterion: int = Field(ge=1, le=9)
    rule_id: str | None = None
    reason: str
    evaluated_by: Literal["code", "model"]
    round: int = 1
    rejected_at: datetime = Field(default_factory=_utcnow)


class ToolCall(BaseModel):
    model_config = ConfigDict(frozen=True)

    call_id: str
    tool: str
    args: dict[str, Any]
    scope_required: Scope
    ok: bool = True
    error: str | None = None
    latency_s: float = 0.0
    called_at: datetime = Field(default_factory=_utcnow)

    @property
    def dedupe_key(self) -> str:
        """§7.8 tool-thrash mitigation keys on this: same tool, tiny arg
        variations."""
        payload = json.dumps(self.args, sort_keys=True, default=str)
        return hashlib.sha256(f"{self.tool}|{payload}".encode()).hexdigest()[:16]


class RouteDecisionRecord(BaseModel):
    """Recorded because retrieval_routing_accuracy is unmeasurable unless the
    run remembers what it decided."""
    model_config = ConfigDict(frozen=True)
    query: str
    shape: str
    strategy: str
    corpus: str | None
    decided_by: str
    confidence: float
    by_rule: bool


class GateReason(str, Enum):
    PUBLISH = "publish"
    POLICY_OVERRIDE = "policy_override"
    LOW_CONFIDENCE = "low_confidence"
    REPEAT_FAILURE = "repeat_failure"


class GateRequest(BaseModel):
    model_config = ConfigDict(frozen=True)
    gate_id: str
    reason: GateReason
    summary: str
    slate_id: str | None = None
    rule_ids: list[str] = Field(default_factory=list)
    required_scope: Scope | None = None
    opened_at: datetime = Field(default_factory=_utcnow)


class GateResolution(BaseModel):
    model_config = ConfigDict(frozen=True)
    gate_id: str
    decision: Literal["approve", "reject", "amend"]
    actor_id: str
    note: str = ""
    resolved_at: datetime = Field(default_factory=_utcnow)


class Confidence(BaseModel):
    model_config = ConfigDict(frozen=True)
    stated: float = Field(ge=0.0, le=1.0)
    evidence_coverage: float = Field(ge=0.0, le=1.0)
    suppressed: bool = False
    suppression_reason: str | None = None


class Phase(str, Enum):
    PLANNING = "planning"
    RETRIEVING = "retrieving"
    ANALYSING = "analysing"
    MERCHANDISING = "merchandising"
    CRITIQUING = "critiquing"
    GATED = "gated"
    EXPLAINING = "explaining"
    DONE = "done"
    FAILED = "failed"


class SubTask(BaseModel):
    id: str
    description: str
    assigned_to: Literal["retriever", "analyst", "merchandiser"]
    done: bool = False


class RunError(BaseModel):
    model_config = ConfigDict(frozen=True)
    kind: str
    detail: str
    node: str | None = None
    at: datetime = Field(default_factory=_utcnow)


# ── reducers: accumulate, never overwrite ────────────────────────────────────

def append_unique_by(key: str):
    def reducer(left: list, right: list) -> list:
        if not right:
            return left
        seen = {getattr(x, key) for x in left}
        return left + [x for x in right if getattr(x, key) not in seen]
    return reducer


def append_all(left: list, right: list) -> list:
    return (left or []) + (right or [])


class MerchandisingRun(BaseModel):
    """The run record. Accumulating fields use reducers; control fields are
    scalars and are overwritten by design — an append-only `phase` is not
    accumulation, it is a bug."""

    run_id: str
    goal: str                      # restated every step — guards plan drift
    caller_id: str
    caller_role: Role
    granted_scopes: frozenset[Scope]
    budget: Budget = Field(default_factory=Budget)

    evidence: Annotated[list[Evidence], append_unique_by("evidence_id")] = Field(default_factory=list)
    claims: Annotated[list[Claim], append_unique_by("claim_id")] = Field(default_factory=list)
    candidate_slates: Annotated[list[Slate], append_unique_by("slate_id")] = Field(default_factory=list)
    rejections: Annotated[list[Rejection], append_all] = Field(default_factory=list)
    injections_detected: Annotated[list[Finding], append_all] = Field(default_factory=list)
    errors: Annotated[list[RunError], append_all] = Field(default_factory=list)
    tool_calls: Annotated[list[ToolCall], append_all] = Field(default_factory=list)
    route_decisions: Annotated[list[RouteDecisionRecord], append_all] = Field(default_factory=list)
    lineage: dict[str, list[str]] = Field(default_factory=dict)

    phase: Phase = Phase.PLANNING
    plan: list[SubTask] = Field(default_factory=list)
    critic_rounds: int = 0
    pending_gate: GateRequest | None = None
    gate_history: list[GateResolution] = Field(default_factory=list)
    confidence: Confidence | None = None
    final_slate_id: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)

    # ── derived views ────────────────────────────────────────────────────────

    def evidence_by_id(self) -> dict[str, Evidence]:
        return {e.evidence_id: e for e in self.evidence}

    def unresolved_claims(self) -> list[Claim]:
        """Claims citing an evidence_id that does not resolve. Critic criterion
        1, and the ungrounded_claim_rate hard gate."""
        known = set(self.evidence_by_id())
        return [c for c in self.claims if not set(c.evidence_ids) <= known]

    def evidence_coverage(self) -> float:
        if not self.claims:
            return 1.0
        known = set(self.evidence_by_id())
        ok = sum(1 for c in self.claims if set(c.evidence_ids) <= known)
        return ok / len(self.claims)

    def has_untrusted_evidence(self) -> bool:
        return any(e.trust == "untrusted" for e in self.evidence)

    def duplicate_tool_calls(self) -> int:
        keys = [t.dedupe_key for t in self.tool_calls]
        return len(keys) - len(set(keys))

    def record_lineage(self, artefact_id: str, produced_by: list[str]) -> None:
        self.lineage.setdefault(artefact_id, []).extend(produced_by)
