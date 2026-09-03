"""The critic — nine enumerable criteria (ARCHITECTURE.md §7.6).

"A critic is not 'ask the model if it's sure.' That is theatre. A real critic
applies named, enumerable rejection criteria and emits a structured rejection
with a reason and a citation."

Criteria 3, 4, 5 and 9 are evaluated IN CODE. Only 1, 2, 6, 7 and 8 involve
judgement, and 1 and 6 are mostly mechanical too. Every rejection cites a
corpus C rule id, so it is traceable to a policy line rather than to a vibe.

ADVERSARIAL SEPARATION (§7.4). In LangGraph every node reads the same state
object, so separation has to be CONSTRUCTED, not assumed. `CriticView` is that
construction: the critic sees the slate, the claims, the resolved evidence, the
policy and the scopes — and NOT the plan, the supervisor's reasoning, or its
own prior rejections. A node that critiques its own working memory
rationalises; one that sees only the output and the policy rejects.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from services.api.agent.state import (
    Claim, Confidence, Evidence, MerchandisingRun, Rejection, Slate,
)
from services.api.core.rbac import Scope, is_write_class

MAX_ROUNDS = 2

# Thresholds sourced from corpus C, not invented here. POL-CLM-05's 0.80 is
# marked PROVISIONAL_UNGROUNDED until D8 defines it from the unanswerable
# briefs — until then a criterion-8 suppression is directional, not decisive.
COVERAGE_FLOOR = 0.80

# Regexes, not literal substrings. The first version matched "increased
# revenue" and sailed past "revenue increased by 12%" — the exact leak
# POL-CLM-01 predicts of any denylist. Word order is the failure mode, so the
# patterns are order-tolerant and anchored on the CAUSAL VERB rather than on a
# phrase. Every leak found from here becomes a permanent golden-set regression
# case per §10.4.
BANNED_CAUSAL_PATTERNS = [
    r"\b(revenue|sales|conversion|lift|uplift|ctr)\b[^.]{0,40}\b"
    r"(increased|rose|grew|improved|jumped|climbed)\b",
    r"\b(increased|raised|grew|improved|boosted|lifted)\b[^.]{0,40}\b"
    r"(revenue|sales|conversion|margin)\b",
    r"\b(drove|caused|resulted in|led to|produced)\b",
    r"\bmeasured\b[^.]{0,20}\b(lift|uplift|increase|effect)\b",
    r"\b(proved|proven|demonstrated that|shows that)\b",
    r"\b(generated|delivered|achieved)\b[^.]{0,20}\b(uplift|lift|increase)\b",
    r"\bbecause of (the |this )?(model|recommendation|personalisation)\b",
]
_CAUSAL_RE = [re.compile(p, re.IGNORECASE) for p in BANNED_CAUSAL_PATTERNS]
REQUIRED_QUALIFIERS = ("projected", "estimated", "modelled", "would be expected")

# Patterns tolerate INTERVENING WORDS. The first version required
# "disregard the policy" adjacently and sailed past "disregard the
# MERCHANDISING policy" — the same leak POL-CLM-01 predicts of any denylist,
# found here by its own red-team test rather than in production. Anchored on
# the imperative verb, with a bounded gap to its object.
INJECTION_PATTERNS = [
    r"ignore\s+(all|any|the)?\s*(previous|prior|above|preceding)",
    r"\b(disregard|override|bypass|forget)\b[^.]{0,40}?"
    r"\b(instruction|rule|polic|quota|guideline|constraint)",
    r"you are now\b",
    r"new instructions?\s*:",
    r"system prompt",
    r"</?(untrusted_content|system|instructions?)>",
    r"\breveal\b[^.]{0,30}?\b(prompt|instruction|system|customer id)",
    r"\bdo not\b[^.]{0,30}?\b(tell|inform|mention|report|disclose)",
    r"\bact as\b",
    r"\bpretend\b[^.]{0,20}?\b(to be|you are|that you)",
    r"\bpre[- ]?authoris(ed|e)\b|\bpre[- ]?approv(ed|e)\b",
]
_INJECTION_RE = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


@dataclass(frozen=True)
class CriticView:
    """The projection the critic is allowed to see. Deliberately narrow."""
    slate: Slate | None
    claims: tuple[Claim, ...]
    evidence: dict[str, Evidence]
    policy_document: str
    policy_rule_ids: frozenset[str]
    granted_scopes: frozenset[Scope]
    attempted_scopes: tuple[Scope, ...]
    slate_meta: dict

    @classmethod
    def project(cls, run: MerchandisingRun, slate: Slate | None,
                policy_document: str = "", policy_rule_ids: frozenset[str] = frozenset(),
                slate_meta: dict | None = None) -> "CriticView":
        return cls(
            slate=slate,
            claims=tuple(run.claims),
            evidence=run.evidence_by_id(),
            policy_document=policy_document,
            policy_rule_ids=policy_rule_ids,
            granted_scopes=run.granted_scopes,
            attempted_scopes=tuple(t.scope_required for t in run.tool_calls),
            slate_meta=slate_meta or {},
        )


# ── Criteria evaluated in CODE ───────────────────────────────────────────────

def criterion_1_citations_resolve(view: CriticView) -> list[Rejection]:
    """A claim cites no evidence, or an evidence_id that does not resolve.
    Feeds the ungrounded_claim_rate hard gate (target 0.00)."""
    out = []
    for c in view.claims:
        missing = [e for e in c.evidence_ids if e not in view.evidence]
        if missing:
            out.append(Rejection(
                slate_id=view.slate.slate_id if view.slate else None,
                criterion=1, rule_id="POL-CLM-03", evaluated_by="code",
                reason=f"claim {c.claim_id} cites unresolvable evidence: {missing}",
            ))
    return out


def criterion_3_long_tail_quota(view: CriticView) -> list[Rejection]:
    """POL-LT-01 / POL-SEG-04. Evaluated in code against the optimiser report."""
    if not view.slate:
        return []
    rep = view.slate.optimiser_report or {}
    required = rep.get("long_tail_required", 0)
    got = sum(1 for s in view.slate.slots if s.is_long_tail)
    if required and got < required:
        return [Rejection(
            slate_id=view.slate.slate_id, criterion=3, rule_id="POL-LT-01",
            evaluated_by="code",
            reason=f"long-tail quota unmet: {got}/{required} slots",
        )]
    return []


def criterion_4_diversity_floor(view: CriticView) -> list[Rejection]:
    """POL-DIV-01..03."""
    if not view.slate:
        return []
    out = []
    for rule in (view.slate.optimiser_report or {}).get("binding_constraints", []):
        if rule in ("POL-DIV-01", "POL-DIV-02", "POL-DIV-03"):
            out.append(Rejection(
                slate_id=view.slate.slate_id, criterion=4, rule_id=rule,
                evaluated_by="code", reason=f"diversity constraint {rule} bound",
            ))
    return out


def criterion_5_availability(view: CriticView) -> list[Rejection]:
    """POL-AVL-01..03. An article out of season or no longer selling."""
    if not view.slate:
        return []
    bad = view.slate_meta.get("unavailable_articles", [])
    if bad:
        return [Rejection(
            slate_id=view.slate.slate_id, criterion=5, rule_id="POL-AVL-01",
            evaluated_by="code",
            reason=f"{len(bad)} unavailable or out-of-season articles in slate",
        )]
    return []


def criterion_6_projected_language(view: CriticView) -> list[Rejection]:
    """POL-CLM-01. Mostly mechanical: a denylist plus a required qualifier on
    claims the model itself declared `projected`. The `kind` enum is what moves
    this out of pure judgement — the model declares the kind, code checks the
    phrasing."""
    out = []
    for c in view.claims:
        low = c.text.lower()
        for rx in _CAUSAL_RE:
            if (m := rx.search(low)):
                out.append(Rejection(
                    slate_id=view.slate.slate_id if view.slate else None,
                    criterion=6, rule_id="POL-CLM-01", evaluated_by="code",
                    reason=f"causal language {m.group(0)!r} in claim {c.claim_id}",
                ))
                break
        else:
            if c.kind == "projected" and not any(q in low for q in REQUIRED_QUALIFIERS):
                out.append(Rejection(
                    slate_id=view.slate.slate_id if view.slate else None,
                    criterion=6, rule_id="POL-CLM-01", evaluated_by="code",
                    reason=f"claim {c.claim_id} is projected but lacks a qualifier",
                ))
    return out


def criterion_7_injection(view: CriticView) -> list[Rejection]:
    """Retrieved content contained instruction-like text. Detections are
    COUNTED, not merely defended against — injection_detection_recall is a
    reported metric (§8.5, §10)."""
    out = []
    for ev in view.evidence.values():
        if ev.trust != "untrusted":
            continue
        for rx in _INJECTION_RE:
            if rx.search(ev.content):
                out.append(Rejection(
                    slate_id=view.slate.slate_id if view.slate else None,
                    criterion=7, rule_id="POL-GOV-04", evaluated_by="code",
                    reason=f"instruction-like text in evidence {ev.evidence_id}: "
                           f"/{rx.pattern}/",
                ))
                break
    return out


def criterion_8_confidence_coverage(view: CriticView, stated: float | None
                                    ) -> tuple[list[Rejection], Confidence | None]:
    """POL-CLM-05. Suppress confidence rather than reject the slate.

    Thin evidence is a reason to stop claiming certainty, not a reason to
    refuse to answer. Conflating the two produces a system that refuses
    constantly and is therefore ignored.
    """
    if stated is None:
        return [], None
    resolvable = sum(1 for c in view.claims
                     if set(c.evidence_ids) <= set(view.evidence))
    coverage = resolvable / len(view.claims) if view.claims else 1.0
    if coverage < COVERAGE_FLOOR:
        return [Rejection(
            slate_id=view.slate.slate_id if view.slate else None,
            criterion=8, rule_id="POL-CLM-05", evaluated_by="code",
            reason=f"evidence coverage {coverage:.2f} < {COVERAGE_FLOOR}; "
                   "confidence suppressed",
        )], Confidence(stated=stated, evidence_coverage=coverage, suppressed=True,
                       suppression_reason="POL-CLM-05")
    return [], Confidence(stated=stated, evidence_coverage=coverage)


def criterion_9_scope(view: CriticView) -> list[Rejection]:
    """The run attempted an action outside its granted scopes. Re-asserted
    here post-hoc even though the tool boundary already blocks it: the boundary
    is the control, this is the evidence. scope_violation_rate is a hard gate."""
    out = []
    for scope in set(view.attempted_scopes):
        if scope not in view.granted_scopes or is_write_class(scope):
            out.append(Rejection(
                slate_id=view.slate.slate_id if view.slate else None,
                criterion=9, rule_id="POL-ESC-06", evaluated_by="code",
                reason=f"tool call required out-of-scope {scope.value!r}",
            ))
    return out


@dataclass
class CritiqueResult:
    passed: bool
    rejections: list[Rejection]
    confidence: Confidence | None
    round: int

    @property
    def by_criterion(self) -> dict[int, int]:
        out: dict[int, int] = {}
        for r in self.rejections:
            out[r.criterion] = out.get(r.criterion, 0) + 1
        return out


def critique(view: CriticView, stated_confidence: float | None = None,
             round_: int = 1) -> CritiqueResult:
    """Run every criterion. Code-evaluated criteria run unconditionally and
    cost nothing; 2 (does the evidence actually contain the fact) is the only
    one genuinely needing a model and is handled by the critic NODE."""
    rejections: list[Rejection] = []
    rejections += criterion_1_citations_resolve(view)
    rejections += criterion_3_long_tail_quota(view)
    rejections += criterion_4_diversity_floor(view)
    rejections += criterion_5_availability(view)
    rejections += criterion_6_projected_language(view)
    rejections += criterion_7_injection(view)
    conf_rej, confidence = criterion_8_confidence_coverage(view, stated_confidence)
    rejections += conf_rej
    rejections += criterion_9_scope(view)

    for r in rejections:
        object.__setattr__(r, "round", round_)

    # Criterion 8 SUPPRESSES rather than rejects — a suppressed-confidence
    # slate still passes and is still shown.
    blocking = [r for r in rejections if r.criterion != 8]
    return CritiqueResult(passed=not blocking, rejections=rejections,
                          confidence=confidence, round=round_)
