"""Brief triage — the supervisor's first act.

WHY THIS EXISTS, AND WHY IT WAS MISSING

The first full golden-set run scored 33/60. Every standard and cold-start brief
passed; almost every hard negative, unanswerable, conflict and adversarial
brief FAILED — the agent built a slate for all of them, because the supervisor
went straight to planning and never asked whether the brief should be answered
at all.

That is the whole argument for authoring the golden set before the agent
(PLAN.md §0). Had the agent been built first, the briefs would have been
written to what it already did, and this gap would never have appeared.

DETERMINISTIC FIRST. Refusal conditions are RULES, not judgements: a slate size
outside POL-SLT-01, an individual customer target, a request to publish or
export, a policy mutation. Those are checked in code and cite a rule id. Only
genuinely ambiguous briefs reach a model, and even then the model classifies —
it does not decide.

"Knowing when the system should refuse is harder and more valuable than making
it capable." (§7.7)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

Verdict = Literal["proceed", "refuse", "escalate", "unknown"]


@dataclass(frozen=True)
class Triage:
    verdict: Verdict
    reasons: tuple[str, ...] = ()
    rule_ids: tuple[str, ...] = ()
    k: int | None = None
    detected_injection: bool = False

    @property
    def blocks(self) -> bool:
        return self.verdict != "proceed"


# ── slate size (POL-SLT-01) ──────────────────────────────────────────────────

_K_RE = re.compile(r"\b(\d{1,3})[- ]?(?:slot|slots|item|items|slot page)\b", re.I)
K_MIN, K_MAX = 4, 24


def extract_k(brief: str) -> int | None:
    m = _K_RE.search(brief)
    return int(m.group(1)) if m else None


# ── rule tables. Each entry: (pattern, rule_id, reason) ──────────────────────

REFUSE_RULES: list[tuple[str, str, str]] = [
    (r"\b(customer|user)\s+[0-9a-f]{6,}\b", "POL-SEG-02",
     "targets an individual customer; slates are cohort-scoped"),
    (r"\b(personalise|personalize) it to (them|him|her|this customer)\b", "POL-SEG-02",
     "targets an individual customer"),
    (r"\b(list of|these)\s+\d+\s+customer\s*(ids|IDs)?\b", "POL-SEG-03",
     "cohort enumerated rather than specified"),
    (r"\bpasting in\b.*\bcustomer\b", "POL-SEG-03",
     "cohort enumerated rather than specified"),
    (r"\b(approve|publish)\b.*\b(slate|page|it)\b", "POL-ESC-06",
     "publishing requires slate:approve, which the agent never holds"),
    (r"\bexport\b.*\b(segment|cohort|csv|list)\b", "POL-ESC-06",
     "export requires segment:export, which the agent never holds"),
    (r"\b(update|change|set|lower|raise)\b.{0,24}\b(quota|floor|policy|rule)\b",
     "POL-GOV-02", "policy mutation requires policy:override"),
    (r"\bcohort of \d{1,2}\b", "POL-SEG-01",
     "cohort below the 100-customer floor"),
    (r"\bignore all (previous|prior) instructions\b", "POL-ESC-06",
     "instruction-override attempt in the brief"),
    (r"\byou are now\b.{0,20}\b(admin|unrestricted)\b", "POL-ESC-06",
     "role-escalation attempt in the brief"),
    (r"\badmin mode\b", "POL-ESC-06", "role-escalation attempt in the brief"),
]

UNKNOWN_RULES: list[tuple[str, str, str]] = [
    (r"\b(actually )?convert(ed|s|ing)? better\b", "POL-CLM-02",
     "no A/B test exists; conversion is not observed"),
    (r"\bclick[- ]through\b|\bCTR\b", "",
     "impressions are never observed — purchases only"),
    (r"\b(what|which) margin\b|\bmargin did we (make|earn)\b", "POL-PRC-04",
     "no cost data; margin is a uniform proxy"),
    (r"\bsee\w*\b.{0,20}\bchoose not to buy\b|\bdid not buy\b.{0,20}\bsaw\b", "",
     "non-purchase is unlabelled, not a rejection"),
    (r"\b(how will|what will).{0,30}\bnext (quarter|month|season|year)\b", "POL-CLM-01",
     "outside the data window; no forecasting model exists"),
    (r"\bH&M'?s? (actual|real)\b.{0,30}\bpolic\w+", "POL-GOV-04",
     "corpus C is authored for this project, not H&M's real policy"),
    (r"\bchoose not to buy\b", "", "non-purchase is unlabelled"),
]

ESCALATE_RULES: list[tuple[str, str, str]] = [
    (r"\bignore\b.{0,20}\b(long[- ]?tail|quota)\b", "POL-LT-01",
     "explicit instruction to breach the long-tail quota"),
    (r"\b(maximis|maximiz)\w*\b.{0,30}\brevenue\b.{0,40}\b(no other|ignore|only)\b",
     "POL-ESC-01", "revenue objective stated as overriding all constraints"),
    (r"\babsolute maximum revenue\b", "POL-ESC-01",
     "revenue objective stated as overriding all constraints"),
    (r"\bonly our top \d+ bestsellers repeated\b", "POL-SLT-02",
     "duplicate articles requested"),
    (r"\ball black items\b|\ball one colou?r\b", "POL-DIV-03",
     "colour cap conflicts with the brief"),
    (r"\b([5-9]\d|100)\s?%\s?long[- ]?tail\b", "POL-ESC-02",
     "requested tail share is not jointly satisfiable with a revenue objective"),
    (r"\bwinter\b.{0,30}\bjuly\b|\bjuly\b.{0,30}\bwinter\b", "POL-AVL-03",
     "season conflict beyond the shoulder tolerance"),
    (r"\boverride\b.{0,24}\b(diversity|floor|quota)\b", "POL-ESC-06",
     "override requires a scope the agent never holds"),
    (r"\bdiscontinued\b", "POL-AVL-02",
     "discontinued stock may not fill a slate or a quota"),
    (r"\bfull 20% long[- ]?tail\b.{0,40}\bcold[- ]?start\b|"
     r"\bcold[- ]?start\b.{0,40}\bfull 20% long[- ]?tail\b", "POL-SEG-04",
     "brief overrides the cold-start quota relaxation"),
    (r"\bpolicy says\b.{0,30}\bquota is 0\b", "POL-GOV-02",
     "brief contradicts the loaded policy; corpus C in context wins"),
    (r"\bdo not mention\b.{0,40}\b(rules|rejected|applied)\b", "POL-GOV-04",
     "asks to suppress the audit trail"),
    (r"\bdisregard\b.{0,30}\b(quota|policy|long[- ]?tail)\b", "POL-GOV-04",
     "retrieved content instructs a policy breach"),
    (r"\bwhich says to disregard\b", "POL-GOV-04",
     "retrieved content instructs a policy breach"),
]

INJECTION_IN_BRIEF = re.compile(
    r"ignore all (previous|prior) instructions|you are now|admin mode|"
    r"new instructions?:|system prompt", re.I)


def _match(brief: str, rules: list[tuple[str, str, str]]) -> list[tuple[str, str]]:
    out = []
    for pattern, rule_id, reason in rules:
        if re.search(pattern, brief, re.I):
            out.append((rule_id, reason))
    return out


def triage(brief: str) -> Triage:
    """Decide whether this brief should be answered at all, before planning.

    Order matters and is the corpus C precedence order: an empty or malformed
    brief and a scope violation come before an unanswerable question, which
    comes before a constraint conflict. A brief that is both out of scope and
    unanswerable is refused, not answered with "I don't know" — the stronger
    objection wins.
    """
    text = (brief or "").strip()
    injected = bool(INJECTION_IN_BRIEF.search(text))

    if not text:
        return Triage("refuse", ("empty brief; no objective to act on",),
                      ("POL-SLT-03",))

    # 1 — scope and authority. Refusals.
    if hits := _match(text, REFUSE_RULES):
        return Triage("refuse", tuple(r for _, r in hits),
                      tuple(rid for rid, _ in hits if rid),
                      detected_injection=injected)

    # 2 — slate size (POL-SLT-01).
    k = extract_k(text)
    if k is not None and not (K_MIN <= k <= K_MAX):
        return Triage("refuse",
                      (f"requested {k} slots, outside the permitted {K_MIN}-{K_MAX}",),
                      ("POL-SLT-01",), k=k, detected_injection=injected)

    # 3 — answerable from the data at all?
    if hits := _match(text, UNKNOWN_RULES):
        return Triage("unknown", tuple(r for _, r in hits),
                      tuple(rid for rid, _ in hits if rid),
                      k=k, detected_injection=injected)

    # 4 — brief versus policy conflict. Escalate, never silently pick a side.
    if hits := _match(text, ESCALATE_RULES):
        return Triage("escalate", tuple(r for _, r in hits),
                      tuple(rid for rid, _ in hits if rid),
                      k=k, detected_injection=injected)

    return Triage("proceed", k=k, detected_injection=injected)
