"""Audit ANY slate against corpus C — including one the optimiser never saw.

WHY THIS IS SEPARATE FROM THE OPTIMISER. The critic's criteria 3, 4 and 5 read
the OptimiserReport, so they check the optimiser's account of its own work.
That is the right instrument for catching a broken optimiser and the wrong one
for the question here: how often would a slate built WITHOUT these constraints
breach the policy? A revenue-ranked page has no report to read.

So this reimplements the same rules against slate CONTENTS. The duplication is
deliberate and is the point — an auditor that called optimise_slots() could
only ever agree with it. A test asserts the two agree on optimiser output,
which is what keeps the duplication honest rather than merely duplicated.

Nothing here is estimated. Every violation is a rule id and the slate that
breached it.
"""

from __future__ import annotations

from dataclasses import dataclass

from services.api.marketing.slots import (
    Candidate, max_per_attribute, required_long_tail_slots)


@dataclass(frozen=True)
class Violation:
    rule_id: str
    detail: str

    def __str__(self) -> str:                          # pragma: no cover
        return f"{self.rule_id}: {self.detail}"


def audit(
    slate: list[Candidate],
    k: int,
    *,
    min_long_tail_share: float = 0.20,
    quota_waiver_below_k: int = 5,
    max_type_divisor: int = 4,
    max_colour_divisor: int = 3,
    attribute_floor: int = 2,
    max_price_ratio: float = 6.0,
) -> list[Violation]:
    """Every corpus C rule this slate breaches. Empty means compliant."""
    out: list[Violation] = []

    # POL-SLT-01 — slate size bounds.
    if not 4 <= k <= 24:
        out.append(Violation("POL-SLT-01", f"k={k} outside 4-24"))
    if len(slate) != k:
        out.append(Violation("POL-SLT-03",
                             f"underfilled: {len(slate)}/{k} slots"))

    # POL-SLT-02 — every slot a distinct article.
    ids = [c.article_id for c in slate]
    if len(set(ids)) != len(ids):
        out.append(Violation("POL-SLT-02",
                             f"{len(ids) - len(set(ids))} duplicate articles"))

    # POL-AVL-01 / POL-AVL-03 — unsellable stock may not fill a slot.
    if n := sum(1 for c in slate if not c.available):
        out.append(Violation("POL-AVL-01", f"{n} unavailable articles"))
    if n := sum(1 for c in slate if not c.season_ok):
        out.append(Violation("POL-AVL-03", f"{n} out-of-season articles"))

    # POL-LT-01 / POL-LT-02 — long-tail quota, with the short-slate waiver.
    need = required_long_tail_slots(k, min_long_tail_share, quota_waiver_below_k)
    got = sum(1 for c in slate if c.is_long_tail)
    if got < need:
        out.append(Violation("POL-LT-01", f"long-tail {got}/{need} slots"))

    # POL-DIV-02 / POL-DIV-03 — attribute concentration caps.
    for rule, key, divisor in (("POL-DIV-02", "product_type", max_type_divisor),
                               ("POL-DIV-03", "colour_group", max_colour_divisor)):
        cap = max_per_attribute(k, divisor, attribute_floor)
        counts: dict[str, int] = {}
        for c in slate:
            v = getattr(c, key)
            counts[v] = counts.get(v, 0) + 1
        worst = max(counts.items(), key=lambda kv: kv[1], default=(None, 0))
        if worst[1] > cap:
            out.append(Violation(rule, f"{worst[1]} of '{worst[0]}' exceeds cap {cap}"))

    # POL-PRC-01 — price coherence across the slate.
    prices = [c.price for c in slate if c.price > 0]
    if max_price_ratio > 0 and len(prices) > 1:
        lo, hi = min(prices), max(prices)
        if lo > 0 and hi / lo > max_price_ratio:
            out.append(Violation(
                "POL-PRC-01",
                f"price spread {hi / lo:.1f}x exceeds {max_price_ratio}x"))

    return out


def revenue_ranked(candidates: list[Candidate], k: int) -> list[Candidate]:
    """The slate you get with no policy at all: the top k by score.

    This is the counterfactual the whole cost case rests on, so it is written
    as the simplest possible thing rather than as a straw man — it is exactly
    what a revenue-ranked module does, and exactly what the optimiser starts
    from before any constraint is applied.
    """
    return sorted(candidates, key=lambda c: (-c.score, c.article_id))[:k]
