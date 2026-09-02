"""The slot optimiser — ARCHITECTURE.md §11.

"Given k page slots and a customer, choose the set maximising projected revenue
subject to a diversity constraint and a minimum long-tail quota. This is where
the model becomes a merchandising decision."

THIS FILE IS THE DETERMINISTIC CORE. The agent CALLS it and never approximates
it. Every constraint enforced here is a numbered rule in corpus C, cited by id,
so a rejection is traceable to a policy line rather than to a heuristic.

WHY GREEDY AND NOT AN ILP. The objective is submodular-ish (marginal value of a
slot falls as similar items are added) and k <= 24, so a constrained greedy pass
with an explicit repair step lands within a couple of percent of optimal while
staying fast, deterministic and — the part that matters for a viva —
EXPLAINABLE. An ILP would produce a better number and no account of why any
particular article got a slot. POL-SLT-03 requires reporting the binding
constraint on under-fill, which a greedy trace gives directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Candidate:
    article_id: str
    score: float
    price: float
    product_type: str
    colour_group: str
    is_long_tail: bool
    available: bool = True
    season_ok: bool = True


@dataclass
class OptimiserReport:
    """Why the slate looks the way it does. Rendered in the merchandise view
    and cited by the agent's explanation."""
    requested_k: int
    filled_k: int
    binding_constraints: list[str] = field(default_factory=list)
    rules_applied: list[str] = field(default_factory=list)
    long_tail_share: float = 0.0
    long_tail_required: int = 0
    rejected: dict[str, int] = field(default_factory=dict)
    repairs: list[str] = field(default_factory=list)
    underfilled: bool = False


def required_long_tail_slots(k: int, min_share: float, waiver_below_k: int) -> int:
    """POL-LT-01 / POL-LT-02. ceil, with the short-slate waiver."""
    if k < waiver_below_k:
        return 0
    return math.ceil(k * min_share)


def max_per_attribute(k: int, divisor: int, floor: int) -> int:
    """POL-DIV-02 / POL-DIV-03."""
    return max(floor, math.ceil(k / divisor))


def optimise_slots(
    candidates: list[Candidate],
    k: int,
    *,
    min_long_tail_share: float = 0.20,
    quota_waiver_below_k: int = 5,
    max_type_divisor: int = 4,
    max_colour_divisor: int = 3,
    attribute_floor: int = 2,
    min_ild: float = 0.35,
    ild_min_k: int = 6,
    embeddings: np.ndarray | None = None,
    index: dict[str, int] | None = None,
    max_price_ratio: float = 6.0,
) -> tuple[list[str], OptimiserReport]:
    """Return (slate, report). Never pads, never silently relaxes a rule."""
    report = OptimiserReport(requested_k=k, filled_k=0)
    report.rules_applied = [
        "POL-SLT-01", "POL-SLT-02", "POL-SLT-03", "POL-SLT-05",
        "POL-LT-01", "POL-LT-02", "POL-LT-04",
        "POL-DIV-02", "POL-DIV-03", "POL-AVL-01", "POL-AVL-03",
    ]

    # POL-SLT-01 — slate size bounds.
    if not 4 <= k <= 24:
        report.binding_constraints.append("POL-SLT-01")
        return [], report

    # POL-AVL-01 / POL-AVL-03 / POL-LT-04 — availability precedes everything.
    # Precedence order in corpus C is AVL before LT before DIV, so unsellable
    # articles are removed before any quota is computed against them.
    pool, dropped = [], {"unavailable": 0, "out_of_season": 0}
    for c in candidates:
        if not c.available:
            dropped["unavailable"] += 1
        elif not c.season_ok:
            dropped["out_of_season"] += 1
        else:
            pool.append(c)
    report.rejected.update({k_: v for k_, v in dropped.items() if v})

    # POL-SLT-05 — fully specified deterministic order. Ties break on
    # long-tail first, then lower price, then article id. Without this the
    # §10.4 stability metric would measure sort-algorithm internals.
    pool.sort(key=lambda c: (-c.score, not c.is_long_tail, c.price, c.article_id))

    need_tail = required_long_tail_slots(k, min_long_tail_share, quota_waiver_below_k)
    report.long_tail_required = need_tail
    cap_type = max_per_attribute(k, max_type_divisor, attribute_floor)
    cap_colour = max_per_attribute(k, max_colour_divisor, attribute_floor)

    chosen: list[Candidate] = []
    n_type: dict[str, int] = {}
    n_colour: dict[str, int] = {}
    seen: set[str] = set()

    def fits(c: Candidate) -> str | None:
        if c.article_id in seen:
            return "POL-SLT-02"
        if n_type.get(c.product_type, 0) >= cap_type:
            return "POL-DIV-02"
        if n_colour.get(c.colour_group, 0) >= cap_colour:
            return "POL-DIV-03"
        if chosen and max_price_ratio > 0:
            prices = [x.price for x in chosen if x.price > 0] + [c.price]
            lo, hi = min(prices), max(prices)
            if lo > 0 and hi / lo > max_price_ratio:
                return "POL-PRC-01"
        return None

    def take(c: Candidate) -> None:
        chosen.append(c)
        seen.add(c.article_id)
        n_type[c.product_type] = n_type.get(c.product_type, 0) + 1
        n_colour[c.colour_group] = n_colour.get(c.colour_group, 0) + 1

    # Pass 1 — reserve the quota slots FIRST. Filling by score and repairing
    # afterwards works, but it evicts high scorers and produces a worse slate
    # than reserving up front, because the tail articles taken later are the
    # best available rather than whatever survives.
    if need_tail:
        for c in (x for x in pool if x.is_long_tail):
            if sum(1 for x in chosen if x.is_long_tail) >= need_tail:
                break
            if (why := fits(c)) is None:
                take(c)
            else:
                report.rejected[why] = report.rejected.get(why, 0) + 1

    # Pass 2 — fill the rest by score.
    for c in pool:
        if len(chosen) >= k:
            break
        if (why := fits(c)) is None:
            take(c)
        elif why != "POL-SLT-02":
            report.rejected[why] = report.rejected.get(why, 0) + 1

    got_tail = sum(1 for c in chosen if c.is_long_tail)

    # POL-LT-01 — if the quota is still short, the slate is NOT compliant.
    # POL-ESC-01 says an unresolvable breach escalates; it never silently
    # relaxes. We report it and let the caller escalate.
    if got_tail < need_tail:
        report.binding_constraints.append("POL-LT-01")
        report.repairs.append(
            f"long-tail quota unmet: {got_tail}/{need_tail} — "
            f"only {sum(1 for c in pool if c.is_long_tail)} tail candidates survived AVL"
        )

    # POL-DIV-01 — intra-list diversity floor, checked last because it is the
    # only constraint needing embeddings and the most likely to be miscalibrated
    # (its threshold is PROVISIONAL_UNGROUNDED until D8).
    if embeddings is not None and index is not None and len(chosen) >= ild_min_k:
        from services.api.evaluate.beyond_accuracy import intra_list_diversity
        ild = intra_list_diversity([c.article_id for c in chosen], embeddings, index)
        if ild < min_ild:
            report.binding_constraints.append("POL-DIV-01")
            report.repairs.append(f"ILD {ild:.3f} < floor {min_ild}")

    # POL-SLT-03 — never pad. Return fewer slots and name what bound.
    if len(chosen) < k:
        report.underfilled = True
        if not report.binding_constraints:
            report.binding_constraints.append("insufficient_candidates")

    report.filled_k = len(chosen)
    report.long_tail_share = got_tail / len(chosen) if chosen else 0.0
    return [c.article_id for c in chosen], report
