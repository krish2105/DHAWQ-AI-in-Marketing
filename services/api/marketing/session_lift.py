"""Per-customer session lift — the number a CMO actually acts on.

WHY THIS EXISTS ALONGSIDE THE COHORT NUMBER
The cohort-slate comparison asks: "one page for a whole segment, versus the
bestseller page." Personalisation cannot win it, and not because the model is
weak — because the relevance signal is aggregate purchase frequency, which IS
what the popularity arm ranks on. The modal preference of a large cohort is,
definitionally, its bestsellers.

A real ecommerce page is not one slate per segment. It is personalised PER
VISITOR. That is the scenario §9's ranking metrics already measure, and there
collaborative beats popularity. This module prices that difference in revenue
rather than leaving it as an NDCG delta a CMO cannot use.

BOTH NUMBERS SHIP. They answer different questions and hiding either would
misrepresent the system:
  session lift  what personalising each visitor's page is worth
  cohort lift   what one slate per segment is worth, and why it converges on
                the bestseller page as the segment widens

RELEVANCE IS THE CUSTOMER'S OWN HELD-OUT PURCHASES. Per-customer ground truth
is independent of both arms — neither can influence what that person actually
bought in the test window — which is exactly the property the cohort version
struggled to find.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ASSUMED_GROSS_MARGIN_RATE = 0.55      # corpus C `margin_proxy`
POSITION_DECAY = 0.9


@dataclass(frozen=True)
class SessionLift:
    model: str
    n_sessions: int
    k: int
    projected_revenue_per_session: float
    baseline_projected_revenue_per_session: float
    projected_incremental_revenue: float
    projected_lift_pct: float
    ci_low_pct: float
    ci_high_pct: float
    hit_rate: float
    baseline_hit_rate: float
    coverage: float
    baseline_coverage: float
    coverage_delta_pp: float

    def as_dict(self) -> dict:
        return {
            "model": self.model, "n_sessions": self.n_sessions, "k": self.k,
            "projected_revenue_per_session": round(self.projected_revenue_per_session, 6),
            "baseline_projected_revenue_per_session":
                round(self.baseline_projected_revenue_per_session, 6),
            "projected_incremental_revenue": round(self.projected_incremental_revenue, 6),
            "projected_lift_pct": round(self.projected_lift_pct, 2),
            "ci95": [round(self.ci_low_pct, 2), round(self.ci_high_pct, 2)],
            "significant": bool(self.ci_low_pct > 0 or self.ci_high_pct < 0),
            "hit_rate": round(self.hit_rate, 4),
            "baseline_hit_rate": round(self.baseline_hit_rate, 4),
            "coverage": round(self.coverage, 4),
            "baseline_coverage": round(self.baseline_coverage, 6),
            "coverage_delta_pp": round(self.coverage_delta_pp, 2),
            "reach_multiple": round(self.coverage / max(self.baseline_coverage, 1e-9), 1),
            "language": "PROJECTED, not measured. No live A/B test exists.",
        }


def _session_revenue(slate: list[str], bought: set[str], price: dict[str, float],
                     margin_rate: float) -> float:
    """Projected margin from one page view.

    A slot earns only if the customer actually bought that article in the test
    window. Position decay reflects that slot 1 is seen more than slot 12 —
    declared rather than fitted, because position bias is UNOBSERVABLE in this
    dataset (§3) and a fitted curve would look scientific while being invented.
    """
    return float(sum(
        (POSITION_DECAY ** i) * price.get(a, 0.0) * margin_rate
        for i, a in enumerate(slate) if a in bought
    ))


def session_lift(
    slates: dict[str, list[str]],
    baseline_slate: list[str],
    truth: dict[str, set[str]],
    price: dict[str, float],
    *,
    model: str,
    catalogue_size: int,
    margin_rate: float = ASSUMED_GROSS_MARGIN_RATE,
    bootstrap: int = 2000,
    seed: int = 20260903,
) -> SessionLift:
    """Personalised page per visitor vs ONE global bestseller page.

    `baseline_slate` is a single list on purpose: the bestseller page is what
    every visitor sees when nobody personalises, which is the counterfactual
    the business is actually choosing against.
    """
    customers = [c for c in slates if truth.get(c)]
    if not customers:
        raise ValueError("no evaluable sessions")

    model_rev = np.array([
        _session_revenue(slates[c], truth[c], price, margin_rate) for c in customers
    ])
    base_rev = np.array([
        _session_revenue(baseline_slate, truth[c], price, margin_rate) for c in customers
    ])

    m, b = float(model_rev.mean()), float(base_rev.mean())
    lift = 100.0 * (m - b) / b if b > 0 else 0.0

    # BOOTSTRAP CI over sessions. A single mean with no interval invites the
    # reader to treat a noisy difference as a finding; §10.5 asks for the
    # number AND what is known about it.
    rng = np.random.default_rng(seed)
    n = len(customers)
    diffs = np.empty(bootstrap)
    for i in range(bootstrap):
        idx = rng.integers(0, n, n)
        bb = base_rev[idx].mean()
        diffs[i] = 100.0 * (model_rev[idx].mean() - bb) / bb if bb > 0 else 0.0
    lo, hi = np.percentile(diffs, [2.5, 97.5])

    shown = {a for c in customers for a in slates[c]}
    return SessionLift(
        model=model, n_sessions=n, k=len(next(iter(slates.values()))),
        projected_revenue_per_session=m,
        baseline_projected_revenue_per_session=b,
        projected_incremental_revenue=m - b,
        projected_lift_pct=lift,
        ci_low_pct=float(lo), ci_high_pct=float(hi),
        hit_rate=float(np.mean([any(a in truth[c] for a in slates[c]) for c in customers])),
        baseline_hit_rate=float(np.mean([any(a in truth[c] for a in baseline_slate)
                                         for c in customers])),
        coverage=len(shown) / max(catalogue_size, 1),
        baseline_coverage=len(set(baseline_slate)) / max(catalogue_size, 1),
        coverage_delta_pp=100.0 * (len(shown) - len(set(baseline_slate))) / max(catalogue_size, 1),
    )


def granularity_curve(
    per_customer: dict[str, list[str]],
    truth: dict[str, set[str]],
    price: dict[str, float],
    cohort_sizes: list[int],
    *,
    margin_rate: float = ASSUMED_GROSS_MARGIN_RATE,
    seed: int = 20260903,
) -> list[dict]:
    """WHERE DOES PERSONALISATION STOP PAYING?

    The cohort finding said personalised slates converge on the bestseller page
    as the cohort widens. That is a direction, not a decision. A merchandiser
    needs the crossover: at what segment size does one shared page stop being
    worth personalising?

    Built by pooling customers into cohorts of increasing size, serving each
    cohort ONE slate (the most common articles among its members' personalised
    pages), and pricing that against the same per-customer ground truth.
    """
    rng = np.random.default_rng(seed)
    customers = [c for c in per_customer if truth.get(c)]
    k = len(next(iter(per_customer.values())))
    out = []

    for size in cohort_sizes:
        if size > len(customers):
            continue
        revs = []
        # Several disjoint cohorts of this size, so one lucky grouping cannot
        # carry the point.
        n_cohorts = max(1, min(12, len(customers) // size))
        order = rng.permutation(len(customers))
        for ci in range(n_cohorts):
            members = [customers[i] for i in order[ci * size:(ci + 1) * size]]
            if len(members) < size:
                break
            counts: dict[str, int] = {}
            for c in members:
                for a in per_customer[c]:
                    counts[a] = counts.get(a, 0) + 1
            shared = [a for a, _ in sorted(counts.items(),
                                           key=lambda kv: (-kv[1], kv[0]))[:k]]
            revs += [_session_revenue(shared, truth[c], price, margin_rate)
                     for c in members]
        if revs:
            out.append({"cohort_size": size, "mean_revenue": float(np.mean(revs)),
                        "n_customers_scored": len(revs)})

    # size 1 is the fully personalised page — the ceiling
    personal = [_session_revenue(per_customer[c], truth[c], price, margin_rate)
                for c in customers]
    out.insert(0, {"cohort_size": 1, "mean_revenue": float(np.mean(personal)),
                   "n_customers_scored": len(personal)})
    ceiling = out[0]["mean_revenue"] or 1.0
    for row in out:
        row["pct_of_personalised"] = round(100.0 * row["mean_revenue"] / ceiling, 1)
        row["mean_revenue"] = round(row["mean_revenue"], 6)
    return out
