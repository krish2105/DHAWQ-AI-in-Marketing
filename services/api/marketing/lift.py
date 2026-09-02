"""Projected incremental revenue — ARCHITECTURE.md §11.

    lift = Σ (P(purchase | recommended) × margin)  —  same for baseline

THE NUMBER DEFENDED IN THE VIVA (§0). It is also the number most easily
overstated, so the vocabulary is enforced rather than trusted: every field
name here contains "projected", and POL-CLM-01 rejects any explanation that
describes it as measured, caused or driven.

WHY IT IS PROJECTED AND NOT MEASURED (§17)
------------------------------------------
There is no live A/B test. This estimates what WOULD have happened under
assumptions that are stated, not laundered: no position bias, no interference
between slots, and offline relevance as a proxy for conversion. All three are
false to some degree. Stating them is what makes the estimate credible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ASSUMED_GROSS_MARGIN_RATE = 0.55     # corpus C `margin_proxy`


@dataclass(frozen=True)
class LiftResult:
    projected_revenue_per_session: float
    baseline_projected_revenue_per_session: float
    projected_incremental_revenue: float
    projected_lift_pct: float
    coverage_cost_pp: float
    k: int
    n_sessions: int
    assumptions: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "projected_revenue_per_session": round(self.projected_revenue_per_session, 6),
            "baseline_projected_revenue_per_session":
                round(self.baseline_projected_revenue_per_session, 6),
            "projected_incremental_revenue": round(self.projected_incremental_revenue, 6),
            "projected_lift_pct": round(self.projected_lift_pct, 2),
            "coverage_cost_pp": round(self.coverage_cost_pp, 2),
            "k": self.k, "n_sessions": self.n_sessions,
            "assumptions": self.assumptions,
            "language_note": "PROJECTED, not measured. No live A/B test exists.",
        }


def purchase_probability(rank: int, relevance: float, position_decay: float = 0.9) -> float:
    """Crude click-through-style decay by slot position.

    Deliberately simple and deliberately declared. A more elaborate propensity
    model would look more scientific without being more true — position bias is
    UNOBSERVABLE in this dataset (§3), so any curve here is an assumption, and
    a transparent assumption beats a sophisticated one that hides its own
    arbitrariness.
    """
    return relevance * (position_decay ** rank)


def projected_revenue(
    slate: list[str], relevance: dict[str, float], price: dict[str, float],
    margin_rate: float = ASSUMED_GROSS_MARGIN_RATE,
) -> float:
    return float(sum(
        purchase_probability(i, relevance.get(a, 0.0)) * price.get(a, 0.0) * margin_rate
        for i, a in enumerate(slate)
    ))


def project_lift(
    slates: dict[str, list[str]], baseline_slates: dict[str, list[str]],
    relevance: dict[str, dict[str, float]], price: dict[str, float],
    model_coverage: float, baseline_coverage: float,
    margin_rate: float = ASSUMED_GROSS_MARGIN_RATE,
) -> LiftResult:
    """Compare a model's slates against the baseline's, per session.

    The coverage cost travels WITH the revenue number, in the same object, so
    it is impossible to quote the upside without the cost. §9: "the tension is
    the finding."
    """
    common = sorted(set(slates) & set(baseline_slates))
    if not common:
        return LiftResult(0, 0, 0, 0, 0, 0, 0, ["no overlapping sessions"])

    k = len(next(iter(slates.values())))
    model_rev = float(np.mean([
        projected_revenue(slates[c], relevance.get(c, {}), price, margin_rate)
        for c in common
    ]))
    base_rev = float(np.mean([
        projected_revenue(baseline_slates[c], relevance.get(c, {}), price, margin_rate)
        for c in common
    ]))
    return LiftResult(
        projected_revenue_per_session=model_rev,
        baseline_projected_revenue_per_session=base_rev,
        projected_incremental_revenue=model_rev - base_rev,
        projected_lift_pct=100.0 * (model_rev - base_rev) / base_rev if base_rev else 0.0,
        coverage_cost_pp=100.0 * (model_coverage - baseline_coverage),
        k=k, n_sessions=len(common),
        assumptions=[
            "No position bias beyond the declared geometric decay (unobservable in H&M data)",
            "No interference between slots",
            f"Uniform gross margin rate of {margin_rate} — corpus C margin_proxy; "
            "true margin is unobservable",
            "Offline relevance used as a proxy for conversion probability",
            "Purchases, not impressions: unpurchased articles are unlabelled, not rejected",
        ],
    )
