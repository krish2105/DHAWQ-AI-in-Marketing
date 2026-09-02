"""Customer lifetime value — BG/NBD + Gamma-Gamma (ARCHITECTURE.md §11).

WHY THIS IS IMPLEMENTED DIRECTLY AND NOT VIA `lifetimes`
--------------------------------------------------------
PLAN.md §13 flagged `lifetimes` as a live dependency landmine: effectively
unmaintained and prone to breaking against current numpy/scipy. Rather than
pin an old scientific stack for the whole project, the two models are ~80 lines
of scipy here. They are also far easier to defend in a viva when you have
written the likelihood yourself.

BG/NBD (Fader, Hardie & Lee 2005) models WHEN a customer buys and when they
silently churn. Gamma-Gamma models HOW MUCH they spend, conditional on buying.
The two are independent by assumption — an assumption that is testable and is
tested below.

EVERY NUMBER HERE IS A PROJECTION. §11 and §16 require the word "projected"
rather than "measured", and the critic enforces it (POL-CLM-01). There is no
A/B test anywhere in DHAWQ.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.special import betaln, gammaln, hyp2f1


@dataclass(frozen=True)
class BGNBDParams:
    r: float; alpha: float; a: float; b: float
    converged: bool = True
    n_customers: int = 0


@dataclass(frozen=True)
class GammaGammaParams:
    p: float; q: float; v: float
    converged: bool = True
    population_mean: float = 0.0


def rfm_matrix(train: pl.DataFrame, as_of=None) -> pl.DataFrame:
    """The (frequency, recency, T, monetary_value) summary both models need.

    frequency x = REPEAT purchase occasions (first purchase excluded — the
    models condition on acquisition).
    recency t_x = age at last purchase.
    T          = total observed age.
    All in days, all inside the D1 window.
    """
    as_of = as_of or train.get_column("t_dat").max()
    g = train.group_by("customer_id").agg(
        pl.col("t_dat").min().alias("first"),
        pl.col("t_dat").max().alias("last"),
        pl.col("t_dat").n_unique().alias("occasions"),
        pl.col("price").sum().alias("total_value"),
    )
    return g.with_columns(
        (pl.col("occasions") - 1).cast(pl.Float64).alias("frequency"),
        (pl.col("last") - pl.col("first")).dt.total_days().cast(pl.Float64).alias("recency"),
        (pl.lit(as_of) - pl.col("first")).dt.total_days().cast(pl.Float64).alias("T"),
        (pl.col("total_value") / pl.col("occasions")).alias("monetary_value"),
    ).filter(pl.col("T") > 0)


# ── BG/NBD ───────────────────────────────────────────────────────────────────

def _bgnbd_nll(params, x, t_x, T):
    r, alpha, a, b = np.exp(params)          # log-space keeps all four positive
    ln_A1 = gammaln(r + x) - gammaln(r) + r * np.log(alpha)
    ln_A2 = betaln(a, b + x) - betaln(a, b)
    ln_A3 = -(r + x) * np.log(alpha + T)
    # The x>0 branch: a customer with repeat purchases may have churned at any
    # point after the last one, which the A4 term accounts for.
    ln_A4 = np.where(
        x > 0,
        np.log(a) - np.log(np.maximum(b + x - 1, 1e-10)) - (r + x) * np.log(alpha + t_x),
        -np.inf,
    )
    m = np.maximum(ln_A3, ln_A4)
    ll = ln_A1 + ln_A2 + m + np.log(
        np.exp(ln_A3 - m) + np.where(x > 0, np.exp(ln_A4 - m), 0.0)
    )
    return -np.sum(ll)


def fit_bgnbd(summary: pl.DataFrame, seed: int = 20260903) -> BGNBDParams:
    x = summary.get_column("frequency").to_numpy()
    t_x = summary.get_column("recency").to_numpy()
    T = summary.get_column("T").to_numpy()
    res = minimize(
        _bgnbd_nll, x0=np.log([1.0, 10.0, 1.0, 1.0]), args=(x, t_x, T),
        method="Nelder-Mead",
        options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-6},
    )
    r, alpha, a, b = np.exp(res.x)
    return BGNBDParams(float(r), float(alpha), float(a), float(b),
                       bool(res.success), len(x))


def expected_purchases(p: BGNBDParams, x, t_x, T, horizon_days: float) -> np.ndarray:
    """E[Y(t) | x, t_x, T] — projected repeat purchases over the horizon."""
    x, t_x, T = np.asarray(x, float), np.asarray(t_x, float), np.asarray(T, float)
    r, al, a, b = p.r, p.alpha, p.a, p.b
    hyp = hyp2f1(r + x, b + x, a + b + x - 1, horizon_days / (al + T + horizon_days))
    num = ((a + b + x - 1) / (a - 1)) * (
        1 - ((al + T) / (al + T + horizon_days)) ** (r + x) * hyp
    )
    denom = 1 + np.where(x > 0, (a / np.maximum(b + x - 1, 1e-10))
                         * ((al + T) / (al + t_x)) ** (r + x), 0.0)
    return np.maximum(num / denom, 0.0)


def probability_alive(p: BGNBDParams, x, t_x, T) -> np.ndarray:
    x, t_x, T = np.asarray(x, float), np.asarray(t_x, float), np.asarray(T, float)
    r, al, a, b = p.r, p.alpha, p.a, p.b
    return 1.0 / (1 + np.where(
        x > 0,
        (a / np.maximum(b + x - 1, 1e-10)) * ((al + T) / (al + t_x)) ** (r + x),
        0.0,
    ))


# ── Gamma-Gamma ──────────────────────────────────────────────────────────────

def _gg_nll(params, x, m):
    p, q, v = np.exp(params)
    return -np.sum(
        gammaln(p * x + q) - gammaln(p * x) - gammaln(q)
        + q * np.log(v) + (p * x - 1) * np.log(m) + (p * x) * np.log(x)
        - (p * x + q) * np.log(v + m * x)
    )


def fit_gamma_gamma(summary: pl.DataFrame) -> GammaGammaParams:
    """Fit on REPEAT buyers with positive spend only.

    The independence assumption between frequency and monetary value is checked
    by the caller; a correlation above ~0.3 invalidates the model and must be
    reported rather than ignored.
    """
    rep = summary.filter((pl.col("frequency") > 0) & (pl.col("monetary_value") > 0))
    x = rep.get_column("frequency").to_numpy()
    m = rep.get_column("monetary_value").to_numpy()
    res = minimize(_gg_nll, x0=np.log([1.0, 1.0, 1.0]), args=(x, m),
                   method="Nelder-Mead", options={"maxiter": 4000})
    p, q, v = np.exp(res.x)
    pop = float(v * p / (q - 1)) if q > 1 else float(np.mean(m))
    return GammaGammaParams(float(p), float(q), float(v), bool(res.success), pop)


def expected_value(gg: GammaGammaParams, x, m) -> np.ndarray:
    """Shrinkage toward the population mean.

    A customer with one observed basket gets a value close to the population
    average; one with twenty gets close to their own. This is the whole point
    of the Bayesian treatment — a naive per-customer mean would give a wildly
    confident estimate off a single purchase.
    """
    x, m = np.asarray(x, float), np.asarray(m, float)
    w = (gg.p * x) / (gg.p * x + gg.q - 1)
    return w * m + (1 - w) * gg.population_mean


def frequency_monetary_correlation(summary: pl.DataFrame) -> float:
    """Gamma-Gamma ASSUMES these are independent. Report the correlation so the
    assumption is visible rather than buried."""
    rep = summary.filter((pl.col("frequency") > 0) & (pl.col("monetary_value") > 0))
    if rep.height < 2:
        return 0.0
    return float(np.corrcoef(rep.get_column("frequency").to_numpy(),
                             rep.get_column("monetary_value").to_numpy())[0, 1])


def clv(summary: pl.DataFrame, bg: BGNBDParams, gg: GammaGammaParams,
        horizon_days: float = 84.0) -> pl.DataFrame:
    """PROJECTED CLV over the horizon. Never 'measured' — POL-CLM-01."""
    purchases = expected_purchases(
        bg, summary.get_column("frequency"), summary.get_column("recency"),
        summary.get_column("T"), horizon_days,
    )
    value = expected_value(gg, summary.get_column("frequency"),
                           summary.get_column("monetary_value"))
    alive = probability_alive(bg, summary.get_column("frequency"),
                              summary.get_column("recency"), summary.get_column("T"))
    return summary.select("customer_id").with_columns(
        pl.Series("projected_purchases", purchases),
        pl.Series("projected_value_per_purchase", value),
        pl.Series("probability_alive", alive),
        pl.Series("projected_clv", purchases * value),
    )
