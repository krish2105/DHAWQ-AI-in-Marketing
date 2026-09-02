"""Ranking quality metrics — ARCHITECTURE.md §9.

Pure functions over (recommended, relevant). No model, no state, no I/O — every
one is a deterministic function with unit tests, per the rule in §0.1. The
harness composes them; they know nothing about recommenders.

A NOTE ON WHAT THESE CAN AND CANNOT MEAN (§3, §16, §17)
------------------------------------------------------
The data has purchases, never impressions. An article the customer did not buy
is UNLABELLED, not rejected. So recall@k is a lower bound, precision@k is
depressed by items that were never shown, and every number here inherits that.
This is the single biggest limitation in the project and it is inherent to the
dataset, not to the method.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def precision_at_k(recommended: Sequence[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top = recommended[:k]
    return sum(1 for a in top if a in relevant) / k


def recall_at_k(recommended: Sequence[str], relevant: set[str], k: int) -> float:
    """Denominator is |relevant|, NOT min(k, |relevant|).

    Capping the denominator inflates the number for customers who bought more
    than k things, which is exactly the heavy-buyer segment already easiest to
    serve — so the cap would flatter the model precisely where it needs least
    help.
    """
    if not relevant:
        return 0.0
    return sum(1 for a in recommended[:k] if a in relevant) / len(relevant)


def average_precision_at_k(recommended: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits, total = 0, 0.0
    for i, a in enumerate(recommended[:k], start=1):
        if a in relevant:
            hits += 1
            total += hits / i
    return total / min(len(relevant), k)


def reciprocal_rank(recommended: Sequence[str], relevant: set[str]) -> float:
    for i, a in enumerate(recommended, start=1):
        if a in relevant:
            return 1.0 / i
    return 0.0


def dcg_at_k(recommended: Sequence[str], relevant: set[str], k: int) -> float:
    return sum(
        1.0 / math.log2(i + 1)
        for i, a in enumerate(recommended[:k], start=1)
        if a in relevant
    )


def ndcg_at_k(recommended: Sequence[str], relevant: set[str], k: int) -> float:
    """Binary-relevance NDCG.

    IDCG uses min(|relevant|, k), so a customer with two held-out purchases is
    scored against the best achievable ranking for two items — not against an
    unreachable ideal of k. Without that, NDCG would be structurally capped
    below 1.0 for light buyers and the cold-start curve in §9 would slope
    downward for a purely arithmetic reason.
    """
    if not relevant:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg_at_k(recommended, relevant, k) / idcg if idcg > 0 else 0.0


def hit_rate_at_k(recommended: Sequence[str], relevant: set[str], k: int) -> float:
    return 1.0 if any(a in relevant for a in recommended[:k]) else 0.0


METRICS = {
    "precision": precision_at_k,
    "recall": recall_at_k,
    "ndcg": ndcg_at_k,
    "map": average_precision_at_k,
    "hit_rate": hit_rate_at_k,
}


def evaluate_user(
    recommended: Sequence[str], relevant: set[str], ks: Sequence[int] = (5, 10, 20)
) -> dict[str, float]:
    out: dict[str, float] = {"mrr": reciprocal_rank(recommended, relevant)}
    for k in ks:
        for name, fn in METRICS.items():
            out[f"{name}@{k}"] = fn(recommended, relevant, k)
    return out
