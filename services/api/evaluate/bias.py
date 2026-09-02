"""Popularity bias and long-tail exposure — ARCHITECTURE.md §9.

"Does the model just re-rank bestsellers? Measure it explicitly."

The head/tail split is READ from the frozen catalogue definition, never
recomputed with a local rule of thumb. Corpus C's `head_and_tail` defines the
head as the top 20% of articles BY COUNT (not by units), computed once per
catalogue version and frozen — so the quota the optimiser enforces and the
exposure this module measures refer to the same set of articles.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl


def head_set(train: pl.DataFrame, head_share: float = 0.20) -> set[str]:
    """Top `head_share` of articles by units sold. Everything else is tail."""
    ranked = (
        train.group_by("article_id").len()
        .sort("len", descending=True)
        .get_column("article_id").to_list()
    )
    return set(ranked[: max(1, int(len(ranked) * head_share))])


def popularity_lift(
    all_slates: Sequence[Sequence[str]], train: pl.DataFrame
) -> float:
    """Mean training popularity of recommended items ÷ mean over the catalogue.

    1.0 means the model recommends articles of average popularity. 5.0 means it
    concentrates on articles five times more popular than average — which is a
    bestseller re-ranker wearing a personalisation label.
    """
    counts = train.group_by("article_id").len().rows_by_key("article_id", unique=True)
    pop = {a: c[0] for a, c in counts.items()}
    catalogue_mean = float(np.mean(list(pop.values()))) if pop else 0.0
    if catalogue_mean == 0:
        return 0.0
    rec = [pop.get(a, 0) for slate in all_slates for a in slate]
    return float(np.mean(rec) / catalogue_mean) if rec else 0.0


def head_share_of_impressions(
    all_slates: Sequence[Sequence[str]], head: set[str]
) -> float:
    total = sum(len(s) for s in all_slates)
    if total == 0:
        return 0.0
    return sum(1 for s in all_slates for a in s if a in head) / total


def concentration_curve(
    all_slates: Sequence[Sequence[str]], catalogue: Sequence[str], points: int = 20
) -> list[tuple[float, float]]:
    """Lorenz-style curve: (share of catalogue, share of impressions).

    This is what the merchandiser actually reads — "4% of articles take 60% of
    impressions" is a sentence a CMO acts on; a Gini coefficient is not.
    """
    from services.api.evaluate.beyond_accuracy import impression_counts

    counts = np.sort(impression_counts(all_slates, catalogue))[::-1]
    total = counts.sum()
    if total == 0:
        return []
    cum = np.cumsum(counts) / total
    n = len(counts)
    return [
        (round((i + 1) / n, 4), round(float(cum[i]), 4))
        for i in np.linspace(0, n - 1, points).astype(int)
    ]
