"""Beyond-accuracy metrics — "the part most projects skip" (ARCHITECTURE.md §9).

This is where the marketing finding lives. Accuracy says whether the ranking is
good; these say what it COSTS. A model that wins NDCG while collapsing coverage
to 4% of the catalogue is a merchandising problem, and naming that trade is the
distinction-level observation the project is built around.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np


def catalogue_coverage(all_slates: Sequence[Sequence[str]], n_catalogue: int) -> float:
    """Share of the catalogue that is ever recommended to anyone.

    Low coverage means dead inventory: articles the retailer paid for and the
    recommender will never show to a single customer.
    """
    if n_catalogue <= 0:
        return 0.0
    return len({a for slate in all_slates for a in slate}) / n_catalogue


def gini(counts: Sequence[float]) -> float:
    """Gini over impression counts. 0 = perfectly even, 1 = one article takes
    everything.

    Computed over the FULL catalogue including never-recommended articles
    (count 0). Computing it only over recommended items would measure
    concentration within the winners and miss the concentration that matters.
    """
    x = np.sort(np.asarray(counts, dtype=np.float64))
    n = x.size
    if n == 0 or x.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * x).sum()) / (n * x.sum()) - (n + 1) / n)


def impression_counts(all_slates: Sequence[Sequence[str]], catalogue: Sequence[str]) -> np.ndarray:
    c = Counter(a for slate in all_slates for a in slate)
    return np.fromiter((c.get(a, 0) for a in catalogue), dtype=np.float64,
                       count=len(catalogue))


def long_tail_exposure(
    all_slates: Sequence[Sequence[str]], head: set[str]
) -> float:
    """Share of impressions landing on articles OUTSIDE the head.

    `head` comes from the frozen catalogue definition in corpus C
    (`head_and_tail`), not recomputed here — a quota that drifts between the
    optimiser and the metric is not a quota.
    """
    total = sum(len(s) for s in all_slates)
    if total == 0:
        return 0.0
    return sum(1 for s in all_slates for a in s if a not in head) / total


def intra_list_diversity(slate: Sequence[str], emb: np.ndarray, index: dict[str, int]) -> float:
    """1 - mean pairwise cosine over the slate's CLIP embeddings.

    Matches the corpus C definition exactly (`intra_list_diversity`), because
    POL-DIV-01 is enforced against this number and the critic and the metric
    must not disagree about what it means.
    """
    rows = [index[a] for a in slate if a in index]
    if len(rows) < 2:
        return 0.0
    v = emb[rows]                              # already unit-norm
    sims = v @ v.T
    n = len(rows)
    off = (sims.sum() - np.trace(sims)) / (n * (n - 1))
    return float(1.0 - off)


def novelty(all_slates: Sequence[Sequence[str]], popularity: dict[str, int],
            n_interactions: int) -> float:
    """Mean self-information of recommended items: -log2(p(item)).

    High novelty means the system is surfacing things a customer is unlikely to
    have met already.
    """
    if n_interactions <= 0:
        return 0.0
    vals = []
    for slate in all_slates:
        for a in slate:
            p = popularity.get(a, 0) / n_interactions
            vals.append(-np.log2(p) if p > 0 else 0.0)
    return float(np.mean(vals)) if vals else 0.0


def serendipity(
    all_slates: dict[str, Sequence[str]],
    relevant: dict[str, set[str]],
    baseline_slates: dict[str, Sequence[str]],
) -> float:
    """Relevant AND unexpected: hits the baseline would not have produced.

    §13 of PLAN flagged this as the weakest of the six — there is no agreed
    formula. The definition used is stated here rather than left implicit:
    a hit that the popularity baseline did not also surface. Reported with that
    caveat attached, never as a bare number.
    """
    num = den = 0
    for cid, slate in all_slates.items():
        rel = relevant.get(cid, set())
        if not rel:
            continue
        base = set(baseline_slates.get(cid, ()))
        num += sum(1 for a in slate if a in rel and a not in base)
        den += len(slate)
    return num / den if den else 0.0
