"""Unit tests for the ranking metrics.

ARCHITECTURE.md §7.1: the deterministic core is tested by unit and property
tests covering 100% of decision paths. These are the functions every headline
number in §9 is built from, so a silent bug here would corrupt the entire
graded contribution.
"""

from __future__ import annotations

import math

import pytest

from services.api.evaluate.ranking import (
    average_precision_at_k,
    dcg_at_k,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

REC = ["a", "b", "c", "d", "e"]


# ── precision ────────────────────────────────────────────────────────────────

def test_precision_counts_hits_over_k():
    assert precision_at_k(REC, {"a", "c"}, 5) == pytest.approx(0.4)


def test_precision_denominator_is_k_not_list_length():
    """k=10 on a 5-item list must be 2/10, not 2/5. Using the list length would
    silently reward a model that returns fewer items."""
    assert precision_at_k(REC, {"a", "c"}, 10) == pytest.approx(0.2)


def test_precision_of_empty_or_zero_k():
    assert precision_at_k([], {"a"}, 5) == 0.0
    assert precision_at_k(REC, {"a"}, 0) == 0.0


# ── recall ───────────────────────────────────────────────────────────────────

def test_recall_denominator_is_all_relevant_not_capped_at_k():
    """A customer with 10 held-out purchases, 2 found in the top-2.

    Capping the denominator at k would give 1.0 — a perfect score for finding
    a fifth of what they bought. That inflation lands hardest on heavy buyers,
    the segment already easiest to serve.
    """
    relevant = {f"r{i}" for i in range(10)} | {"a", "b"}
    assert recall_at_k(["a", "b"], relevant, 2) == pytest.approx(2 / 12)


def test_recall_with_no_relevant_items_is_zero_not_nan():
    assert recall_at_k(REC, set(), 5) == 0.0


# ── NDCG ─────────────────────────────────────────────────────────────────────

def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k(["a", "b"], {"a", "b"}, 5) == pytest.approx(1.0)


def test_ndcg_reaches_one_for_light_buyers():
    """THE bug this guards: if IDCG used k instead of min(|relevant|, k), a
    customer with one held-out purchase ranked first would score ~0.4 at k=10,
    and the §9 cold-start curve would slope down for a purely arithmetic
    reason that has nothing to do with the model."""
    assert ndcg_at_k(["a"] + [f"x{i}" for i in range(9)], {"a"}, 10) == pytest.approx(1.0)


def test_ndcg_rewards_higher_positions():
    early = ndcg_at_k(["a", "x", "y"], {"a"}, 3)
    late = ndcg_at_k(["x", "y", "a"], {"a"}, 3)
    assert early > late


def test_ndcg_is_zero_with_no_relevant():
    assert ndcg_at_k(REC, set(), 5) == 0.0


def test_dcg_uses_log2_of_rank_plus_one():
    assert dcg_at_k(["a", "b"], {"a", "b"}, 2) == pytest.approx(
        1 / math.log2(2) + 1 / math.log2(3)
    )


# ── MAP / MRR ────────────────────────────────────────────────────────────────

def test_average_precision_rewards_early_clustering():
    front = average_precision_at_k(["a", "b", "x", "y"], {"a", "b"}, 4)
    spread = average_precision_at_k(["a", "x", "y", "b"], {"a", "b"}, 4)
    assert front > spread
    assert front == pytest.approx(1.0)


def test_reciprocal_rank_is_inverse_of_first_hit():
    assert reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(["x", "y"], {"a"}) == 0.0


def test_hit_rate_is_binary():
    assert hit_rate_at_k(REC, {"c"}, 5) == 1.0
    assert hit_rate_at_k(REC, {"z"}, 5) == 0.0
    assert hit_rate_at_k(REC, {"e"}, 3) == 0.0


# ── monotonicity property ────────────────────────────────────────────────────

@pytest.mark.parametrize("metric", [recall_at_k, ndcg_at_k, hit_rate_at_k])
def test_metrics_are_monotone_non_decreasing_in_k(metric):
    """Looking further down the list can never find fewer relevant items.
    Precision is excluded: it legitimately falls as k grows."""
    relevant = {"c", "e"}
    vals = [metric(REC, relevant, k) for k in (1, 2, 3, 4, 5)]
    assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:]))


# ── the cold-article contract ────────────────────────────────────────────────

def test_arms_declare_only_what_is_structurally_true():
    """`can_score_cold_articles` is about whether a finite score comes back,
    NOT about whether cold articles reach a slate.

    Conflating the two produced a real bug: the weighted hybrid declared
    cold-article support and delivered zero impressions. Surfacing is measured
    in the eval artefact, never declared here.
    """
    from services.api.models.baseline import PopularityRecency
    from services.api.models.collaborative import ImplicitALS
    from services.api.models.content import ContentKNN
    from services.api.models.hybrid import Hybrid

    assert ImplicitALS().can_score_cold_articles is False
    assert ContentKNN().can_score_cold_articles is True
    assert Hybrid(mode="weighted").can_score_cold_articles is True
    assert PopularityRecency().can_score_cold_articles is True
    assert not hasattr(ContentKNN(), "handles_cold_articles"), (
        "the ambiguous name must not come back"
    )
