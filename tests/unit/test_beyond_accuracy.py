"""Unit tests for beyond-accuracy metrics — where the marketing finding lives."""

from __future__ import annotations

import numpy as np
import pytest

from services.api.evaluate.beyond_accuracy import (
    catalogue_coverage,
    gini,
    intra_list_diversity,
    long_tail_exposure,
    novelty,
)


def test_coverage_counts_distinct_articles_across_all_slates():
    assert catalogue_coverage([["a", "b"], ["b", "c"]], 10) == pytest.approx(0.3)


def test_coverage_of_nothing_is_zero():
    assert catalogue_coverage([], 10) == 0.0
    assert catalogue_coverage([["a"]], 0) == 0.0


def test_gini_is_zero_for_perfectly_even_exposure():
    assert gini([5, 5, 5, 5]) == pytest.approx(0.0, abs=1e-9)


def test_gini_approaches_one_for_total_concentration():
    assert gini([0] * 999 + [1000]) > 0.99


def test_gini_includes_never_recommended_articles():
    """Concentration measured only over winners misses the concentration that
    matters. Adding unrecommended articles must RAISE Gini."""
    assert gini([10, 10, 10] + [0] * 97) > gini([10, 10, 10])


def test_long_tail_exposure_is_share_outside_the_head():
    assert long_tail_exposure([["h1", "h2", "t1", "t2"]], head={"h1", "h2"}) == pytest.approx(0.5)


def test_intra_list_diversity_of_identical_items_is_zero():
    emb = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    assert intra_list_diversity(["a", "b"], emb, {"a": 0, "b": 1}) == pytest.approx(0.0, abs=1e-6)


def test_intra_list_diversity_of_orthogonal_items_is_one():
    emb = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    assert intra_list_diversity(["a", "b"], emb, {"a": 0, "b": 1}) == pytest.approx(1.0, abs=1e-6)


def test_intra_list_diversity_needs_at_least_two_items():
    emb = np.array([[1.0, 0.0]], dtype=np.float32)
    assert intra_list_diversity(["a"], emb, {"a": 0}) == 0.0


def test_novelty_is_higher_for_rarer_items():
    pop = {"common": 900, "rare": 1}
    assert novelty([["rare"]], pop, 1000) > novelty([["common"]], pop, 1000)
