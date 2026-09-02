"""Unit tests for the D1 rule functions.

These use synthetic frames so each rule is exercised in isolation, including
the edge cases the real data happens not to contain. The invariant tests cover
the real artefact; these cover the logic.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from pipelines.subsample import (
    cold_start_customers,
    fixed_point_filter,
    image_path,
    split_date_for,
    temporal_split,
    window_bounds,
)


def _txns(rows):
    return pl.DataFrame(
        rows, schema={"t_dat": pl.Date, "customer_id": pl.Utf8, "article_id": pl.Utf8},
        orient="row",
    )


# ── R1 ───────────────────────────────────────────────────────────────────────

def test_window_is_84_days_inclusive():
    """12 weeks inclusive spans 84 days, so start = end - 83. Off by one here
    silently changes the denominator of every support count."""
    start, end = window_bounds(date(2020, 9, 22))
    assert (end - start).days == 83
    assert start == date(2020, 7, 1)


def test_split_gives_ten_weeks_train_two_weeks_test():
    start, end = window_bounds(date(2020, 9, 22))
    split = split_date_for(start)
    assert (split - start).days == 70
    assert (end - split).days == 13


# ── R2 ───────────────────────────────────────────────────────────────────────

def test_image_path_shards_on_first_three_chars():
    """H&M shards by the first 3 chars of the ZERO-PADDED id. If article_id is
    ever read as an int the leading zero vanishes and every lookup fails."""
    assert image_path("0663713001").parts[-2:] == ("066", "0663713001.jpg")


# ── R3 — the fixed point ─────────────────────────────────────────────────────

def test_fixed_point_output_satisfies_its_own_rule():
    """THE regression test for the single-pass bug.

    Article A2 has exactly 2 purchases and is dropped. Both its purchases
    belong to customer C1, whose remaining occasions then fall below the floor.
    A single pass would keep C1 and violate the stated rule in its own output.
    """
    rows = []
    for d in range(1, 4):                       # C1: 3 occasions, 2 via A2
        rows.append((date(2020, 7, d), "C1", "A1" if d == 1 else "A2"))
    for c in ("C2", "C3", "C4"):                # three healthy customers on A1
        for d in range(1, 4):
            rows.append((date(2020, 7, d), c, "A1"))

    out, arts, custs, iters, converged, trace = fixed_point_filter(
        _txns(rows), min_article_purchases=3, min_customer_baskets=3
    )
    assert converged
    assert "A2" not in arts
    assert "C1" not in custs, "single-pass bug: C1 survived on dropped-article purchases"

    per_cust = out.group_by("customer_id").agg(pl.col("t_dat").n_unique().alias("b"))
    assert per_cust.get_column("b").min() >= 3
    assert out.group_by("article_id").len().get_column("len").min() >= 3


def test_customer_support_counts_occasions_not_line_items():
    """PLAN.md §7 R3: distinct t_dat. A customer buying 5 items on one day is
    5 rows but ONE shopping occasion, and must not clear a floor of 3."""
    rows = [(date(2020, 7, 1), "C1", f"A{i}") for i in range(6)]
    _, _, custs, *_ = fixed_point_filter(
        _txns(rows), min_article_purchases=1, min_customer_baskets=3
    )
    assert "C1" not in custs


def test_fixed_point_reports_non_convergence_rather_than_lying():
    """Capped mid-descent must return converged=False. Reporting True here is
    how a violating parquet ships."""
    rows = [
        (date(2020, 7, (i % 20) + 1), f"C{i}", f"A{i // 3}") for i in range(60)
    ]
    *_, converged, trace = fixed_point_filter(
        _txns(rows), min_article_purchases=50, min_customer_baskets=50, max_iters=1
    )
    assert converged is False
    assert len(trace) == 1


def test_trace_records_every_iteration():
    rows = [(date(2020, 7, d), f"C{c}", "A1") for c in range(5) for d in range(1, 4)]
    *_, iters, converged, trace = fixed_point_filter(
        _txns(rows), min_article_purchases=1, min_customer_baskets=3
    )
    assert converged and len(trace) == iters


# ── R4 / R5 ──────────────────────────────────────────────────────────────────

def test_temporal_split_boundary_is_half_open():
    """split_date belongs to TEST, not train."""
    rows = [(date(2020, 9, 8), "C1", "A1"), (date(2020, 9, 9), "C1", "A1")]
    train, test = temporal_split(_txns(rows), date(2020, 9, 9))
    assert train.height == 1 and test.height == 1
    assert train.get_column("t_dat").max() < test.get_column("t_dat").min()


def test_cold_start_includes_customers_with_no_training_history():
    """The genuinely-unseen case. An inner join would silently omit C2, which
    is the most important cold-start row there is."""
    train = _txns([(date(2020, 7, 1), "C1", "A1")])
    test = _txns([(date(2020, 9, 9), "C1", "A1"), (date(2020, 9, 9), "C2", "A1")])
    cold = cold_start_customers(train, test, max_purchases=3)
    assert set(cold) == {"C1", "C2"}


def test_cold_start_excludes_deep_history_customers():
    train = _txns([(date(2020, 7, d), "C1", "A1") for d in range(1, 6)])
    test = _txns([(date(2020, 9, 9), "C1", "A1")])
    assert cold_start_customers(train, test, max_purchases=3) == []
