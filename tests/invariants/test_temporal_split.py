"""A1 — the temporal leak assertion.

PLAN.md §12 lists this under invariants rather than unit tests because it
guards an architectural claim, not a function. ARCHITECTURE.md §6 states
"Temporal split, never random. Assert max(train_date) < min(test_date) in a
test." This is that test.

It runs against the FROZEN PARQUET, not against synthetic data. A leak
assertion that only ever sees a fixture proves nothing about the artefact every
downstream session actually loads.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from pipelines.common import DATA_PROCESSED, MANIFEST_DIR, sha256_file

TRAIN = DATA_PROCESSED / "transactions_train.parquet"
TEST = DATA_PROCESSED / "transactions_test.parquet"
MANIFEST = MANIFEST_DIR / "subsample_v1.json"

pytestmark = pytest.mark.skipif(
    not (TRAIN.exists() and TEST.exists()),
    reason="D1 artefacts absent — run pipelines/01_subsample.py",
)


@pytest.fixture(scope="module")
def train() -> pl.DataFrame:
    return pl.read_parquet(TRAIN)


@pytest.fixture(scope="module")
def test_df() -> pl.DataFrame:
    return pl.read_parquet(TEST)


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def test_A1_no_temporal_leak(train, test_df):
    """The one that matters. A random split lets the model see the future and
    inflates every ranking metric in §9."""
    max_train = train.get_column("t_dat").max()
    min_test = test_df.get_column("t_dat").min()
    assert max_train < min_test, (
        f"TEMPORAL LEAK: max(train)={max_train} >= min(test)={min_test}"
    )


def test_A1_splits_are_non_empty(train, test_df):
    """A leak assertion passes trivially if one side is empty."""
    assert train.height > 0
    assert test_df.height > 0


def test_A1_no_date_appears_in_both_splits(train, test_df):
    """Stronger than max < min: no single day may straddle the boundary."""
    overlap = set(train.get_column("t_dat").unique()) & set(
        test_df.get_column("t_dat").unique()
    )
    assert not overlap, f"dates in both splits: {sorted(overlap)[:5]}"


def test_A1_split_matches_manifest(train, test_df, manifest):
    """The boundary in the data is the boundary the manifest claims."""
    split = manifest["window"]["split_date"]
    assert str(train.get_column("t_dat").max()) < split
    assert str(test_df.get_column("t_dat").min()) >= split


def test_A2_every_test_article_survived_the_filter(train, test_df, manifest):
    """PLAN.md §7 A2 as actually written: test articles are a subset of the
    KEPT set. This is the real no-leakage claim — nothing enters the test split
    that the subsampling rule excluded."""
    articles = pl.read_parquet(DATA_PROCESSED / "articles.parquet")
    kept = set(articles.get_column("article_id"))
    assert set(test_df.get_column("article_id").unique()) <= kept


def test_cold_articles_are_recorded_not_hidden(train, test_df, manifest):
    """Articles present only in test are legitimate — all their purchases land
    in the final two weeks. Collaborative cannot score them, so they MUST be a
    declared stratum rather than silent contamination of an aggregate metric.

    This test does not assert the count is zero. It asserts the manifest tells
    the truth about it, so §9 cannot report a blended NDCG by accident.
    """
    observed = set(test_df.get_column("article_id").unique()) - set(
        train.get_column("article_id").unique()
    )
    recorded = manifest["strata"]["cold_articles"]
    assert recorded["n"] == len(observed), "manifest understates the cold-article stratum"
    if observed:
        assert recorded["share_of_test_rows"] > 0


def test_A3_A4_counts_within_recorded_bounds(manifest):
    from pipelines.subsample import EXPECT_ARTICLES, EXPECT_CUSTOMERS

    counts = manifest["counts"]
    lo, hi = EXPECT_ARTICLES
    assert lo <= counts["articles"] <= hi
    lo, hi = EXPECT_CUSTOMERS
    assert lo <= counts["customers"] <= hi


def test_A6_fixed_point_actually_converged(manifest):
    """Not 'ran to the cap' — genuinely stopped moving. The last two iterations
    must be identical."""
    fp = manifest["fixed_point"]
    assert fp["converged"] is True
    trace = fp["trace"]
    assert len(trace) >= 2
    assert (trace[-1]["articles"], trace[-1]["customers"]) == (
        trace[-2]["articles"], trace[-2]["customers"]
    )


def test_A7_manifest_hashes_match_disk(manifest):
    """The manifest describes the bytes actually on disk. Without this, every
    downstream session could be reading a different artefact than the one the
    manifest documents."""
    for name, rec in manifest["outputs"].items():
        path = DATA_PROCESSED / f"{name}.parquet"
        assert sha256_file(path) == rec["sha256"], f"{name}.parquet has drifted"


def test_support_thresholds_hold_in_the_output(train, test_df, manifest):
    """The fixed point's whole purpose: the frozen artefact must satisfy the
    rule it claims to enforce. A single-pass filter fails exactly here."""
    p = manifest["parameters"]
    allt = pl.concat([train, test_df])

    per_article = allt.group_by("article_id").len().get_column("len").min()
    assert per_article >= p["min_article_purchases"], (
        f"an article has {per_article} purchases, below the stated floor"
    )

    per_customer = (
        allt.group_by("customer_id")
        .agg(pl.col("t_dat").n_unique().alias("b"))
        .get_column("b").min()
    )
    assert per_customer >= p["min_customer_baskets"], (
        f"a customer has {per_customer} shopping occasions, below the stated floor"
    )
