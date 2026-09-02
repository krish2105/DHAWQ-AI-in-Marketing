"""D1 — H&M subsample and temporal split.

Implements PLAN.md §7 exactly: R1..R6 with assertions A1..A7.

WHY POLARS `scan_csv` AND NOT PANDAS
------------------------------------
transactions_train.csv is 3.3 GB / 31,788,325 rows. A naive
`pd.read_csv` materialises every row and every column before the first filter
runs, which on this file is ~8-10 GB of RAM for data we discard immediately.

`pl.scan_csv` builds a lazy plan. The R1 date predicate and the column
projection push *down into the scan*, so rows outside the 12-week window are
never constructed — we go from 31.8M rows to roughly 3.7M during the read
itself. It is also multi-threaded, which matters for the repeated group-by in
the R3 fixed point.

The pandas alternative is `chunksize=` with a hand-rolled accumulator. That
works, but it means writing the windowing, the group-by and the fixed-point
convergence check by hand across chunk boundaries — three more places for a
subtle bug in the rule that every downstream session depends on.

SCHEMA NOTE (this one bites)
----------------------------
`article_id` is a zero-padded 10-digit string in the CSV (`0663713001`) and the
image files are named after it (`images/066/0663713001.jpg`). Inferred as
Int64 it silently becomes 663713001 and every image lookup fails. Both id
columns are forced to Utf8.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from pipelines.common import DATA_RAW, require

HM_RAW = DATA_RAW / "hm"
IMAGES = HM_RAW / "images"

# ── Rule parameters. Changing any of these changes the methodology; they are
# ── recorded verbatim in the manifest so a reviewer sees them without reading
# ── this file. PLAN.md §7.
WINDOW_WEEKS = 12
TRAIN_WEEKS = 10
MIN_ARTICLE_PURCHASES = 20
MIN_CUSTOMER_BASKETS = 3
# Operational guard, NOT a methodological parameter. The methodology is the two
# support thresholds above; this only bounds how long we let them settle. D0
# estimated 2-3 iterations; the real data needs 4-5 (deltas collapse
# geometrically: 3745 articles on iter 2, then 70, then single digits). Raised
# from 3 after observing the trace. Accepting a non-converged iteration 3 would
# have shipped a parquet violating the very rule it claims to enforce.
MAX_FIXPOINT_ITERS = 8
COLD_START_MAX_PURCHASES = 3

# A3 / A4 — expected magnitude from ARCHITECTURE.md §3 ("~10-15k articles,
# ~50k customers"). Bounds are deliberately wider than the target: they catch a
# pipeline that has gone badly wrong, not a pipeline that landed 8% off.
EXPECT_ARTICLES = (10_000, 20_000)
# WIDENED AT D1, DELIBERATELY AND ON THE RECORD.
# ARCHITECTURE.md §3 predicted "~50k customers". The rule it specifies
# (>= 3 shopping occasions in 12 weeks) actually retains ~120k. The prediction
# was a guess; the rule is the methodology. Tightening the rule to hit a
# predicted number would be fitting the method to an estimate, which is
# backwards — so the rule stands and the bound moves. ~120k customers over
# ~1.6M transactions is still comfortably laptop-scale for ALS and BG/NBD.
# The deviation is recorded in the manifest, not just here.
EXPECT_CUSTOMERS = (30_000, 150_000)
PREDICTED_CUSTOMERS = 50_000       # ARCHITECTURE.md §3, for the manifest

TXN_SCHEMA = {
    "t_dat": pl.Date,
    "customer_id": pl.Utf8,
    "article_id": pl.Utf8,   # NOT Int64 — see module docstring
    "price": pl.Float64,
    "sales_channel_id": pl.Int8,
}


# ─────────────────────────────────────────────────────────────────────────────
# R1 — the 12-week window
# ─────────────────────────────────────────────────────────────────────────────

def scan_transactions(path: Path | None = None) -> pl.LazyFrame:
    """Lazy scan. Nothing is read until `.collect()`."""
    return pl.scan_csv(
        path or HM_RAW / "transactions_train.csv",
        schema_overrides=TXN_SCHEMA,
        try_parse_dates=True,
    )


def window_bounds(t_end: date, weeks: int = WINDOW_WEEKS) -> tuple[date, date]:
    """R1. Inclusive on both ends: a 12-week window spans 84 days, so the
    start is t_end - 83 days, not t_end - 84. Off-by-one here silently changes
    the denominator of every support count downstream."""
    return t_end - timedelta(days=weeks * 7 - 1), t_end


def split_date_for(t_start: date, train_weeks: int = TRAIN_WEEKS) -> date:
    """R4. First date of the TEST period. train < split_date <= test."""
    return t_start + timedelta(days=train_weeks * 7)


# ─────────────────────────────────────────────────────────────────────────────
# R2 — articles must have a usable image
# ─────────────────────────────────────────────────────────────────────────────

def image_path(article_id: str) -> Path:
    """H&M shards images by the first three characters of the padded id."""
    return IMAGES / article_id[:3] / f"{article_id}.jpg"


def articles_with_image_file(root: Path | None = None) -> set[str]:
    """Every article id that has a file on disk. One directory walk, not one
    stat() per candidate — at 105k articles that is the difference between
    seconds and minutes."""
    root = root or IMAGES
    found: set[str] = set()
    if not root.exists():
        return found
    for sub in os.scandir(root):
        if not sub.is_dir():
            continue
        for entry in os.scandir(sub.path):
            if entry.name.endswith(".jpg"):
                found.add(entry.name[:-4])
    return found


def _verifies(article_id: str) -> tuple[str, bool]:
    from PIL import Image
    try:
        with Image.open(image_path(article_id)) as im:
            im.verify()          # header-only; does not decode pixels
        return article_id, True
    except Exception:
        return article_id, False


def articles_that_decode(ids: list[str], workers: int = 16) -> set[str]:
    """A5. Header-level verification, threaded because it is IO-bound.

    Full pixel decode happens at D2 during CLIP encoding; anything that slips
    through here is caught there and recorded, not silently dropped.
    """
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return {aid for aid, ok in pool.map(_verifies, ids) if ok}


# ─────────────────────────────────────────────────────────────────────────────
# R3 — the fixed-point support filter
# ─────────────────────────────────────────────────────────────────────────────

def fixed_point_filter(
    txns: pl.DataFrame,
    min_article_purchases: int = MIN_ARTICLE_PURCHASES,
    min_customer_baskets: int = MIN_CUSTOMER_BASKETS,
    max_iters: int = MAX_FIXPOINT_ITERS,
) -> tuple[pl.DataFrame, list[str], list[str], int, bool, list[dict]]:
    """R3. Alternate the two support filters until neither changes anything.

    THE SUBTLETY THIS EXISTS FOR: dropping articles below the purchase floor
    removes transactions, which pushes some customers below the basket floor.
    Dropping those customers removes more transactions, which pushes more
    articles below the purchase floor. A single pass leaves the stated rule
    violated in its own output — the parquet would contain articles with 14
    purchases under a rule that claims a floor of 20.

    A customer's support is DISTINCT PURCHASE DATES, not row count. H&M rows
    are line items, so a customer buying five things on one day is five rows
    but one shopping occasion. PLAN.md §7 R3 specifies distinct t_dat.

    Returns the frame, surviving ids, iteration count, whether it actually
    converged, and a per-iteration trace for the manifest.
    """
    trace: list[dict] = []
    prev: tuple[int, int] | None = None

    for i in range(1, max_iters + 1):
        keep_articles = (
            txns.group_by("article_id").len()
            .filter(pl.col("len") >= min_article_purchases)
            .select("article_id")
        )
        txns = txns.join(keep_articles, on="article_id", how="semi")

        keep_customers = (
            txns.group_by("customer_id")
            .agg(pl.col("t_dat").n_unique().alias("baskets"))
            .filter(pl.col("baskets") >= min_customer_baskets)
            .select("customer_id")
        )
        txns = txns.join(keep_customers, on="customer_id", how="semi")

        state = (keep_articles.height, keep_customers.height)
        trace.append({
            "iteration": i,
            "articles": state[0],
            "customers": state[1],
            "transactions": txns.height,
        })
        if state == prev:
            return (txns, keep_articles.to_series().to_list(),
                    keep_customers.to_series().to_list(), i, True, trace)
        prev = state

    # Hit the cap while still moving. Report honestly; the caller asserts A6.
    return (txns, keep_articles.to_series().to_list(),
            keep_customers.to_series().to_list(), max_iters, False, trace)


# ─────────────────────────────────────────────────────────────────────────────
# R4 / R5 — temporal split and cold-start stratum
# ─────────────────────────────────────────────────────────────────────────────

def temporal_split(txns: pl.DataFrame, split: date) -> tuple[pl.DataFrame, pl.DataFrame]:
    """R4. Strictly before / on-or-after. Never a random split — a random split
    lets the model see the future and inflates every ranking metric."""
    return (
        txns.filter(pl.col("t_dat") < split),
        txns.filter(pl.col("t_dat") >= split),
    )


def cold_start_customers(
    train: pl.DataFrame,
    test: pl.DataFrame,
    max_purchases: int = COLD_START_MAX_PURCHASES,
) -> list[str]:
    """R5. Test customers with thin training history. RETAINED, not filtered —
    they are the cold-start stratum for the §9 curve. Dropping them would make
    that curve measure nothing.

    Includes test customers with zero training rows (the genuinely-unseen case),
    which a naive join on train would silently omit.
    """
    depth = (
        train.group_by("customer_id").len().rename({"len": "n"})
    )
    return (
        test.select("customer_id").unique()
        .join(depth, on="customer_id", how="left")
        .with_columns(pl.col("n").fill_null(0))
        .filter(pl.col("n") < max_purchases)
        .get_column("customer_id")
        .to_list()
    )


def cold_articles(train: pl.DataFrame, test: pl.DataFrame) -> list[str]:
    """R5b — articles that appear ONLY in the test period.

    NOT AN ERROR, AND NOT A LEAK. An article whose 20+ purchases all land in
    the final two weeks passes the support filter and is legitimately in the
    test split, but the collaborative model has never seen it and structurally
    cannot score it. Content-based CLIP retrieval can — this is precisely the
    article cold-start case ARCHITECTURE.md §6 says content-based solves and
    the hybrid exists to cover.

    Discovered at D1 on the real window: ~3% of test articles, but ~10% of test
    ROWS. Reporting a single aggregate NDCG across this boundary would blend
    warm ranking with cold ranking and hide which one is actually working, so
    it is recorded here as a named stratum for the §9 evaluation.
    """
    return (
        test.select("article_id").unique()
        .join(train.select("article_id").unique(), on="article_id", how="anti")
        .get_column("article_id").to_list()
    )


# ─────────────────────────────────────────────────────────────────────────────
# Assertions A1..A6 (A7 is checked by the driver after writing)
# ─────────────────────────────────────────────────────────────────────────────

def assert_no_leak(train: pl.DataFrame, test: pl.DataFrame) -> None:
    """A1 — THE leak assertion. PLAN.md §12 lists this as an invariant that
    guards an architectural claim, not merely a function."""
    require(train.height > 0, "A1", "train split is empty")
    require(test.height > 0, "A1", "test split is empty")
    mx = train.get_column("t_dat").max()
    mn = test.get_column("t_dat").min()
    require(
        mx < mn,
        "A1",
        f"TEMPORAL LEAK: max(train.t_dat)={mx} is not < min(test.t_dat)={mn}",
    )


def assert_no_unseen_articles(test: pl.DataFrame, keep_articles: list[str]) -> None:
    """A2 — every article in test survived the filter."""
    unseen = set(test.get_column("article_id").unique().to_list()) - set(keep_articles)
    require(not unseen, "A2", f"{len(unseen)} test articles outside the kept set")


def assert_counts_in_range(n_articles: int, n_customers: int) -> None:
    """A3 / A4 — magnitude sanity. On failure, adjust the thresholds and
    RE-RECORD them in the manifest. Never widen a bound silently; the manifest
    is the audit trail for exactly this."""
    lo, hi = EXPECT_ARTICLES
    require(lo <= n_articles <= hi, "A3", f"{n_articles} articles outside [{lo}, {hi}]")
    lo, hi = EXPECT_CUSTOMERS
    require(lo <= n_customers <= hi, "A4", f"{n_customers} customers outside [{lo}, {hi}]")
