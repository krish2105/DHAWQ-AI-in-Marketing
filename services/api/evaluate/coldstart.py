"""Cold-start stratification — ARCHITECTURE.md §9.

"Stratify every metric by user history depth: 0, 1-2, 3-9, 10+. Report the
curve. Personalisation that only works for heavy buyers is a known and
important limitation."

D1 also surfaced an ARTICLE-side stratum: 348 articles (9.9% of test rows)
appear only in the test period. The collaborative arm cannot score them at all.
Both axes are stratified here, because a single aggregate NDCG blends warm and
cold ranking and hides which one is actually working.
"""

from __future__ import annotations

import polars as pl

USER_BUCKETS: list[tuple[str, int, int]] = [
    ("0", 0, 0),
    ("1-2", 1, 2),
    ("3-9", 3, 9),
    ("10+", 10, 10**9),
]


def user_history_depth(train: pl.DataFrame) -> dict[str, int]:
    """Depth = distinct shopping occasions, matching the D1 support rule so the
    two never disagree about what a 'purchase' is."""
    g = train.group_by("customer_id").agg(pl.col("t_dat").n_unique().alias("n"))
    return {c: int(n) for c, n in zip(g.get_column("customer_id"), g.get_column("n"))}


def bucket_for(depth: int) -> str:
    for label, lo, hi in USER_BUCKETS:
        if lo <= depth <= hi:
            return label
    return "10+"


def stratify_users(customer_ids: list[str], depths: dict[str, int]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {label: [] for label, _, _ in USER_BUCKETS}
    for c in customer_ids:
        out[bucket_for(depths.get(c, 0))].append(c)
    return out


def cold_article_set(train: pl.DataFrame, test: pl.DataFrame) -> set[str]:
    """Articles in test with no training interaction. Mirrors pipelines
    R5b — duplicated deliberately so the API layer never imports pipelines."""
    return set(test.get_column("article_id").unique()) - set(
        train.get_column("article_id").unique()
    )
