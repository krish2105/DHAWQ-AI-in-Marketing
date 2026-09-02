"""RFM segmentation — ARCHITECTURE.md §11. Standard, expected, cheap.

Quintile scoring on Recency, Frequency, Monetary. Quintiles rather than fixed
thresholds because the price field is normalised with no currency (§3), so any
absolute cut-off would be meaningless. Segment labels follow the conventional
retail taxonomy so a merchandiser recognises them without a legend.
"""

from __future__ import annotations

from datetime import date

import polars as pl

SEGMENTS: list[tuple[str, str]] = [
    ("champions",            "R>=4 and F>=4"),
    ("loyal",                "F>=4"),
    ("big_spenders",         "M>=4"),
    ("promising",            "R>=4 and F<=2"),
    ("at_risk",              "R<=2 and F>=3"),
    ("hibernating",          "R<=2 and F<=2"),
    ("needs_attention",      "otherwise"),
]


def rfm_table(train: pl.DataFrame, as_of: date | None = None) -> pl.DataFrame:
    """Per-customer R/F/M with quintile scores and a segment label.

    Frequency is DISTINCT SHOPPING OCCASIONS, consistent with the D1 support
    rule. Counting line items would rank a customer who bought five things once
    above one who returned four separate times, which inverts the loyalty
    signal the segmentation exists to capture.
    """
    as_of = as_of or train.get_column("t_dat").max()

    base = train.group_by("customer_id").agg(
        (pl.lit(as_of) - pl.col("t_dat").max()).dt.total_days().alias("recency_days"),
        pl.col("t_dat").n_unique().alias("frequency"),
        pl.col("price").sum().alias("monetary"),
        pl.len().alias("line_items"),
    )

    # Recency is reversed: fewer days since last purchase is a BETTER score.
    return (
        base.with_columns(
            (5 - pl.col("recency_days").qcut(5, labels=[str(i) for i in range(5)])
                 .cast(pl.Int32)).alias("R"),
            (pl.col("frequency").rank("dense").qcut(
                5, labels=[str(i) for i in range(5)], allow_duplicates=True)
             .cast(pl.Int32) + 1).alias("F"),
            (pl.col("monetary").qcut(
                5, labels=[str(i) for i in range(5)], allow_duplicates=True)
             .cast(pl.Int32) + 1).alias("M"),
        )
        .with_columns(segment_expr().alias("segment"))
    )


def segment_expr() -> pl.Expr:
    """Label assignment as a single expression — evaluated in polars, not in a
    Python loop over 120k rows."""
    R, F, M = pl.col("R"), pl.col("F"), pl.col("M")
    return (
        pl.when((R >= 4) & (F >= 4)).then(pl.lit("champions"))
        .when(F >= 4).then(pl.lit("loyal"))
        .when(M >= 4).then(pl.lit("big_spenders"))
        .when((R >= 4) & (F <= 2)).then(pl.lit("promising"))
        .when((R <= 2) & (F >= 3)).then(pl.lit("at_risk"))
        .when((R <= 2) & (F <= 2)).then(pl.lit("hibernating"))
        .otherwise(pl.lit("needs_attention"))
    )


def segment_summary(rfm: pl.DataFrame) -> pl.DataFrame:
    """Aggregates only. The agent's `rfm_segment` tool returns this shape and
    never individual rows — POL-SEG-02 and the §13.2 RBAC row denying the agent
    individual customer records."""
    return (
        rfm.group_by("segment").agg(
            pl.len().alias("customers"),
            pl.col("recency_days").mean().round(1).alias("mean_recency_days"),
            pl.col("frequency").mean().round(2).alias("mean_frequency"),
            pl.col("monetary").mean().round(4).alias("mean_monetary"),
        ).sort("customers", descending=True)
    )
