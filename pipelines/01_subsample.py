#!/usr/bin/env python3
"""D1 driver — run the subsample and temporal split.

    python3 pipelines/01_subsample.py

Writes data/processed/*.parquet and pipelines/manifests/subsample_v1.json.
Fails loudly on any of A1..A7. Logic lives in pipelines/subsample.py so it is
importable and testable; this file is the runner.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import polars as pl

from pipelines import subsample as R
from pipelines.common import (
    DATA_PROCESSED,
    require,
    sha256_file,
    step,
    write_manifest,
)

MANIFEST_NAME = "subsample_v1"


def main() -> int:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    print("D1 — subsample + temporal split")

    # ── R1 ───────────────────────────────────────────────────────────────────
    with step("R1a scan for max(t_dat)"):
        t_end = (
            R.scan_transactions()
            .select(pl.col("t_dat").max())
            .collect()
            .item()
        )
    t_start, t_end = R.window_bounds(t_end)
    split = R.split_date_for(t_start)
    print(f"      window {t_start} .. {t_end}   split at {split}")

    with step("R1b load window (predicate pushed into scan)"):
        txns = (
            R.scan_transactions()
            .filter(pl.col("t_dat").is_between(t_start, t_end))
            .collect()
        )
    n_window = txns.height
    print(f"      {n_window:,} transactions in window "
          f"({n_window / 31_788_325:.1%} of the file)")

    # ── R2 ───────────────────────────────────────────────────────────────────
    with step("R2a index image files"):
        on_disk = R.articles_with_image_file()
    with step("R2b restrict window to articles with an image"):
        txns = txns.filter(pl.col("article_id").is_in(list(on_disk)))
    candidates = txns.get_column("article_id").unique().to_list()
    print(f"      {len(on_disk):,} images on disk · "
          f"{len(candidates):,} candidate articles in window")

    with step(f"R2c verify {len(candidates):,} images decode (A5)"):
        decodable = R.articles_that_decode(candidates)
    n_bad = len(candidates) - len(decodable)
    if n_bad:
        print(f"      {n_bad} corrupt images dropped")
        txns = txns.filter(pl.col("article_id").is_in(list(decodable)))

    # ── R3 ───────────────────────────────────────────────────────────────────
    with step("R3 fixed-point support filter"):
        txns, keep_articles, keep_customers, iters, converged, trace = R.fixed_point_filter(txns)
    for t in trace:
        print(f"      iter {t['iteration']}: {t['articles']:,} articles · "
              f"{t['customers']:,} customers · {t['transactions']:,} txns")
    require(converged, "A6",
            f"fixed point still moving after {R.MAX_FIXPOINT_ITERS} iterations — "
            "the support thresholds are wrong for this window")

    # ── R4 / R5 ──────────────────────────────────────────────────────────────
    with step("R4 temporal split"):
        train, test = R.temporal_split(txns, split)
    with step("R5a identify cold-start customers (retained)"):
        cold = R.cold_start_customers(train, test)
    with step("R5b identify cold-start articles (retained)"):
        cold_arts = R.cold_articles(train, test)
        cold_art_rows = test.join(
            pl.DataFrame({"article_id": cold_arts}, schema={"article_id": pl.Utf8}),
            on="article_id", how="semi",
        ).height if cold_arts else 0

    # ── assertions ───────────────────────────────────────────────────────────
    with step("A1 leak assertion"):
        R.assert_no_leak(train, test)
    with step("A2 no unseen test articles"):
        R.assert_no_unseen_articles(test, keep_articles)
    with step("A3/A4 count magnitude"):
        R.assert_counts_in_range(len(keep_articles), len(keep_customers))

    # ── R6 freeze ────────────────────────────────────────────────────────────
    with step("R6a project article + customer tables"):
        articles = (
            pl.scan_csv(R.HM_RAW / "articles.csv",
                        schema_overrides={"article_id": pl.Utf8})
            .filter(pl.col("article_id").is_in(keep_articles))
            .collect()
        )
        customers = (
            pl.scan_csv(R.HM_RAW / "customers.csv",
                        schema_overrides={"customer_id": pl.Utf8})
            .filter(pl.col("customer_id").is_in(keep_customers))
            .collect()
        )

    outputs = {
        "articles": articles,
        "customers": customers,
        "transactions_train": train,
        "transactions_test": test,
    }
    with step("R6b write parquet"):
        paths = {}
        for name, df in outputs.items():
            p = DATA_PROCESSED / f"{name}.parquet"
            df.write_parquet(p, compression="zstd")
            paths[name] = p

    with step("A7 hash written artefacts"):
        hashes = {n: sha256_file(p) for n, p in paths.items()}
        for n, p in paths.items():
            require(p.exists() and p.stat().st_size > 0, "A7", f"{n}.parquet is empty")

    # Both readings of "customers with >= 3 transactions" are recorded. PLAN.md
    # §7 R3 specifies distinct dates; ARCHITECTURE.md §3 says "transactions",
    # which on line-item rows is a different and larger number. The choice is
    # methodological, so both are in the manifest rather than one in a comment.
    rows_per_customer = (
        train.group_by("customer_id").len().get_column("len")
    )

    manifest = write_manifest(MANIFEST_NAME, {
        "rule_source": "PLAN.md §7 (R1..R6, A1..A7)",
        "parameters": {
            "window_weeks": R.WINDOW_WEEKS,
            "train_weeks": R.TRAIN_WEEKS,
            "min_article_purchases": R.MIN_ARTICLE_PURCHASES,
            "min_customer_baskets": R.MIN_CUSTOMER_BASKETS,
            "customer_support_definition": "distinct t_dat (shopping occasions)",
            "customer_support_note": (
                "ARCHITECTURE.md §3 says 'transactions'; H&M rows are line items, "
                "so distinct dates is the stricter and more meaningful reading. "
                "PLAN.md §7 R3 specifies distinct t_dat. Mean line-items per "
                f"retained customer in train: {rows_per_customer.mean():.2f}."
            ),
            "max_fixpoint_iters": R.MAX_FIXPOINT_ITERS,
            "cold_start_max_purchases": R.COLD_START_MAX_PURCHASES,
        },
        "window": {"t_start": t_start, "t_end": t_end, "split_date": split},
        "counts": {
            "transactions_in_window": n_window,
            "images_on_disk": len(on_disk),
            "candidate_articles": len(candidates),
            "corrupt_images_dropped": n_bad,
            "articles": len(keep_articles),
            "customers": len(keep_customers),
            "transactions_total": txns.height,
            "transactions_train": train.height,
            "transactions_test": test.height,
            "cold_start_customers": len(cold),
            "cold_start_articles": len(cold_arts),
            "cold_start_article_test_rows": cold_art_rows,
        },
        "strata": {
            "note": (
                "Evaluation must stratify on these. A single aggregate NDCG "
                "blends warm and cold ranking and hides which is working."
            ),
            "cold_customers": {
                "n": len(cold),
                "definition": f"test customers with < {R.COLD_START_MAX_PURCHASES} train purchases",
            },
            "cold_articles": {
                "n": len(cold_arts),
                "test_rows": cold_art_rows,
                "share_of_test_rows": round(cold_art_rows / max(test.height, 1), 4),
                "definition": "articles present in test but absent from train",
                "why": (
                    "Passes the support filter with all purchases in the final two "
                    "weeks. Collaborative cannot score them; content-based can. "
                    "ARCHITECTURE.md §6."
                ),
            },
        },
        "fixed_point": {"iterations": iters, "converged": converged, "trace": trace},
        "deviations_from_plan": [
            {
                "item": "MAX_FIXPOINT_ITERS",
                "planned": 3, "actual": R.MAX_FIXPOINT_ITERS, "observed_need": iters,
                "reason": (
                    "Operational guard, not methodology. PLAN.md §7 estimated 2-3 "
                    "iterations; the real window needs more. Deltas collapse "
                    "geometrically, so the thresholds are sound — the cap was not."
                ),
            },
            {
                "item": "A4 expected customer range",
                "planned": [30_000, 80_000], "actual": list(R.EXPECT_CUSTOMERS),
                "predicted": R.PREDICTED_CUSTOMERS, "observed": len(keep_customers),
                "reason": (
                    "ARCHITECTURE.md §3 predicted ~50k customers. The rule it "
                    "specifies retains ~120k. The rule is the methodology and the "
                    "count was a guess, so the rule stands and the bound moved. "
                    "Tightening min_customer_baskets to hit 50k would be fitting "
                    "the method to an estimate."
                ),
            },
        ],
        "assertions": {a: "PASS" for a in ("A1", "A2", "A3", "A4", "A5", "A6", "A7")},
        "outputs": {n: {"path": str(p.relative_to(R.DATA_RAW.parent.parent)),
                        "rows": outputs[n].height,
                        "sha256": hashes[n]} for n, p in paths.items()},
    })

    print()
    print("  RESULT")
    print(f"    articles              {len(keep_articles):>10,}")
    print(f"    customers             {len(keep_customers):>10,}")
    print(f"    transactions (total)  {txns.height:>10,}")
    print(f"      train               {train.height:>10,}  "
          f"{train.get_column('t_dat').min()} .. {train.get_column('t_dat').max()}")
    print(f"      test                {test.height:>10,}  "
          f"{test.get_column('t_dat').min()} .. {test.get_column('t_dat').max()}")
    print(f"    cold-start customers  {len(cold):>10,}  (retained as a stratum)")
    print(f"    cold-start articles   {len(cold_arts):>10,}  "
          f"({cold_art_rows:,} test rows = {cold_art_rows / test.height:.1%})")
    print(f"    fixed point converged in {iters} iteration(s)")
    print()
    print(f"  manifest: {manifest.relative_to(Path.cwd())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
