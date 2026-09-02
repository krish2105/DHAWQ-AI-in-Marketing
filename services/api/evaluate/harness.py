"""D4 — the recommender evaluation harness.

Fits every arm on train, scores held-out test purchases, and writes a run
manifest into eval/artifacts/. Those manifests ARE corpus B (ARCHITECTURE.md
§3) — the agent's Analyst node retrieves over them, so the shape here is a
retrieval surface, not just a log.

SAMPLING. There are 56,379 test customers. Scoring all of them across five arms
is ~3.8 billion float operations per full sweep, which turns a 40-second
iteration into a 10-minute one and buys no statistical power — the standard
error on NDCG at n=6,000 is already ~0.005. The sample is STRATIFIED by history
depth so every §9 cold-start bucket is represented at usable size, and both the
size and the seed are recorded in the manifest.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from services.api.core.artifacts import (
    articles, canonical_ids, embeddings, article_index, manifest, test, train,
)
from services.api.evaluate import bias as B
from services.api.evaluate import beyond_accuracy as BA
from services.api.evaluate import coldstart as CS
from services.api.evaluate.ranking import evaluate_user

ARTIFACT_DIR = Path(__file__).resolve().parents[3] / "eval" / "artifacts"
KS = (5, 10, 20)
TOP_K = 20
SAMPLE_PER_BUCKET = 1500
SEED = 20260903


def build_arms(seed: int = SEED) -> list:
    from services.api.models.baseline import PopularityRecency
    from services.api.models.collaborative import ImplicitALS
    from services.api.models.content import ContentKNN
    from services.api.models.hybrid import Hybrid

    return [
        PopularityRecency(seed=seed),
        ContentKNN(seed=seed),
        ImplicitALS(seed=seed),
        Hybrid(mode="weighted", seed=seed),
        Hybrid(mode="cascade", seed=seed),
    ]


def sample_customers(tr: pl.DataFrame, te: pl.DataFrame,
                     per_bucket: int = SAMPLE_PER_BUCKET, seed: int = SEED
                     ) -> tuple[list[str], dict[str, list[str]]]:
    """Stratified by history depth. Uniform sampling would give ~30 customers
    in the 0-purchase bucket, and the cold-start curve — the thing §9 says to
    report — would be noise."""
    rng = np.random.default_rng(seed)
    depths = CS.user_history_depth(tr)
    eligible = te.get_column("customer_id").unique().sort().to_list()
    strata = CS.stratify_users(eligible, depths)

    chosen: dict[str, list[str]] = {}
    for label, members in strata.items():
        if not members:
            chosen[label] = []
            continue
        take = min(per_bucket, len(members))
        idx = rng.choice(len(members), size=take, replace=False)
        chosen[label] = sorted(members[i] for i in idx)
    return sorted({c for v in chosen.values() for c in v}), chosen


def ground_truth(te: pl.DataFrame, customers: list[str]) -> dict[str, set[str]]:
    want = pl.DataFrame({"customer_id": customers}, schema={"customer_id": pl.Utf8})
    g = (
        te.join(want, on="customer_id", how="semi")
        .group_by("customer_id").agg(pl.col("article_id").unique())
    )
    return {c: set(a) for c, a in zip(g.get_column("customer_id"),
                                     g.get_column("article_id"))}


def run(per_bucket: int = SAMPLE_PER_BUCKET, seed: int = SEED,
        write: bool = True) -> dict:
    tr, te = train(), test()
    cat = canonical_ids()
    emb, aidx = embeddings(), article_index()
    head = B.head_set(tr)
    depths = CS.user_history_depth(tr)
    cold_arts = CS.cold_article_set(tr, te)

    customers, strata = sample_customers(tr, te, per_bucket, seed)
    truth = ground_truth(te, customers)
    customers = [c for c in customers if truth.get(c)]

    pop_counts = tr.group_by("article_id").len().rows_by_key("article_id", unique=True)
    popularity = {a: c[0] for a, c in pop_counts.items()}
    n_train = tr.height

    print(f"D4 — evaluating {len(customers):,} sampled test customers")
    for label, members in strata.items():
        print(f"      bucket {label:>4}: {len([c for c in members if c in truth]):,}")

    results: dict[str, dict] = {}
    baseline_slates: dict[str, list[str]] = {}

    for arm in build_arms(seed):
        t0 = time.perf_counter()
        arm.fit(tr)
        fit_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        slates = {c: arm.recommend(c, TOP_K) for c in customers}
        rec_s = time.perf_counter() - t0

        if arm.name == "popularity":
            baseline_slates = slates

        per_user = {c: evaluate_user(slates[c], truth[c], KS) for c in customers}
        keys = next(iter(per_user.values())).keys()
        overall = {k: float(np.mean([m[k] for m in per_user.values()])) for k in keys}

        by_bucket = {}
        for label, members in strata.items():
            ms = [per_user[c] for c in members if c in per_user]
            by_bucket[label] = (
                {"n": len(ms), **{k: float(np.mean([m[k] for m in ms])) for k in keys}}
                if ms else {"n": 0}
            )

        all_slates = list(slates.values())
        counts = BA.impression_counts(all_slates, cat)
        cold_hits = sum(
            1 for c in customers for a in slates[c] if a in cold_arts
        )

        results[arm.name] = {
            "ranking": overall,
            "by_history_depth": by_bucket,
            "beyond_accuracy": {
                "catalogue_coverage": BA.catalogue_coverage(all_slates, len(cat)),
                "gini": BA.gini(counts),
                "long_tail_exposure": BA.long_tail_exposure(all_slates, head),
                "novelty": BA.novelty(all_slates, popularity, n_train),
                "serendipity": BA.serendipity(slates, truth, baseline_slates),
                "mean_intra_list_diversity": float(np.mean([
                    BA.intra_list_diversity(s[:10], emb, aidx) for s in all_slates[:800]
                ])),
            },
            "bias": {
                "popularity_lift": B.popularity_lift(all_slates, tr),
                "head_share_of_impressions": B.head_share_of_impressions(all_slates, head),
                "concentration_curve": B.concentration_curve(all_slates, cat),
            },
            "cold_articles": {
                "handled_by_arm": arm.handles_cold_articles,
                "impressions_on_cold_articles": cold_hits,
            },
            "timing": {"fit_seconds": round(fit_s, 2),
                       "recommend_seconds": round(rec_s, 2),
                       "ms_per_customer": round(1000 * rec_s / max(len(customers), 1), 2)},
        }
        print(f"  {arm.name:<18} ndcg@10={overall['ndcg@10']:.4f}  "
              f"cov={results[arm.name]['beyond_accuracy']['catalogue_coverage']:.3f}  "
              f"gini={results[arm.name]['beyond_accuracy']['gini']:.3f}  "
              f"({fit_s:.1f}s fit)")

    run_id = f"run_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
    art = {
        "run_id": run_id,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "corpus": "B",
        "kind": "recommender_evaluation",
        "data": {
            "subsample_manifest": manifest()["outputs"],
            "window": manifest()["window"],
            "n_articles": len(cat),
            "n_test_customers_total": te.get_column("customer_id").n_unique(),
        },
        "protocol": {
            "split": "temporal",
            "leak_assertion": "A1 max(train.t_dat) < min(test.t_dat) — enforced in tests",
            "ks": list(KS), "top_k": TOP_K,
            "sample_per_bucket": per_bucket, "seed": seed,
            "n_evaluated": len(customers),
            "sampling": "stratified by history depth; uniform sampling starves the 0-purchase bucket",
            "relevance": "binary — held-out purchase in the test window",
            "known_limitation": (
                "Purchases, not impressions. An unpurchased article is UNLABELLED, "
                "not rejected, so precision is depressed and recall is a lower bound. "
                "ARCHITECTURE.md §3, §16."
            ),
        },
        "strata": {
            "history_depth": {k: len([c for c in v if c in truth]) for k, v in strata.items()},
            "cold_articles": {"n": len(cold_arts),
                              "note": "collaborative structurally cannot score these"},
        },
        "results": results,
        "frontier": [
            {"model": n,
             "ndcg@10": r["ranking"]["ndcg@10"],
             "coverage": r["beyond_accuracy"]["catalogue_coverage"],
             "gini": r["beyond_accuracy"]["gini"],
             "long_tail_exposure": r["beyond_accuracy"]["long_tail_exposure"]}
            for n, r in results.items()
        ],
    }

    if write:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        p = ARTIFACT_DIR / f"{run_id}.json"
        p.write_text(json.dumps(art, indent=2))
        print(f"\n  wrote {p.relative_to(Path.cwd())}")
    return art


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    run()
