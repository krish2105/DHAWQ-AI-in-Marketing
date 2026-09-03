#!/usr/bin/env python3
"""D12 — benchmark the LLM re-ranker against the other four arms.

    python3 eval/bench_reranker.py [--n 150] [--stability-n 8]

ARCHITECTURE.md §6.1 admits this arm on one condition: it faces the SAME
evaluation as everything else, plus rank stability and cost/latency per 1,000
slates.

SAMPLE SIZE IS SMALLER THAN THE MAIN HARNESS, DELIBERATELY AND ON THE RECORD.
Local inference costs seconds per slate, so scoring all 5,061 sampled customers
would take hours and buy no statistical power. The comparison arms are re-run
on the SAME customers so the numbers are like-for-like — comparing a 150-user
re-ranker against a 5,061-user hybrid would be a comparison of sample sizes,
not of models.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np

from services.api.core.artifacts import canonical_ids, test, train
from services.api.evaluate import beyond_accuracy as BA
from services.api.evaluate import bias as B
from services.api.evaluate.harness import KS, TOP_K, ground_truth, sample_customers
from services.api.evaluate.ranking import evaluate_user
from services.api.models.baseline import PopularityRecency
from services.api.models.collaborative import ImplicitALS
from services.api.models.content import ContentKNN
from services.api.models.hybrid import Hybrid
from services.api.models.llm_reranker import LLMReranker, rank_stability


def _provider(model: str):
    from services.api.agent.llm import OllamaProvider
    p = OllamaProvider(model=model)
    return p if p.available() else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--stability-n", type=int, default=8)
    ap.add_argument("--model", default="llama3.2:3b",
                    help="local model for the re-ranker. qwen3 runs a thinking "
                         "pass and is ~10x slower per slate for no gain on a "
                         "permutation task.")
    args = ap.parse_args()

    tr, te = train(), test()
    cat = canonical_ids()
    head = B.head_set(tr)

    allc, _ = sample_customers(tr, te, per_bucket=max(args.n // 4, 1))
    truth = ground_truth(te, allc)
    customers = [c for c in allc if truth.get(c)][: args.n]
    print(f"D12 — LLM re-ranker benchmark on {len(customers)} customers "
          f"(all arms on the SAME customers)")

    arms = [
        PopularityRecency(), ContentKNN(), ImplicitALS(),
        Hybrid(mode="weighted"), LLMReranker(provider=_provider(args.model)),
    ]

    results = {}
    for arm in arms:
        print(f"  fitting {arm.name} ...", flush=True)
        arm.fit(tr)
        slates = {}
        for i, c in enumerate(customers):
            slates[c] = arm.recommend(c, TOP_K)
            if arm.name == "llm_reranker" and (i + 1) % 25 == 0:
                print(f"    {i + 1}/{len(customers)}", flush=True)

        per = {c: evaluate_user(slates[c], truth[c], KS) for c in customers}
        keys = next(iter(per.values())).keys()
        all_slates = list(slates.values())
        results[arm.name] = {
            "ranking": {k: float(np.mean([m[k] for m in per.values()])) for k in keys},
            "beyond_accuracy": {
                "catalogue_coverage": BA.catalogue_coverage(all_slates, len(cat)),
                "gini": BA.gini(BA.impression_counts(all_slates, cat)),
                "long_tail_exposure": BA.long_tail_exposure(all_slates, head),
            },
            "bias": {"popularity_lift": B.popularity_lift(all_slates, tr)},
        }
        print(f"    ndcg@10={results[arm.name]['ranking']['ndcg@10']:.4f} "
              f"cov={results[arm.name]['beyond_accuracy']['catalogue_coverage']:.3f}")

    reranker = arms[-1]
    print(f"  measuring rank stability ({args.stability_n} customers x 5 runs) ...")
    stability = rank_stability(reranker, customers[: args.stability_n], runs=5)
    telemetry = reranker.telemetry.as_dict()

    base = results["hybrid_weighted"]["ranking"]["ndcg@10"]
    llm = results["llm_reranker"]["ranking"]["ndcg@10"]
    delta_pp = (llm - base) * 100

    art = {
        "run_id": f"reranker_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "corpus": "B", "kind": "llm_reranker_benchmark",
        "protocol": {
            "n_customers": len(customers), "top_k": TOP_K, "ks": list(KS),
            "candidate_pool": reranker.pool,
            "base_arm": reranker.base.name,
            "sample_note": (
                "Smaller than the main harness because local inference costs "
                "seconds per slate. All arms are re-run on the SAME customers, "
                "so the comparison is like-for-like."
            ),
        },
        "results": results,
        "rank_stability": stability,
        "cost_and_latency": telemetry,
        "verdict": {
            "ndcg10_delta_pp_vs_hybrid": round(delta_pp, 3),
            "beats_hybrid": bool(llm > base),
            "reading": (
                f"The re-ranker {'beats' if llm > base else 'does NOT beat'} the "
                f"hybrid by {delta_pp:+.3f}pp NDCG@10 at "
                f"{telemetry['latency_per_1000_slates_s']:.0f}s per 1,000 slates "
                f"against a sub-second deterministic arm. §6.1: a re-ranker that "
                f"wins by a fraction of a point at many times the cost has lost."
            ),
        },
        "production_path": False,
        "production_note": (
            "Benchmarked arm only. Not exposed in the agent tool catalogue — "
            "RecommendIn's model enum omits it, so the agent cannot select it "
            "even by accident."
        ),
    }

    out = REPO / "eval" / "artifacts" / f"{art['run_id']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(art, indent=2))

    print()
    print(f"  {'arm':<18}{'NDCG@10':>9}{'coverage':>10}{'gini':>8}")
    for name, r in results.items():
        print(f"  {name:<18}{r['ranking']['ndcg@10']:>9.4f}"
              f"{r['beyond_accuracy']['catalogue_coverage']:>10.3f}"
              f"{r['beyond_accuracy']['gini']:>8.3f}")
    print()
    print(f"  rank stability   max delta {stability['max_rank_delta']} positions · "
          f"churn {stability['mean_slate_churn']:.1%}")
    print(f"  latency          {telemetry['mean_latency_s']:.2f}s/slate · "
          f"{telemetry['latency_per_1000_slates_s']:.0f}s per 1,000")
    print(f"  invalid outputs  {telemetry['invalid_output_rate']:.1%} · "
          f"failures {telemetry['failure_rate']:.1%}")
    print()
    print(f"  {art['verdict']['reading']}")
    print(f"\n  wrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
