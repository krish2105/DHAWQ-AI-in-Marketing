#!/usr/bin/env python3
"""D21b — precompute cohort candidates so the API never fits a model.

    python3 pipelines/06_precompute_cohorts.py

WHY THIS EXISTS
The deployed API was OOM-killed on a 512MB instance. Measured, not guessed:

    artefacts loaded        207 MB
    + ContentKNN.fit        952 MB   <- 119,033 x 768 float profiles = 366MB

Three attempts to shrink the fit in place (lazy profiles, CSR baskets, a fully
numeric sort) all stayed above 600MB, because the cost is not any one structure
— it is holding 1.39M interactions and a 42MB embedding matrix in a process
that also has to serve requests.

The right fix was not a smaller fit. It was noticing that THE API DOES NOT NEED
TO FIT ANYTHING. Every other number in DHAWQ is a frozen build-time artefact
the service reads; cohort candidates were the one thing still being computed at
runtime, and they are as static as the rest. Nothing about a cohort's
candidates changes between requests.

So they are computed HERE, on the full stack, with the real hybrid — and the
deployed service reads a 2MB JSON file. That is strictly better than the light
mode it replaces: light mode substituted a weaker arm, this serves the actual
hybrid's output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import polars as pl

from pipelines.common import DATA_PROCESSED, require, sha256_file, step, write_manifest

OUT = DATA_PROCESSED / "cohorts"
TOP_K = 200
COHORT_SAMPLE = 400          # members averaged per segment
MODELS = ["hybrid_weighted", "hybrid_cascade", "collaborative", "content", "popularity"]


def _simulate(segment: str, members: list[str], candidates: dict,
              tr: pl.DataFrame, k: int = 12, model: str = "hybrid_weighted") -> dict:
    """Exactly what the runtime endpoint computed, done once at build time."""
    from services.api.core.artifacts import articles, test
    from services.api.evaluate.bias import head_set
    from services.api.marketing.lift import project_lift
    from services.api.marketing.slots import Candidate, optimise_slots

    head = head_set(tr)   # build time: the parquet is already loaded here
    arts = articles()
    meta = {r[0]: {"prod_name": r[1], "product_type_name": r[2],
                   "colour_group_name": r[3]}
            for r in arts.select("article_id", "prod_name", "product_type_name",
                                 "colour_group_name").iter_rows()}
    prices = dict(tr.group_by("article_id").agg(pl.col("price").mean()).iter_rows())

    def build(model_name: str, quota: float | None) -> dict:
        cands = candidates[model_name][segment]
        cs = [
            Candidate(article_id=a, score=1.0 - i / max(len(cands), 1),
                      price=float(prices.get(a, 0.0) or 0.0),
                      product_type=(meta.get(a) or {}).get("product_type_name") or "unknown",
                      colour_group=(meta.get(a) or {}).get("colour_group_name") or "unknown",
                      is_long_tail=a not in head)
            for i, a in enumerate(cands) if a in meta
        ]
        constraints = {} if quota is None else {"min_long_tail_share": quota}
        slate, report = optimise_slots(cs, k, **constraints)
        return {
            "model": model_name,
            "slate": [{"article_id": a, "position": i + 1,
                       "is_long_tail": a not in head, **meta.get(a, {})}
                      for i, a in enumerate(slate)],
            "report": report.__dict__,
            "long_tail_share": sum(1 for a in slate if a not in head) / max(len(slate), 1),
        }

    model_side = build(model, None)
    unconstrained = build(model, 0.0)
    baseline = build("popularity", 0.0)

    # Relevance is the target cohort's held-out purchases — the only signal
    # independent of both arms.
    cohort_test = test().join(
        pl.DataFrame({"customer_id": members[:3000]}, schema={"customer_id": pl.Utf8}),
        on="customer_id", how="semi")
    purch = cohort_test.group_by("article_id").len().rename({"len": "n"})
    max_n = float(purch.get_column("n").max() or 1)
    rel = {"sim": {a: float(n) / max_n for a, n in
                   zip(purch.get_column("article_id"), purch.get_column("n"))}}

    def ids(side): return [a["article_id"] for a in side["slate"]]

    def lift(m, b, mc, bc):
        return project_lift({"sim": ids(m)}, {"sim": ids(b)}, rel, prices,
                            model_coverage=mc, baseline_coverage=bc).as_dict()

    return {
        "k": k, "segment": segment, "cohort_size": len(members),
        "model": model_side, "unconstrained": unconstrained, "baseline": baseline,
        "decomposition": {
            "personalisation_effect": lift(unconstrained, baseline,
                                           unconstrained["long_tail_share"],
                                           baseline["long_tail_share"]),
            "quota_cost": lift(model_side, unconstrained,
                               model_side["long_tail_share"],
                               unconstrained["long_tail_share"]),
            "combined": lift(model_side, baseline, model_side["long_tail_share"],
                             baseline["long_tail_share"]),
            "reading": (
                "personalisation_effect isolates the model against the "
                "bestseller page with neither carrying a quota. quota_cost "
                "isolates what POL-LT-01 costs by holding the model fixed."
            ),
        },
        "the_finding": (
            "Personalisation wins PER CUSTOMER and loses PER COHORT SLATE, and "
            "those are not in conflict. Ranking metrics evaluate a list per "
            "customer, where collaborative beats popularity. A slate is ONE "
            "page shown to a whole cohort, so the best it can do is target the "
            "cohort's modal preference — and the modal preference of any large "
            "cohort is, definitionally, its bestsellers. Personalised slates pay "
            "off for small, sharply-defined cohorts and converge on the "
            "bestseller page as the cohort widens."
        ),
        "known_bias": (
            "THIS METRIC STRUCTURALLY FAVOURS THE BESTSELLER PAGE. Projected "
            "revenue is estimated from held-out purchase FREQUENCY, which is "
            "exactly what the popularity arm ranks on. No offline estimator "
            "built from observed purchases can show personalisation winning; "
            "only a live A/B test could, and there isn't one."
        ),
    }


def main() -> int:
    from services.api.core.artifacts import canonical_ids, train
    from services.api.marketing.rfm import rfm_table

    OUT.mkdir(parents=True, exist_ok=True)
    print("D21b — precomputing cohort candidates")

    tr = train()
    ids = canonical_ids()

    with step("RFM segmentation"):
        seg = rfm_table(tr)
        segments = {
            s: seg.filter(pl.col("segment") == s).get_column("customer_id")
            .sort().to_list()
            for s in seg.get_column("segment").unique().to_list()
        }
    for s, m in sorted(segments.items()):
        print(f"      {s:<18} {len(m):>7,} customers")

    from services.api.models.baseline import PopularityRecency
    from services.api.models.collaborative import ImplicitALS
    from services.api.models.content import ContentKNN
    from services.api.models.hybrid import Hybrid

    arms = {
        "popularity": PopularityRecency(), "content": ContentKNN(),
        "collaborative": ImplicitALS(),
        "hybrid_weighted": Hybrid(mode="weighted"),
        "hybrid_cascade": Hybrid(mode="cascade"),
    }

    out: dict[str, dict[str, list[str]]] = {}
    for name in MODELS:
        with step(f"fit {name}"):
            arm = arms[name].fit(tr)
        for sname, members in sorted(segments.items()):
            # The cohort score is the mean of its members' score vectors —
            # identical to what the runtime tool computed, just done once.
            scores = None
            for c in members[:COHORT_SAMPLE]:
                v = arm.score_customer(c)
                if not np.isfinite(v).any():
                    continue
                v = np.where(np.isfinite(v), v, np.nan)
                scores = v if scores is None else np.nansum([scores, v], axis=0)
            if scores is None:
                continue
            order = np.argsort(-np.nan_to_num(scores, nan=-np.inf))[:TOP_K]
            out.setdefault(name, {})[sname] = [ids[i] for i in order]
        print(f"      {name:<18} {len(out.get(name, {}))} segments")

    require(bool(out), "C1", "no cohort candidates produced")

    # The segment and CLV payloads are equally static and equally expensive:
    # rfm_table groups 1.39M rows, and BG/NBD runs Nelder-Mead over 119k
    # customers. Both were computed per request, which is why /segments showed
    # an empty skeleton on the deployed instance.
    with step("RFM segment aggregates"):
        from services.api.marketing.rfm import segment_summary
        summary = segment_summary(seg)
        rfm_payload = {"segments": [dict(zip(summary.columns, r))
                                    for r in summary.iter_rows()]}

    with step("projected CLV (BG/NBD + Gamma-Gamma)"):
        from services.api.marketing.clv import (
            clv, fit_bgnbd, fit_gamma_gamma, frequency_monetary_correlation,
            rfm_matrix,
        )
        matrix = rfm_matrix(tr)
        bg, gg = fit_bgnbd(matrix), fit_gamma_gamma(matrix)
        cl = clv(matrix, bg, gg)
        v = cl.get_column("projected_clv").to_numpy()
        alive = cl.get_column("probability_alive").to_numpy()
        hi = float(np.percentile(v, 99))
        counts, edges = np.histogram(np.clip(v, 0, hi), bins=24)
        clv_payload = {
            "n_customers": cl.height,
            "params": {"bgnbd": bg.__dict__, "gamma_gamma": gg.__dict__},
            "projected_clv": {
                "mean": float(v.mean()), "median": float(np.median(v)),
                "deciles": [float(np.percentile(v, q)) for q in range(10, 100, 10)],
                "p90": float(np.percentile(v, 90)),
            },
            "probability_alive": {"mean": float(alive.mean())},
            "histogram": [{"x": round(float(edges[i]), 5), "n": int(counts[i])}
                          for i in range(len(counts))],
            "assumption_check": {
                "frequency_monetary_correlation":
                    frequency_monetary_correlation(matrix),
                "note": ("Gamma-Gamma assumes frequency and monetary value are "
                         "independent. The observed correlation is reported so "
                         "the assumption is visible rather than buried."),
            },
            "language": "PROJECTED, not measured. No live A/B test exists.",
        }
    print(f"      {clv_payload['n_customers']:,} customers · "
          f"r = {clv_payload['assumption_check']['frequency_monetary_correlation']:.3f}")

    (OUT / "segments.json").write_text(json.dumps(
        {"rfm": rfm_payload, "clv": clv_payload}))

    # The slot optimiser needs the head/tail split and a mean price per
    # article. Both derive from the frozen training split and never change, so
    # loading an 18MB transaction parquet per request to recompute them is pure
    # waste — and it was the last thing keeping the API's memory high.
    with step("catalogue facts (head/tail split, mean price)"):
        from services.api.evaluate.bias import head_set
        prices = dict(tr.group_by("article_id").agg(pl.col("price").mean()).iter_rows())
        (OUT / "catalogue.json").write_text(json.dumps({
            "head": sorted(head_set(tr)),
            "prices": {a: round(float(p or 0.0), 6) for a, p in prices.items()},
            "head_share": 0.20,
            "note": ("Head is the top 20% of articles by units sold in the "
                     "training window, frozen with the catalogue so the quota "
                     "the optimiser enforces and the exposure the metrics "
                     "measure refer to the same set."),
        }))

    # The slot simulation is static per (segment, k, model) too — it reads the
    # same candidates and the same held-out purchases every time.
    with step("slot simulations"):
        sims = {}
        for sname in sorted(segments):
            try:
                sims[sname] = _simulate(sname, segments[sname], out, tr)
            except Exception as exc:            # a thin segment is not fatal
                print(f"      {sname}: skipped ({exc})")
        (OUT / "simulations.json").write_text(json.dumps(sims, default=str))
    print(f"      {len(sims)} segments simulated")

    path = OUT / "candidates.json"
    path.write_text(json.dumps({
        "version": 1, "top_k": TOP_K, "cohort_sample": COHORT_SAMPLE,
        "models": MODELS,
        "segment_sizes": {s: len(m) for s, m in segments.items()},
        "candidates": out,
    }))

    write_manifest("cohorts_v1", {
        "top_k": TOP_K, "cohort_sample": COHORT_SAMPLE,
        "models": MODELS,
        "segments": {s: len(m) for s, m in segments.items()},
        "why": (
            "The API reads these instead of fitting at runtime. Fitting cost "
            "952MB peak and OOM-killed a 512MB instance; cohort candidates are "
            "as static as every other artefact in DHAWQ, so they belong at "
            "build time. This serves the REAL hybrid's output, unlike the light "
            "mode it replaces, which substituted a weaker arm."
        ),
        "outputs": {
            "candidates.json": {"bytes": path.stat().st_size,
                                "sha256": sha256_file(path)},
            "segments.json": {"bytes": (OUT / "segments.json").stat().st_size,
                              "sha256": sha256_file(OUT / "segments.json")},
            "catalogue.json": {"bytes": (OUT / "catalogue.json").stat().st_size,
                               "sha256": sha256_file(OUT / "catalogue.json")},
        },
    })
    print(f"\n  wrote {path.relative_to(Path.cwd())} "
          f"({path.stat().st_size/1e6:.1f}MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
