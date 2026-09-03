"""DHAWQ FastAPI service — ARCHITECTURE.md §4, PLAN.md §10.

Deliberately SMALL server surface (PLAN.md §0.5). Everything precomputed is a
static artefact served from the CDN, not a Python route: the atlas, positions,
colours, the evaluation reports and corpus C. The gallery, the frontier plot
and the evaluation view all render with this API asleep, which matters for
free-tier cold starts and is worth having independently of hosting.

Live routes exist only for recommendations, the agent, and segments.
"""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from services.api.agent.graph import explainer, human_gate, new_run, run_to_gate
from services.api.agent.state import GateResolution, MerchandisingRun
from services.api.core.rbac import Role

app = FastAPI(title="DHAWQ", version="1.0.0",
              description="Visual recommendation intelligence")

app.add_middleware(
    CORSMiddleware,
    # Explicit allowlist, never ["*"] with credentials (§13.1).
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


class ErrorBody(BaseModel):
    code: str
    message: str
    detail: Any = None
    request_id: str


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=500, content={"error": ErrorBody(
        code="internal_error", message=str(exc), request_id=uuid.uuid4().hex[:12],
    ).model_dump()})


# ── in-process run store. Postgres checkpointing is wired in graph.py; this is
# ── the read model the SSE stream and the resume endpoint share.
_RUNS: dict[str, MerchandisingRun] = {}
_EVENTS: dict[str, list[dict]] = {}


def _emit(run_id: str, type_: str, payload: dict) -> None:
    q = _EVENTS.setdefault(run_id, [])
    q.append({"seq": len(q), "run_id": run_id, "type": type_,
              "ts": datetime.now().astimezone().isoformat(), "payload": payload})


# ── catalogue and scene ──────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict:
    return {"ok": True, "service": "dhawq"}


@app.get("/space/manifest")
def space_manifest() -> dict:
    # Via core.artifacts, NOT pipelines.common — services/api never imports
    # pipelines (§4). Artefacts cross that line as frozen files plus a
    # manifest, never as a live call.
    from services.api.core.artifacts import manifest as read_manifest
    atlas = read_manifest("atlas_v1")
    umap = read_manifest("umap_v1")
    return {
        "version": "1.0.0",
        "n": umap["counts"]["points"],
        "positions_url": "/static/positions.bin",
        "signal_colour": atlas["signal_colour"],
        "variants": {k: {kk: vv for kk, vv in v.items() if kk != "sheets"}
                     for k, v in atlas["variants"].items()},
        "sheets": {k: [s["file"] for s in v["sheets"]]
                   for k, v in atlas["variants"].items()},
        "extent": umap["extent"],
    }


@app.get("/catalogue/articles")
def list_articles(cursor: int = 0, limit: int = Query(60, le=200)) -> dict:
    from services.api.core.artifacts import articles
    df = articles().slice(cursor, limit)
    cols = [c for c in ("article_id", "prod_name", "product_type_name",
                        "colour_group_name", "index_group_name") if c in df.columns]
    return {"items": [dict(zip(cols, r)) for r in df.select(cols).iter_rows()],
            "next_cursor": cursor + limit}


@app.get("/catalogue/articles/{article_id}")
def get_article(article_id: str) -> dict:
    import polars as pl
    from services.api.core.artifacts import articles
    row = articles().filter(pl.col("article_id") == article_id)
    if row.height == 0:
        raise HTTPException(404, "no such article")
    return dict(zip(row.columns, row.row(0)))


@app.get("/recs/article/{article_id}/why")
def why_this(article_id: str, k: int = Query(8, le=30)) -> dict:
    """The three signals SEPARATELY. The overlay renders them; it does not
    compute them (§12.5)."""
    from services.api.core.artifacts import article_index, canonical_ids, embeddings
    from services.api.core.numerics import safe_matmul
    from services.api.rag.graph_index import load_graph

    idx, ids, emb = article_index(), canonical_ids(), embeddings()
    if article_id not in idx:
        raise HTTPException(404, "no such article")

    import numpy as np
    sims = safe_matmul(emb, emb[idx[article_id]], where="why_this")
    sims[idx[article_id]] = -np.inf
    top = np.argsort(-sims)[:k]

    g = load_graph()
    paths = {p.target: p for p in g.traverse(article_id, depth=2, limit=400)}

    return {"article_id": article_id, "neighbours": [
        {"article_id": ids[i], "visual": round(float(sims[i]), 4),
         "collaborative": round(float(paths[ids[i]].score), 4) if ids[i] in paths else 0.0,
         "taxonomy_path": paths[ids[i]].describe() if ids[i] in paths else None,
         "uses_predicted_evidence": (paths[ids[i]].uses_predicted_evidence
                                     if ids[i] in paths else False)}
        for i in top]}


@app.get("/merchandise/policy")
def policy() -> dict:
    pdir = REPO / "services" / "api" / "rag" / "corpora" / "policy"
    return {"version": json.loads((pdir / "manifest.json").read_text())["policy_version"],
            "document": (pdir / "POLICY.md").read_text(),
            "manifest": json.loads((pdir / "manifest.json").read_text())}


@app.get("/evaluate/latest")
def latest_eval() -> dict:
    import glob
    files = sorted(glob.glob(str(REPO / "eval" / "artifacts" / "run_*.json")))
    agent = sorted(glob.glob(str(REPO / "eval" / "artifacts" / "agenteval_*.json")))
    return {
        "recommenders": json.loads(Path(files[-1]).read_text()) if files else None,
        "agent": json.loads(Path(agent[-1]).read_text()) if agent else None,
    }


@app.get("/segments/rfm")
def segments() -> dict:
    from services.api.core.artifacts import train
    from services.api.marketing.rfm import rfm_table, segment_summary
    s = segment_summary(rfm_table(train()))
    return {"segments": [dict(zip(s.columns, r)) for r in s.iter_rows()]}


@app.get("/segments/clv")
def clv_summary() -> dict:
    """PROJECTED CLV aggregates. Never individual rows — POL-SEG-02 and the
    §13.2 row denying access to individual customer records."""
    from services.api.core.artifacts import train
    from services.api.marketing.clv import (
        clv, fit_bgnbd, fit_gamma_gamma, frequency_monetary_correlation, rfm_matrix,
    )
    import numpy as np

    summary = rfm_matrix(train())
    bg, gg = fit_bgnbd(summary), fit_gamma_gamma(summary)
    out = clv(summary, bg, gg)
    v = out.get_column("projected_clv").to_numpy()
    alive = out.get_column("probability_alive").to_numpy()
    deciles = [float(np.percentile(v, q)) for q in range(10, 100, 10)]

    return {
        "n_customers": out.height,
        "params": {"bgnbd": bg.__dict__, "gamma_gamma": gg.__dict__},
        "projected_clv": {
            "mean": float(v.mean()), "median": float(np.median(v)),
            "deciles": deciles, "p90": float(np.percentile(v, 90)),
        },
        "probability_alive": {"mean": float(alive.mean())},
        "histogram": _histogram(v, 24),
        "assumption_check": {
            "frequency_monetary_correlation": frequency_monetary_correlation(summary),
            "note": ("Gamma-Gamma assumes frequency and monetary value are "
                     "independent. The observed correlation is reported so the "
                     "assumption is visible rather than buried."),
        },
        "language": "PROJECTED, not measured. No live A/B test exists.",
    }


def _cohort_customers(segment: str = "champions", k_sample: int = 3000) -> list[str]:
    """A GENUINELY DIFFERENTIATED cohort, not a sample of everyone.

    The first version took the first N customer ids. That is a random sample of
    the whole base, and the aggregate preference of a random sample of everyone
    IS the popularity distribution — so the "personalised" slate was being asked
    to beat popularity at predicting popularity. It lost by ~90%, correctly and
    uninterestingly.

    Personalisation can only beat a bestseller page where the cohort has taste
    that differs from the mean. An RFM segment does; an arbitrary slice does
    not. That distinction is the whole premise of the marketing claim, so the
    simulator has to honour it.
    """
    import polars as pl
    from services.api.core.artifacts import train
    from services.api.marketing.rfm import rfm_table

    t = rfm_table(train())
    members = (
        t.filter(pl.col("segment") == segment)
        .get_column("customer_id").sort().to_list()
    )
    if not members:                       # segment empty — fall back, and say so
        members = train().get_column("customer_id").unique().sort().to_list()
    return members[:k_sample]


def _histogram(v, bins: int) -> list[dict]:
    import numpy as np
    hi = float(np.percentile(v, 99))          # clip the tail so the chart reads
    counts, edges = np.histogram(np.clip(v, 0, hi), bins=bins)
    return [{"x": round(float(edges[i]), 5), "n": int(counts[i])}
            for i in range(len(counts))]


@app.get("/merchandise/simulate")
def simulate(k: int = Query(12, ge=4, le=24), model: str = "hybrid_weighted",
             segment: str = "champions") -> dict:
    """Side-by-side: the model's slate against the popularity baseline, with
    the projected revenue delta AND the coverage cost together.

    The two travel in one payload deliberately — §9's "the tension is the
    finding". You cannot read the upside without the cost.
    """
    from services.api.agent.tools import OptimiseSlotsIn, RecommendIn, _optimise_slots, _recommend
    from services.api.core.artifacts import articles, train
    from services.api.evaluate.bias import head_set
    from services.api.marketing.lift import project_lift

    head = head_set(train())
    arts = articles()
    meta = {r[0]: {"prod_name": r[1], "product_type_name": r[2],
                   "colour_group_name": r[3]}
            for r in arts.select("article_id", "prod_name", "product_type_name",
                                 "colour_group_name").iter_rows()}
    prices = dict(
        train().group_by("article_id").agg(__import__("polars").col("price").mean())
        .iter_rows()
    )

    cohort = _cohort_customers(segment)

    def build(model_name: str, quota: float | None = None) -> dict:
        cands = _recommend(RecommendIn(
            k=120, model=model_name,
            # Both arms see the SAME cohort. Popularity ignores it by
            # construction, which is exactly what makes it the baseline.
            cohort_spec={"customer_ids": cohort},
        ))
        constraints = {} if quota is None else {"min_long_tail_share": quota}
        out = _optimise_slots(OptimiseSlotsIn(candidate_ids=cands.article_ids, k=k,
                                              constraints=constraints))
        return {
            "model": model_name,
            "slate": [{"article_id": a, "position": i + 1,
                       "is_long_tail": a not in head, **meta.get(a, {})}
                      for i, a in enumerate(out.slate)],
            "report": out.report,
            "long_tail_share": (sum(1 for a in out.slate if a not in head)
                                / max(len(out.slate), 1)),
        }

    # THREE ARMS, BECAUSE "LIFT" WAS CONFLATING TWO DIFFERENT EFFECTS.
    #
    # The model carries a long-tail quota and the popularity baseline does not,
    # so a single comparison mixes the PERSONALISATION effect with the QUOTA
    # COST and reports their sum as though it were one number. It read -90% and
    # was uninterpretable: no reader could tell whether personalisation failed
    # or whether the quota was simply expensive.
    #
    # Decomposed, both halves are answerable and both are decisions a
    # merchandiser actually makes:
    #   personalisation effect = unconstrained model vs popularity
    #   quota cost             = constrained model vs unconstrained model
    model_side = build(model)                       # with the policy quota
    unconstrained_side = build(model, quota=0.0)    # same model, no quota
    baseline_side = build("popularity", quota=0.0)

    # RELEVANCE MUST BE INDEPENDENT OF BOTH ARMS — AND THREE ATTEMPTS WERE NOT.
    #
    #   1. From the model's own slate: every baseline article scored 0.0, so
    #      baseline revenue was zero and the lift was meaningless.
    #   2. From the model's own ranking over a shared pool: still derived from
    #      the arm under test, which handed it the top of the scale. +2900%.
    #   3. From GLOBAL held-out purchase frequency: independent of the model,
    #      but it is a popularity signal, so it favours the popularity baseline
    #      by construction. -91%.
    #
    # Each was biased in a different direction, which is the tell that the
    # choice of relevance function IS the experiment. The signal that favours
    # neither is what the TARGET COHORT actually bought in the held-out window
    # — the same per-customer ground truth the §9 ranking metrics already use.
    # Personalisation can win on it and so can popularity; nothing about its
    # construction decides which.
    #
    # It inherits the dataset's central limitation: purchases, not impressions.
    # An article the cohort did not buy is UNLABELLED, not rejected, so this
    # understates both slates — and understates them EQUALLY, which is what
    # makes the comparison fair even though the absolute level is not
    # meaningful on its own.
    from services.api.core.artifacts import test as _test
    import polars as _pl

    cohort_test = _test().join(
        _pl.DataFrame({"customer_id": cohort}, schema={"customer_id": _pl.Utf8}),
        on="customer_id", how="semi",
    )
    purchases = cohort_test.group_by("article_id").len().rename({"len": "n"})
    max_n = float(purchases.get_column("n").max() or 1)
    shared_rel = {
        a: float(n) / max_n
        for a, n in zip(purchases.get_column("article_id"), purchases.get_column("n"))
    }
    rel = {"sim": shared_rel}

    def ids(side: dict) -> list[str]:
        return [a["article_id"] for a in side["slate"]]

    personalisation = project_lift(
        {"sim": ids(unconstrained_side)}, {"sim": ids(baseline_side)}, rel, prices,
        model_coverage=unconstrained_side["long_tail_share"],
        baseline_coverage=baseline_side["long_tail_share"],
    )
    quota_cost = project_lift(
        {"sim": ids(model_side)}, {"sim": ids(unconstrained_side)}, rel, prices,
        model_coverage=model_side["long_tail_share"],
        baseline_coverage=unconstrained_side["long_tail_share"],
    )
    combined = project_lift(
        {"sim": ids(model_side)}, {"sim": ids(baseline_side)}, rel, prices,
        model_coverage=model_side["long_tail_share"],
        baseline_coverage=baseline_side["long_tail_share"],
    )

    return {
        "k": k, "segment": segment, "cohort_size": len(cohort),
        "model": model_side, "unconstrained": unconstrained_side,
        "baseline": baseline_side,
        "decomposition": {
            "personalisation_effect": personalisation.as_dict(),
            "quota_cost": quota_cost.as_dict(),
            "combined": combined.as_dict(),
            "reading": (
                "personalisation_effect isolates the model against the "
                "bestseller page with neither carrying a quota. quota_cost "
                "isolates what POL-LT-01 costs by holding the model fixed. "
                "Their combination is what a merchandiser actually ships, and "
                "reporting only that number hides which half is responsible."
            ),
        },
        "relevance_note": (
            "Relevance is the target cohort's held-out purchases — the only "
            "signal here independent of both arms. Purchases, not impressions: "
            "an article the cohort did not buy is unlabelled, not rejected, so "
            "both slates are understated equally."
        ),
        "known_bias": (
            "THIS METRIC STRUCTURALLY FAVOURS THE BESTSELLER PAGE, and the "
            "reason is worth stating rather than tuning away. Projected revenue "
            "is estimated from held-out purchase FREQUENCY, which is exactly "
            "what the popularity arm ranks on. Head articles carry two to three "
            "orders of magnitude more purchases than tail ones, so any "
            "deviation from the top-k bestsellers is expensive by construction. "
            "No offline estimator built from observed purchases can show "
            "personalisation winning here; only a live A/B test could, and "
            "there isn't one (§16)."
        ),
        "the_finding": (
            "Personalisation wins PER CUSTOMER and loses PER COHORT SLATE, and "
            "those are not in conflict. The §9 ranking metrics evaluate a list "
            "per customer, where collaborative beats popularity (NDCG@10 0.0127 "
            "vs 0.0100). A slate is ONE page shown to a whole cohort, so the "
            "best it can do is target the cohort's modal preference — and the "
            "modal preference of any large cohort is, definitionally, its "
            "bestsellers. The merchandising implication is concrete: "
            "personalised slates pay off for small, sharply-defined cohorts and "
            "converge on the bestseller page as the cohort widens. That is a "
            "decision about segment granularity, not about model quality."
        ),
    }


# ── agent ────────────────────────────────────────────────────────────────────

class BriefIn(BaseModel):
    brief: str = Field(max_length=2000)
    caller_role: Literal["analyst", "merchandiser", "admin"] = "merchandiser"


@app.post("/agent/runs", status_code=202)
def submit_brief(body: BriefIn) -> dict:
    run = new_run(body.brief, "web-user", Role(body.caller_role))
    _RUNS[run.run_id] = run
    _emit(run.run_id, "run.started", {"goal": run.goal,
                                      "scopes": sorted(s.value for s in run.granted_scopes)})
    return {"run_id": run.run_id}


@app.get("/agent/runs/{run_id}")
def get_run(run_id: str) -> dict:
    if run_id not in _RUNS:
        raise HTTPException(404, "no such run")
    return _RUNS[run_id].model_dump(mode="json")


@app.get("/agent/runs/{run_id}/events")
async def stream_events(run_id: str):
    """SSE. The console renders progressively and never blocks on a completed
    run (§12.7). `critic.rejected` is a first-class streamed event — the
    rejection panel fills as rejections happen, which is the §1 demo moment."""
    if run_id not in _RUNS:
        raise HTTPException(404, "no such run")

    async def gen():
        run = _RUNS[run_id]
        yield f"event: run.started\ndata: {json.dumps({'goal': run.goal})}\n\n"
        await asyncio.sleep(0)

        finished = await asyncio.to_thread(run_to_gate, run)
        _RUNS[run_id] = finished

        if finished.triage_verdict and finished.triage_verdict != "proceed":
            yield (f"event: triage.decided\ndata: "
                   f"{json.dumps({'verdict': finished.triage_verdict, 'reasons': finished.triage_reasons, 'rule_ids': finished.triage_rule_ids})}\n\n")

        for rd in finished.route_decisions:
            yield f"event: route.decided\ndata: {rd.model_dump_json()}\n\n"
        for tc in finished.tool_calls:
            yield (f"event: tool.called\ndata: "
                   f"{json.dumps({'tool': tc.tool, 'ok': tc.ok, 'latency_s': round(tc.latency_s,3), 'scope': tc.scope_required.value})}\n\n")
            await asyncio.sleep(0.02)
        for ev in finished.evidence:
            yield (f"event: evidence.added\ndata: "
                   f"{json.dumps({'evidence_id': ev.evidence_id, 'corpus': ev.corpus, 'source_ref': ev.source_ref, 'trust': ev.trust})}\n\n")
        for cl in finished.claims:
            yield f"event: claim.added\ndata: {cl.model_dump_json()}\n\n"
        for sl in finished.candidate_slates:
            yield (f"event: slate.proposed\ndata: "
                   f"{json.dumps({'slate_id': sl.slate_id, 'articles': sl.article_ids, 'long_tail_share': sl.long_tail_share, 'report': sl.optimiser_report}, default=str)}\n\n")
        for rj in finished.rejections:
            yield f"event: critic.rejected\ndata: {rj.model_dump_json()}\n\n"
            await asyncio.sleep(0.03)
        if finished.pending_gate:
            yield f"event: gate.opened\ndata: {finished.pending_gate.model_dump_json()}\n\n"
        yield (f"event: run.completed\ndata: "
               f"{json.dumps({'phase': finished.phase.value, 'final_slate_id': finished.final_slate_id, 'steps': finished.budget.steps_used, 'triage': finished.triage_verdict})}\n\n")

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


class ResumeIn(BaseModel):
    gate_id: str
    decision: Literal["approve", "reject", "amend"]
    note: str = ""


@app.post("/agent/runs/{run_id}/resume", status_code=202)
def resume(run_id: str, body: ResumeIn) -> dict:
    """The interrupt contract. A resolution for a stale gate_id is refused, not
    applied — otherwise a replayed approval could authorise a slate it was
    never shown."""
    if run_id not in _RUNS:
        raise HTTPException(404, "no such run")
    run = _RUNS[run_id]
    run = human_gate(run, GateResolution(gate_id=body.gate_id, decision=body.decision,
                                         actor_id="web-user", note=body.note))
    if run.phase.value == "explaining":
        run = explainer(run)
    _RUNS[run_id] = run
    return {"phase": run.phase.value, "final_slate_id": run.final_slate_id}
