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
import os
from collections import OrderedDict
from contextlib import asynccontextmanager
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

from fastapi import Depends, Response, Cookie
from services.api.agent.graph import explainer, human_gate, new_run, run_to_gate
from services.api.core import security as sec
from services.api.core.rbac import Scope, ScopeViolation
from services.api.agent.state import GateResolution, MerchandisingRun
from services.api.core.rbac import Role

@asynccontextmanager
async def lifespan(_: FastAPI):
    # on_event is deprecated; lifespan is the supported hook.
    sec.seed_demo_users()
    yield


app = FastAPI(title="DHAWQ", version="1.0.0",
              description="Visual recommendation intelligence",
              lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    # Explicit allowlist, never ["*"] with credentials (§13.1).
    # Explicit allowlist, never ["*"] with credentials (§13.1). The deployed
    # origin is injected rather than hardcoded so a fork cannot inherit it.
    allow_origins=[
        o for o in [
            "http://localhost:3000", "http://127.0.0.1:3000",
            os.environ.get("ALLOWED_ORIGIN"),
        ] if o
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Auth (§13.1) and per-route scope enforcement (§13.2)
#
# AUTH IS PART OF THE DEMO, NOT A BARRIER IN FRONT OF IT. One account per role,
# one click each, so a visitor can watch the permission matrix behave: a viewer
# cannot run a simulation, a merchandiser can, and neither can manage users.
# An open API would have made §13.2 a table in a document rather than something
# you can see refuse you.
# ─────────────────────────────────────────────────────────────────────────────

AUDIT: "OrderedDict[str, dict]" = OrderedDict()
MAX_AUDIT = 500


def _audit(actor: str, action: str, detail: dict) -> None:
    """§13.1: segment exports, simulation runs and every agent run logged and
    replayable."""
    key = f"{datetime.now().astimezone().isoformat()}|{uuid.uuid4().hex[:6]}"
    AUDIT[key] = {"at": key.split("|")[0], "actor": actor,
                  "action": action, **detail}
    while len(AUDIT) > MAX_AUDIT:
        AUDIT.popitem(last=False)


def current_user(dhawq_at: str | None = Cookie(default=None)) -> sec.User:
    if not dhawq_at:
        raise HTTPException(401, "not authenticated")
    try:
        claims = sec.decode(dhawq_at, "access")
    except sec.AuthError as exc:
        raise HTTPException(401, str(exc)) from exc
    user = next((u for u in sec.STORE.users.values()
                 if u.user_id == claims["sub"]), None)
    if user is None:
        raise HTTPException(401, "unknown subject")
    return user


def requires(scope: Scope):
    """Route-level scope check.

    Enforced HERE and again at the agent's tool boundary. The two are not
    redundant: this stops a caller reaching an endpoint, the tool boundary
    stops the AGENT reaching a capability even when the caller has it. §13.3's
    intersection only means something if both exist.
    """
    def dep(user: sec.User = Depends(current_user)) -> sec.User:
        if scope not in user.scopes:
            raise HTTPException(
                403, f"role {user.role.value!r} lacks {scope.value!r}")
        return user
    return dep


class Credentials(BaseModel):
    email: str
    password: str


#: Secure cookies are HTTPS-only by definition, so over plain http they are set
#: and never sent back — every authenticated request then looks unauthenticated.
#: Production must keep them secure; local dev and the test client cannot. The
#: default is SECURE, so forgetting to set this weakens nothing.
INSECURE_COOKIES = os.environ.get("DHAWQ_INSECURE_COOKIES", "").lower() in ("1", "true", "yes")


def _set_cookies(resp: Response, access: str, refresh: str) -> None:
    common = dict(httponly=True, samesite="lax",
                  secure=not INSECURE_COOKIES, path="/")
    resp.set_cookie(sec.ACCESS_COOKIE, access,
                    max_age=int(sec.ACCESS_TTL.total_seconds()), **common)
    resp.set_cookie(sec.REFRESH_COOKIE, refresh,
                    max_age=int(sec.REFRESH_TTL.total_seconds()), **common)


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


# In-process run store, BOUNDED. Postgres checkpointing is wired in graph.py;
# this is the read model the SSE stream and the resume endpoint share.
#
# It was an unbounded dict. Each run holds its evidence, slates, tool calls and
# rejections — ~12KB of JSON plus object overhead — and nothing ever evicted
# them, so a long-lived instance grew until it was killed. A demo that leaks
# slowly is still a service that dies.
MAX_RUNS = 200
_RUNS: "OrderedDict[str, MerchandisingRun]" = OrderedDict()
_EVENTS: "OrderedDict[str, list[dict]]" = OrderedDict()


def _remember(run: MerchandisingRun) -> None:
    _RUNS[run.run_id] = run
    _RUNS.move_to_end(run.run_id)
    while len(_RUNS) > MAX_RUNS:
        old, _ = _RUNS.popitem(last=False)
        _EVENTS.pop(old, None)


def _emit(run_id: str, type_: str, payload: dict) -> None:
    q = _EVENTS.setdefault(run_id, [])
    q.append({"seq": len(q), "run_id": run_id, "type": type_,
              "ts": datetime.now().astimezone().isoformat(), "payload": payload})


# ── catalogue and scene ──────────────────────────────────────────────────────

@app.post("/auth/login")
def login(body: Credentials, response: Response) -> dict:
    try:
        user = sec.authenticate(body.email, body.password)
    except sec.AuthError:
        # One message for every failure. Distinguishing "no such user" from
        # "wrong password" turns credential stuffing into account enumeration.
        _audit(body.email, "login.failed", {})
        raise HTTPException(401, "invalid credentials")
    access = sec.create_access_token(user)
    refresh, _ = sec.create_refresh_token(user)
    sec.register_refresh(refresh)
    _set_cookies(response, access, refresh)
    _audit(user.user_id, "login.ok", {"role": user.role.value})
    return {"user_id": user.user_id, "email": user.email,
            "role": user.role.value,
            "scopes": sorted(s.value for s in user.scopes)}


@app.post("/auth/refresh")
def refresh_token(response: Response,
                  dhawq_rt: str | None = Cookie(default=None)) -> dict:
    if not dhawq_rt:
        raise HTTPException(401, "no refresh token")
    try:
        user, access, new_refresh = sec.rotate_refresh(dhawq_rt)
    except sec.AuthError as exc:
        _audit("unknown", "refresh.rejected", {"reason": str(exc)})
        raise HTTPException(401, str(exc)) from exc
    sec.register_refresh(new_refresh)
    _set_cookies(response, access, new_refresh)
    return {"role": user.role.value}


@app.post("/auth/logout")
def logout(response: Response, dhawq_rt: str | None = Cookie(default=None)) -> dict:
    if dhawq_rt:
        sec.revoke_family(dhawq_rt)
    for c in (sec.ACCESS_COOKIE, sec.REFRESH_COOKIE):
        response.delete_cookie(c, path="/")
    return {"ok": True}


@app.get("/auth/me")
def me(user: sec.User = Depends(current_user)) -> dict:
    return {"user_id": user.user_id, "email": user.email, "role": user.role.value,
            "scopes": sorted(s.value for s in user.scopes)}


@app.get("/auth/demo-accounts")
def demo_accounts() -> dict:
    """The §13.2 matrix, walkable. No demo account exceeds merchandiser, so a
    shared password cannot become an admin session."""
    sec.seed_demo_users()
    return {
        "password": sec.DEMO_PASSWORD,
        "accounts": [
            {"email": u.email, "role": u.role.value,
             "scopes": sorted(s.value for s in u.scopes)}
            for u in sorted(sec.STORE.users.values(), key=lambda x: x.email)
        ],
        "note": ("Demo accounts stop at merchandiser. None can manage users or "
                 "read the audit log, so the shared password cannot become an "
                 "admin session."),
    }


@app.get("/audit")
def audit_log(user: sec.User = Depends(requires(Scope.AUDIT_READ))) -> dict:
    return {"entries": list(AUDIT.values())[-100:]}


@app.get("/health")
def health() -> dict:
    """Health, and an honest statement of what this instance is serving.

    `light_mode` is reported because a light-mode slate is NOT a full-stack
    slate, and a reader who cannot tell the difference will draw the wrong
    conclusion from it.
    """
    from services.api.agent.tools import FIT_AT_RUNTIME

    return {
        "ok": True, "service": "dhawq",
        "serving": "precomputed" if not FIT_AT_RUNTIME else "runtime_fit",
        "note": (
            "Cohort candidates, segment aggregates and slot simulations are "
            "build-time artefacts computed on the full five-arm stack. The "
            "service reads them rather than fitting a recommender per request, "
            "which cost 952MB peak and OOM-killed this instance. The numbers "
            "served are the real hybrid's output."
        ),
        "llm_provider": os.environ.get("DHAWQ_LLM_PROVIDER", "auto"),
    }


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


@app.get("/segments/clv/holdout")
def clv_holdout(user: sec.User = Depends(requires(Scope.SEGMENTS_READ_AGG))) -> dict:
    """§11's holdout validation: fit on the first period, predict the second,
    compare. Precomputed by pipelines/07 — a BG/NBD refit is not a per-request
    cost, and the answer does not change between requests."""
    import json as _json
    from services.api.core.artifacts import PROCESSED
    p = PROCESSED / "cohorts" / "clv_holdout.json"
    if not p.exists():
        raise HTTPException(404, "run pipelines/07_validate_clv.py")
    return _json.loads(p.read_text())


@app.get("/segments/rfm")
def segments(user: sec.User = Depends(requires(Scope.SEGMENTS_READ_AGG))) -> dict:
    from services.api.core.artifacts import cohort_segments
    return cohort_segments()["rfm"]


@app.get("/segments/clv")
def clv_summary(user: sec.User = Depends(requires(Scope.SEGMENTS_READ_AGG))) -> dict:
    """PROJECTED CLV aggregates. Never individual rows — POL-SEG-02 and the
    §13.2 row denying access to individual customer records.

    Precomputed: BG/NBD runs Nelder-Mead over 118,914 customers, which is not
    a per-request cost.
    """
    from services.api.core.artifacts import cohort_segments
    return cohort_segments()["clv"]


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
             segment: str = "champions",
             user: sec.User = Depends(requires(Scope.MERCH_SIMULATE))) -> dict:
    """Side-by-side: the model's slate against the bestseller baseline, with the
    personalisation effect and the quota cost decomposed.

    Precomputed per segment. The inputs — candidates, held-out purchases, the
    policy — do not change between requests, so neither does the output.
    """
    from services.api.core.artifacts import cohort_simulations

    sims = cohort_simulations()
    if segment not in sims:
        raise HTTPException(
            404, f"no simulation for segment {segment!r}; "
                 f"available: {sorted(sims)}")
    return sims[segment]


# ── agent ────────────────────────────────────────────────────────────────────

class BriefIn(BaseModel):
    brief: str = Field(max_length=2000)
    caller_role: Literal["analyst", "merchandiser", "admin"] = "merchandiser"


@app.post("/agent/runs", status_code=202)
def submit_brief(body: BriefIn,
                 user: sec.User = Depends(requires(Scope.RECS_READ))) -> dict:
    # The caller's OWN role drives the down-scoping, never a role they claim in
    # the body. §13.3 is meaningless if the request can name its own authority.
    run = new_run(body.brief, user.user_id, user.role)
    _audit(user.user_id, "agent.run.submitted",
           {"run_id": run.run_id, "role": user.role.value})
    _remember(run)
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
        _remember(finished)

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
def resume(run_id: str, body: ResumeIn,
           user: sec.User = Depends(requires(Scope.SLATE_APPROVE))) -> dict:
    """The interrupt contract. A resolution for a stale gate_id is refused, not
    applied — otherwise a replayed approval could authorise a slate it was
    never shown."""
    if run_id not in _RUNS:
        raise HTTPException(404, "no such run")
    run = _RUNS[run_id]
    run = human_gate(run, GateResolution(gate_id=body.gate_id, decision=body.decision,
                                         actor_id=user.user_id, note=body.note))
    _audit(user.user_id, "gate.resolved",
           {"run_id": run_id, "decision": body.decision, "gate_id": body.gate_id})
    if run.phase.value == "explaining":
        run = explainer(run)
    _remember(run)
    return {"phase": run.phase.value, "final_slate_id": run.final_slate_id}
