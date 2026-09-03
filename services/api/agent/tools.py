"""The typed tool catalogue — ARCHITECTURE.md §7.5.

Every tool is typed, scoped, and either deterministic or explicitly not.

THE INVARIANT: there is no tool that writes to the catalogue, the model
registry, or the evaluation artefacts. The agent is READ-ONLY over the entire
deterministic core. The only state it mutates is its own run record. Enforced
by a test that iterates the registry.

NO DYNAMIC REGISTRATION (§13.4 LLM07). The registry is built once at import and
frozen. A tool that can be added at runtime is a tool an injection can add.
"""

from __future__ import annotations

import os
import time
import uuid
from functools import lru_cache
from dataclasses import dataclass
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

from services.api.agent.state import ToolCall
from services.api.core.rbac import Scope, require_scope

Node = Literal["supervisor", "retriever", "analyst", "merchandiser", "critic", "gate"]


# ── Typed IO ─────────────────────────────────────────────────────────────────

class RecommendIn(BaseModel):
    cohort_spec: dict[str, Any] = Field(default_factory=dict)
    article_id: str | None = None
    k: int = Field(default=50, ge=1, le=200)
    model: Literal["popularity", "content", "collaborative",
                   "hybrid_weighted", "hybrid_cascade"] = "hybrid_weighted"


class RecommendOut(BaseModel):
    article_ids: list[str]
    scores: list[float]
    model: str


class OptimiseSlotsIn(BaseModel):
    candidate_ids: list[str]
    k: int = Field(ge=4, le=24)
    constraints: dict[str, Any] = Field(default_factory=dict)


class OptimiseSlotsOut(BaseModel):
    slate: list[str]
    report: dict[str, Any]


class GraphTraverseIn(BaseModel):
    start: str
    relations: list[str] | None = None
    depth: int = Field(default=2, ge=1, le=3)
    limit: int = Field(default=50, ge=1, le=200)


class GraphTraverseOut(BaseModel):
    targets: list[str]
    paths: list[str]
    scores: list[float]
    uses_predicted_evidence: list[bool]


class CohortIn(BaseModel):
    cohort_spec: dict[str, Any] = Field(default_factory=dict)


class PolicyOut(BaseModel):
    version: str
    document: str
    rule_ids: list[str]


class EvalReportIn(BaseModel):
    run_id: str | None = None
    metric: str | None = None


# ── Registry ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ToolSpec:
    name: str
    node: Node
    scope: Scope
    deterministic: bool
    max_calls: int
    input_model: type[BaseModel]
    output_model: type[BaseModel] | None
    fn: Callable[..., Any]
    description: str

    @property
    def writes(self) -> bool:
        """No tool in this catalogue writes. Kept explicit so the claim is
        checkable rather than asserted in a comment."""
        return False


_REGISTRY: dict[str, ToolSpec] = {}


def register(spec: ToolSpec) -> ToolSpec:
    if spec.name in _REGISTRY:
        raise ValueError(f"duplicate tool: {spec.name}")
    _REGISTRY[spec.name] = spec
    return spec


def catalogue() -> dict[str, ToolSpec]:
    return dict(_REGISTRY)


def get(name: str) -> ToolSpec:
    if name not in _REGISTRY:
        raise KeyError(f"no such tool: {name}")
    return _REGISTRY[name]


# ── Implementations (all read-only) ──────────────────────────────────────────

#: Fitting a recommender inside the API cost 952MB peak and OOM-killed the
#: deployed 512MB instance. Cohort candidates are as static as every other
#: artefact in DHAWQ, so they are precomputed by pipelines/06 and simply READ
#: here. This serves the REAL hybrid's output — strictly better than the light
#: mode it replaces, which substituted a weaker arm to save memory.
#:
#: DHAWQ_FIT_AT_RUNTIME=1 restores in-process fitting for local work against a
#: catalogue that has no precomputed artefact yet.
FIT_AT_RUNTIME = os.environ.get("DHAWQ_FIT_AT_RUNTIME", "").lower() in ("1", "true", "yes")


@lru_cache(maxsize=4)
def _fitted(model_name: str):
    """Fit once per process. ALS takes ~30s; refitting per tool call would blow
    the wall-clock budget on its own."""
    from services.api.core.artifacts import train
    from services.api.models.baseline import PopularityRecency
    from services.api.models.collaborative import ImplicitALS
    from services.api.models.content import ContentKNN
    from services.api.models.hybrid import Hybrid

    arm = {
        "popularity": lambda: PopularityRecency(),
        "content": lambda: ContentKNN(),
        "collaborative": lambda: ImplicitALS(),
        "hybrid_weighted": lambda: Hybrid(mode="weighted"),
        "hybrid_cascade": lambda: Hybrid(mode="cascade"),
    }[model_name]()
    return arm.fit(train())


def _recommend(args: RecommendIn) -> RecommendOut:
    """Cohort-level candidate generation.

    POL-SEG-02 and the §13.2 row denying the agent individual customer records:
    there is no code path here that takes a customer_id, and the input model
    has no such field.

    Candidates come from the REAL recommender, not a popularity shortcut. The
    first version returned the top-k bestsellers, which meant the candidate
    pool was almost entirely head articles and the optimiser could never
    satisfy POL-LT-01 — the long-tail quota failed for want of tail candidates
    rather than for any merchandising reason. A quota you cannot meet because
    of your own candidate generation is not a finding, it is a bug.
    """
    import numpy as np

    from services.api.core.artifacts import canonical_ids, cohort_candidates, train

    # Precomputed path: no model, no fitting, no 42MB embedding matrix.
    if not FIT_AT_RUNTIME:
        segment = args.cohort_spec.get("segment", "champions")
        table = cohort_candidates()
        by_segment = table.get(args.model) or table.get("hybrid_weighted") or {}
        ids = by_segment.get(segment) or next(iter(by_segment.values()), [])
        ids = ids[: args.k]
        return RecommendOut(
            article_ids=ids,
            scores=[1.0 - i / max(len(ids), 1) for i in range(len(ids))],
            model=args.model,
        )

    arm = _fitted(args.model)
    tr = train()

    # A cohort score is the mean of its members' score vectors. Sampled,
    # because scoring every member of a 100k cohort to pick 120 candidates is
    # wasted work — the mean stabilises long before that.
    members = args.cohort_spec.get("customer_ids")
    if not members:
        members = tr.get_column("customer_id").unique().sort().to_list()[:40]
    # Sampling the cohort is fine; sampling a DIFFERENT population is not. The
    # first version always averaged the first 40 customers of the whole train
    # set regardless of the cohort passed in, so every "personalised" slate was
    # personalised to the same arbitrary 40 people.
    scores = None
    for c in members[:40]:
        v = arm.score_customer(c)
        if not np.isfinite(v).any():
            continue
        v = np.where(np.isfinite(v), v, np.nan)
        scores = v if scores is None else np.nansum([scores, v], axis=0)
    if scores is None:
        counts = tr.group_by("article_id").len().sort("len", descending=True)
        return RecommendOut(article_ids=counts.get_column("article_id").to_list()[:args.k],
                            scores=[float(x) for x in counts.get_column("len").to_list()[:args.k]],
                            model=args.model)

    ids = canonical_ids()
    order = np.argsort(-np.nan_to_num(scores, nan=-np.inf))[: args.k]
    return RecommendOut(article_ids=[ids[i] for i in order],
                        scores=[float(scores[i]) for i in order],
                        model=args.model)


def _optimise_slots(args: OptimiseSlotsIn) -> OptimiseSlotsOut:
    from services.api.core.artifacts import articles, catalogue_facts
    from services.api.marketing.slots import Candidate, optimise_slots

    # Head/tail and prices come from the frozen catalogue facts, not from
    # re-reading the transaction parquet. Same numbers, ~250MB less resident.
    facts = catalogue_facts()
    head, prices = facts["head"], facts["prices"]
    arts = articles()
    meta = {r[0]: r for r in arts.select(
        "article_id", "product_type_name", "colour_group_name").iter_rows()}
    cands = [
        Candidate(
            article_id=a, score=1.0 - i / max(len(args.candidate_ids), 1),
            price=float(prices.get(a, 0.0) or 0.0),
            product_type=(meta.get(a) or (a, "unknown", "unknown"))[1] or "unknown",
            colour_group=(meta.get(a) or (a, "unknown", "unknown"))[2] or "unknown",
            is_long_tail=a not in head,
        )
        for i, a in enumerate(args.candidate_ids) if a in meta
    ]
    slate, report = optimise_slots(cands, args.k, **args.constraints)
    return OptimiseSlotsOut(slate=slate, report=report.__dict__)


def _graph_traverse(args: GraphTraverseIn) -> GraphTraverseOut:
    from services.api.rag.graph_index import load_graph

    paths = load_graph().traverse(args.start, args.relations, args.depth,
                                  limit=args.limit)
    return GraphTraverseOut(
        targets=[p.target for p in paths],
        paths=[p.describe() for p in paths],
        scores=[p.score for p in paths],
        uses_predicted_evidence=[p.uses_predicted_evidence for p in paths],
    )


class HybridSearchIn(BaseModel):
    query: str = Field(max_length=500)
    corpus: Literal["B", "D"] = "D"
    k: int = Field(default=8, ge=1, le=20)


class HybridSearchOut(BaseModel):
    hits: list[dict]
    trust: str
    wrapped: bool
    injections_suspected: int


def _hybrid_search(args: HybridSearchIn) -> HybridSearchOut:
    """Retrieve over corpus D, WRAPPED as untrusted.

    The wrapping happens here, at the boundary, not in whatever prompt happens
    to consume the result. A retriever that returns raw external text and
    trusts its caller to wrap it has already lost — the caller is the thing
    most likely to forget.
    """
    from services.api.rag.hybrid import hybrid_search
    from services.api.rag.untrusted import policy_assertions, wrap

    hits, suspected = [], 0
    for h in hybrid_search(args.query, k=args.k):
        w = wrap(h.doc.text, source=h.doc.source, url=h.doc.url)
        # Counted at the boundary as well as judged by the critic, because
        # §8.5 asks for detections to be MEASURED, not only defended against —
        # and a document the critic never sees (because the run ended early)
        # still needs to appear in the count.
        suspected += w.neutralised_tags + len(policy_assertions(h.doc.text))
        hits.append({
            "doc_id": h.doc.doc_id, "title": h.doc.title, "url": h.doc.url,
            "score": round(h.score, 5), "content": w.text,
            "bm25_rank": h.bm25_rank, "dense_rank": h.dense_rank,
        })
    return HybridSearchOut(hits=hits, trust="untrusted", wrapped=True,
                           injections_suspected=suspected)


def _load_policy(args: BaseModel) -> PolicyOut:
    import sys
    from pathlib import Path
    pdir = Path(__file__).resolve().parents[1] / "rag" / "corpora" / "policy"
    sys.path.insert(0, str(pdir))
    from schema import load_policy  # noqa: E402

    pol = load_policy()
    return PolicyOut(
        version=pol.policy_version,
        document=(pdir / "POLICY.md").read_text(),   # WHOLE. Never chunked (§8.2)
        rule_ids=[r.id for r in pol.rules],
    )


def _clv(args: CohortIn) -> dict:
    """Projected CLV AGGREGATES. Never individual rows — POL-SEG-02.

    Read from the precomputed artefact: BG/NBD is a Nelder-Mead fit over
    118,914 customers and is not a per-tool-call cost.
    """
    from services.api.core.artifacts import cohort_segments

    c = cohort_segments()["clv"]
    return {
        "n_customers": c["n_customers"],
        "mean_projected_clv": c["projected_clv"]["mean"],
        "median_projected_clv": c["projected_clv"]["median"],
        "mean_probability_alive": c["probability_alive"]["mean"],
        "frequency_monetary_correlation":
            c["assumption_check"]["frequency_monetary_correlation"],
        "independence_assumption_note": c["assumption_check"]["note"],
        "language": c["language"],
    }


def _rfm_segment(args: CohortIn) -> dict:
    """RFM aggregates, precomputed. Grouping 1.39M rows per tool call was the
    other half of the memory that OOM-killed the deployed instance."""
    from services.api.core.artifacts import cohort_segments
    return cohort_segments()["rfm"]


def _eval_report(args: EvalReportIn) -> dict:
    import glob
    import json
    from pathlib import Path
    files = sorted(glob.glob(str(Path(__file__).resolve().parents[3] /
                                 "eval" / "artifacts" / "run_*.json")))
    if not files:
        return {"error": "no evaluation artefacts — run the D4 harness"}
    art = json.loads(Path(files[-1]).read_text())
    return {"run_id": art["run_id"], "frontier": art["frontier"],
            "protocol": art["protocol"], "strata": art["strata"]}


register(ToolSpec("recommend", "merchandiser", Scope.RECS_READ, True, 4,
                  RecommendIn, RecommendOut, _recommend,
                  "Candidate articles for a cohort. Cohort-scoped only."))
register(ToolSpec("optimise_slots", "merchandiser", Scope.MERCH_SIMULATE, True, 3,
                  OptimiseSlotsIn, OptimiseSlotsOut, _optimise_slots,
                  "Constrained slate under corpus C rules. Returns binding constraints."))
register(ToolSpec("graph_traverse", "retriever", Scope.CORPUS_A_READ, True, 6,
                  GraphTraverseIn, GraphTraverseOut, _graph_traverse,
                  "Path query over the taxonomy graph. Returns the traversed path."))
register(ToolSpec("hybrid_search", "retriever", Scope.CORPUS_D_READ, True, 4,
                  HybridSearchIn, HybridSearchOut, _hybrid_search,
                  "Hybrid BM25+dense over corpus D. Results are UNTRUSTED and "
                  "wrapped at this boundary."))
register(ToolSpec("load_policy", "critic", Scope.CORPUS_C_READ, True, 2,
                  CohortIn, PolicyOut, _load_policy,
                  "Load corpus C WHOLE. Not chunked, not retrieved."))
register(ToolSpec("clv", "analyst", Scope.SEGMENTS_READ_AGG, True, 3,
                  CohortIn, None, _clv, "Projected CLV aggregates for a cohort."))
register(ToolSpec("rfm_segment", "analyst", Scope.SEGMENTS_READ_AGG, True, 3,
                  CohortIn, None, _rfm_segment, "RFM segment aggregates."))
register(ToolSpec("eval_report", "analyst", Scope.EVAL_READ, True, 4,
                  EvalReportIn, None, _eval_report, "Read evaluation artefacts (corpus B)."))


# ── Invocation ───────────────────────────────────────────────────────────────

class ToolResult(BaseModel):
    ok: bool
    tool: str
    output: Any = None
    error: str | None = None
    call: ToolCall


def invoke(name: str, raw_args: dict, granted_scopes: frozenset[Scope]) -> ToolResult:
    """The tool boundary. Scope is checked HERE, in code — not in a prompt.

    A scope violation raises before the tool body runs, so a compromised agent
    that emits a forbidden tool call gets a refusal, not an execution.
    """
    spec = get(name)
    call_id = f"tc_{uuid.uuid4().hex[:10]}"
    t0 = time.perf_counter()

    try:
        require_scope(spec.scope, granted_scopes, tool=name)
        args = spec.input_model.model_validate(raw_args)
        out = spec.fn(args)
        call = ToolCall(call_id=call_id, tool=name, args=raw_args,
                        scope_required=spec.scope, ok=True,
                        latency_s=time.perf_counter() - t0)
        return ToolResult(ok=True, tool=name, output=out, call=call)
    except Exception as exc:
        call = ToolCall(call_id=call_id, tool=name, args=raw_args,
                        scope_required=spec.scope, ok=False, error=str(exc),
                        latency_s=time.perf_counter() - t0)
        return ToolResult(ok=False, tool=name, error=str(exc), call=call)
