# DHAWQ — Executable Plan

## Context

`ARCHITECTURE.md` (1274 lines) is the source of truth and is not re-litigated here.
This document turns it into something buildable: concrete file paths, type shapes,
decision functions, thresholds, and a build order. Everything below either
implements a stated §-decision or fills a gap the architecture left open.

Where I filled a gap, it is marked **[GAP]**. Where I am pushing back, it is in §13
and nowhere else — the body of this plan assumes the architecture as written.

Repo state: empty apart from `ARCHITECTURE.md`, `.gitignore`, `data/`.
`data/` is not touched this session (H&M 28.7GB still downloading).

---

## 0. The one change to §14, and its three consequences

Golden-set authoring moves from D11 to sit alongside D8, **before** the agent exists.
This is right, and it is more load-bearing than it first looks. Writing the briefs
first means D8 produces **three** artefacts, not one:

1. **`eval/golden/briefs_v1.yaml`** — 60 stratified, hand-labelled briefs.
2. **`services/api/agent/tools.py` as signatures only** — no bodies. The briefs are
   the requirements document for the tool catalogue. If a brief needs a capability,
   a typed signature appears; if no brief needs it, no tool exists. This is how you
   avoid a catalogue built from imagination.
3. **The numeric thresholds for critic criteria 3, 4 and 8** — these cannot be
   invented. The 8 constraint-conflicting briefs *define* what "breaches the
   long-tail quota" means; the 6 unanswerables *define* the evidence-coverage
   floor below which confidence is suppressed. Today those numbers have no source.
   After D8 they have one.

**[GAP] Corpus C is not in §14 at all.** ~15 pages of merchandising policy must
exist before the golden set (you cannot write "must escalate on quota conflict"
without a quota) and before the critic. It gets its own slot, D1.5, 3h.

**[GAP] The FPI→H&M attribute classifier is not in §14 either.** §3 specifies it,
§8.3 depends on it for `has_season` / `has_usage` edges. ~5h, unbudgeted. Folded
into D6 with the hours added, or cut (§13).

---

## 0.5 Decisions taken at D0

Three open questions were settled before planning the build. They are recorded here
because each one changes code, not just intent.

### D0-1 · Four specialists, then measure

Build all four nodes as §7.4 specifies. At D11, run the golden set twice — with the
Analyst/Merchandiser split and with them merged into one `Quant` node — and compare
`tool_selection_accuracy` and `step_efficiency`. Merge if the split buys nothing.

Cost: ~1h of extra eval config, and the node bodies must be written so a merge is a
graph-wiring change rather than a rewrite (shared tool-invocation helper, no
node-specific state fields). Payoff: the §7.4 claim becomes measured rather than
argued, and the honest viva answer is available either way.

### D0-2 · Minimise spend; tier the models

The instruction was free-tier-first, so the cost work is not a footnote. There is no
free tier on the Anthropic API, so the lever is spend reduction, in this order:

1. **Prompt-cache corpus C.** It is byte-stable and it is the largest single block
   in the run. Cache reads are a tenth of base input price. This is the single
   biggest saving and it costs nothing but correct prefix ordering.
2. **Tier the models by what each call is actually for.**

   | Node / call | Model | Why |
   |---|---|---|
   | Router shape classifier | `claude-haiku-4-5` | 6-way classification, `effort: low` |
   | Retriever, Explainer | `claude-sonnet-5` | extraction and narration, not judgement |
   | Supervisor decomposition | `claude-opus-5` | plan quality is what drifts |
   | Critic criteria 1, 2, 6, 7, 8 | `claude-opus-5` | the judgement you defend |

   Criteria 3, 4, 5, 9 are code and cost nothing. The model mix is disclosed in the
   README next to the numbers, per §10.2's disclosure discipline.
3. **Batch API (50% off) for everything single-shot.** The D12 re-ranker benchmark
   and the router-classifier eval are not interactive loops — they batch. The agent
   loop itself cannot.
4. **Cap the D12 re-ranker benchmark to a sampled 500 test users**, not the full
   test set. Re-ranking top-50 for every held-out user is quietly the largest token
   sink in the whole project and buys no extra statistical power at this scale. The
   sample size goes in the manifest.

Expected: **~$0.04–0.06 per brief**, a full 60-brief golden run around **$3**, and
total API spend across all 21 sessions including re-runs in the **$40–80** range.
The §10.5 target of `≤ $0.05` stands, but it is now a designed number with a stated
mechanism rather than an aspiration.

**Cache-invalidation constraint this imposes on §7.2:** the `goal` restated every
step is volatile text. In the system prompt it invalidates the corpus-C cache prefix
on every step and you pay full price throughout. It goes in the `messages` array,
after the last cache breakpoint. Tool definitions must also be emitted in a stable
sorted order — an unsorted tool list is a silent cache invalidator.

### D0-3 · Vercel (web) + Render (API + Postgres)

One Postgres carries pgvector, the catalogue, the LangGraph checkpointer and the
audit log.

**Design consequence — make the server surface as small as possible**, both for
free-tier headroom and for cold-start resilience. Everything precomputed becomes a
static asset on Vercel's CDN rather than a Python route:

- `atlas_*.ktx2`, `positions.bin`, `colours.bin`, `/space/manifest` — static
- `/evaluate/*` — served from frozen eval artefacts as static JSON, not computed
- `/merchandise/policy` — static (corpus C does not change at runtime)

That leaves only `/recs/*`, `/agent/*`, `/segments/*` and `/auth/*` needing a live
server. The gallery, the frontier plot and the evaluation report all render with the
API asleep — which is worth having independently of hosting.

**[VERIFY before D21]** Free tiers on managed hosts commonly spin services down
after inactivity and expire free databases after a fixed window. Both would be
serious for a graded live MVP: a 50-second cold start during grading, or a database
deleted mid-term. Check Render's current terms early, and budget either a keep-warm
cron or a paid starter tier for the grading window. This is cheap to handle in
advance and expensive to discover on submission day. It is checked at **D14**, not
D21, so there is slack to change host.

---

## 1. Repo layout

Matches §4 exactly where §4 is concrete; additions are marked.

```
dhawq/
├── apps/web/                                   # Next.js 16, App Router
│   ├── app/
│   │   ├── layout.tsx                          # theme script, font vars, dir=
│   │   ├── page.tsx                            # 3D embedding space
│   │   ├── product/[id]/page.tsx
│   │   ├── merchandise/page.tsx
│   │   ├── agent/page.tsx
│   │   ├── segments/page.tsx
│   │   └── evaluate/page.tsx
│   ├── components/space/                       # Scene, InstancedCloud, Atlas,
│   │   │                                       # LODPoints, NeighbourLines,
│   │   │                                       # FlyToController, Fallback2D
│   ├── components/product/
│   ├── components/agent/                       # TraceTimeline, RejectionCard,
│   │                                           # GatePanel, EvidencePopover
│   ├── components/ui/                          # restyled shadcn primitives
│   ├── lib/
│   │   ├── api.ts                              # typed client, generated types
│   │   ├── sse.ts                              # EventSource + Last-Event-ID
│   │   └── theme.ts
│   └── styles/tokens.css                       # ADDED — the token layer (§9)
├── services/api/
│   ├── main.py                                 # ADDED — app factory, middleware
│   ├── routers/                                # catalogue, recs, space, evaluate,
│   │                                           # segments, merchandise, agent, auth
│   ├── models/
│   │   ├── content.py  collaborative.py  hybrid.py  llm_reranker.py  baseline.py
│   ├── embed/
│   │   ├── extract.py  project.py  index.py
│   ├── agent/
│   │   ├── state.py  graph.py  tools.py  gates.py  trace.py
│   │   ├── prompts/                            # ADDED — versioned, hashed
│   │   └── nodes/                              # supervisor, retriever, analyst,
│   │                                           # merchandiser, critic, explainer
│   ├── rag/
│   │   ├── router.py  graph_index.py  hybrid.py  rerank.py
│   │   ├── untrusted.py  citations.py
│   │   └── corpora/                            # corpora live WITH the code that
│   │       ├── policy/                         # reads them, not at repo root
│   │       │   ├── policy.yaml                 # source of truth (D1.5)
│   │       │   ├── schema.py  render.py        # typed + generated
│   │       │   ├── POLICY.md                   # GENERATED — loads whole (§8.2)
│   │       │   └── manifest.json               # GENERATED — size vs threshold
│   │       └── external/<crawl_date>/          # corpus D, pinned, immutable
│   ├── evaluate/
│   │   ├── ranking.py  beyond_accuracy.py  bias.py  coldstart.py
│   │   ├── agent_eval.py  rag_eval.py  calibration.py
│   ├── marketing/
│   │   ├── rfm.py  clv.py  slots.py  lift.py
│   ├── core/
│   │   ├── security.py  rbac.py
│   │   ├── config.py                           # ADDED — pydantic-settings
│   │   ├── db.py                               # ADDED — engine, session, pgvector
│   │   └── schemas.py                          # ADDED — shared Pydantic DTOs
│   └── alembic/                                # ADDED — migrations
├── pipelines/                                  # ADDED — offline, build-time only.
│   ├── 01_subsample.py                         # NEVER imported by the API.
│   ├── 02_embed.py
│   ├── 03_project_umap.py
│   ├── 04_build_atlas.py
│   ├── 05_build_graph.py
│   ├── 06_train_attr_classifier.py
│   ├── 07_crawl_corpus_d.py
│   └── manifests/                              # every pipeline writes one
├── eval/
│   ├── golden/briefs_v1.yaml                   # + composition header
│   ├── redteam/injections_v1.yaml
│   ├── failures/<date>/
│   └── run.py                                  # one command, writes README table
├── tests/                                      # ADDED
│   ├── unit/                                   # deterministic core — 100% paths
│   ├── property/                               # slot optimiser, lift, RFM bands
│   ├── integration/                            # graph topology, gates, resume
│   └── invariants/                             # temporal split, atlas index, RBAC
├── docker-compose.yml                          # ADDED — postgres + pgvector
├── .github/workflows/ci.yml                    # ADDED — the five gates
└── data/                                       # gitignored
```

**Corpus location — revised at D1.5.** An earlier draft of this plan put the
corpora at repo root (`corpora/C_policy/`). They now live under
`services/api/rag/corpora/`, beside the code that reads them. Corpus C is not
inert data — it ships with a Pydantic schema, a generator and integrity checks,
and it is imported by the critic and the slot optimiser. Filing it at repo root
would have separated a module from its own tests and made the import direction
below harder to enforce.

**Two structural rules that keep §4's permission boundary real:**

- `pipelines/` is import-forbidden from `services/api/`. Enforced by a lint rule in
  CI. Anything the API needs from a pipeline arrives as a **frozen artefact plus a
  manifest**, never as a live call. This is what makes "build-time, never runtime"
  a property of the repo rather than a promise in a doc.
- `services/api/agent/` may import from `marketing/`, `models/`, `evaluate/`,
  `rag/` — but never the reverse. Enforced the same way. That import direction *is*
  the §4 diagram.

---

## 2. State model and Budget

`services/api/agent/state.py`. Specification, not implementation.

### Budget

Budget carries limits **and** consumption in one object so a node can ask
"can I afford this?" without reaching for a second field.

```
class Budget(BaseModel):
    model_config = ConfigDict(frozen=True)

    # Limits — set at run creation, never mutated
    max_steps: int = 24
    max_tokens: int = 250_000
    max_wall_clock_s: float = 120.0
    max_calls_per_tool: int = 4
    max_critic_rounds: int = 2
    max_retrieval_fanouts: int = 2

    # Consumption — advanced by .charge(), which returns a new Budget
    steps_used: int = 0
    tokens_used: int = 0
    started_at: datetime
    tool_call_counts: dict[str, int] = {}
    critic_rounds_used: int = 0

    def remaining(self) -> BudgetRemaining
    def would_exceed(self, *, steps=0, tokens=0, tool=None) -> BudgetBreach | None
    def charge(self, ...) -> "Budget"          # raises BudgetExhausted
```

Frozen + `charge()` returning a new instance means budget accounting survives
LangGraph's state merging without a mutable-shared-object bug. `would_exceed`
is the pre-flight check; nodes call it before an expensive action, not after.

`max_wall_clock_s = 120` against a stated p95 target of 25s gives 5× headroom —
the budget is a circuit breaker, not the SLO.

### The claim/evidence spine

```
class Evidence(BaseModel):
    evidence_id: str                    # stable, content-addressed: sha256(...)[:16]
    corpus: Literal["A", "B", "C", "D"]
    source_ref: str                     # node id / run_id / policy §  / url
    content: str
    retrieved_at: datetime
    trust: Literal["trusted", "untrusted"]
    injection_findings: list[Finding] = []
    produced_by: list[str] = []         # tool_call_ids

class Claim(BaseModel):
    claim_id: str
    text: str
    evidence_ids: list[str]
    kind: Literal["factual", "projected", "policy"]   # ADDED — see below

    @field_validator("evidence_ids")
    @classmethod
    def must_be_grounded(cls, v): ...    # exactly as §7.2
```

**[GAP] `kind` is an addition.** Critic criterion 6 ("asserts measured or causal
lift rather than projected") is much cheaper and more reliable to enforce against a
declared enum than against free text. `kind="projected"` claims get a lint pass in
code for causal verbs; the model declares the kind, code checks the phrasing. That
moves criterion 6 halfway out of judgement and into a rule — a small win, honestly
described as partial.

`evidence_id` is content-addressed rather than a counter, so the same policy
paragraph retrieved twice deduplicates by construction, and citation validity can
be checked without a lookup table.

### Run state

```
class MerchandisingRun(BaseModel):
    # Identity and authority
    run_id: str
    goal: str
    caller_id: str
    caller_role: Role
    granted_scopes: frozenset[Scope]
    budget: Budget

    # Accumulating — append-only, all with LangGraph reducers
    evidence:            Annotated[list[Evidence],   append_by_id]
    claims:              Annotated[list[Claim],      append_by_id]
    candidate_slates:    Annotated[list[Slate],      append_by_id]
    rejections:          Annotated[list[Rejection],  append]
    injections_detected: Annotated[list[Finding],    append]
    errors:              Annotated[list[RunError],   append]
    tool_calls:          Annotated[list[ToolCall],   append]      # ADDED
    route_decisions:     Annotated[list[RouteDecision], append]   # ADDED
    lineage: dict[str, list[str]] = {}

    # Control — overwritten, deliberately
    phase: Phase
    plan: list[SubTask] = []
    critic_rounds: int = 0
    pending_gate: GateRequest | None = None
    gate_history: list[GateResolution] = []
    confidence: Confidence | None = None
    final_slate_id: str | None = None
```

**`tool_calls` and `route_decisions` are additions, and they are not optional.**
Two tuning metrics in §10.1 — `tool_selection_accuracy` and
`retrieval_routing_accuracy` — are unmeasurable unless the run *records the
decision it made* in a form the golden set can be scored against. Retrofitting this
after D11 would mean re-running every brief. This is a direct downstream
consequence of moving the golden set earlier: writing the labels first shows you
what the state has to remember.

`ToolCall` carries `dedupe_key = sha256(tool_name + canonical_json(args))`, which
is what §7.8's tool-thrash mitigation actually keys on.

**On accumulate-never-overwrite:** it applies to the evidence spine and the audit
surfaces. `phase`, `critic_rounds` and `pending_gate` are control variables and are
overwritten by design — an append-only `phase` is not accumulation, it is a bug.
The distinction is: anything a user or auditor would want to see is append-only;
anything the graph needs to branch on is a scalar.

### Checkpointing

`langgraph.checkpoint.postgres.PostgresSaver`, wired at D9 (not later), thread id =
`run_id`. Two things this must get right from the start:

- The checkpointer's tables live in the **same** Postgres as pgvector and the
  catalogue, but in a separate schema (`agent_runs`). One database, one connection
  pool, one deploy.
- A resumed run re-validates `granted_scopes` against the caller's *current* role
  before continuing. A checkpoint is not a capability. Without this, a run
  checkpointed while its caller was a merchandiser resumes with those scopes after
  the caller is demoted — a real privilege-escalation path through the resume
  endpoint, and exactly the sort of thing `scope_violation_rate = 0.00` should
  catch but silently would not.

---

## 3. LangGraph topology and interrupts

### Nodes

```
START → supervisor
supervisor → {retriever, analyst, merchandiser, critic, explainer, END}   (routed)
retriever    → supervisor
analyst      → supervisor
merchandiser → supervisor
critic       → {supervisor (reject, rounds<2), human_gate (pass), END (final reject)}
human_gate   → {explainer (approve), supervisor (amend), END (reject)}
explainer    → END
budget_guard → runs as a pre-hook on every node transition
```

The supervisor is a hub, not a chain. Specialists always return to it, which is
what makes the `goal` restatement (§7.8 plan-drift mitigation) a single place
rather than six.

### One gate node, typed reasons — not six interrupt points

§7.7 lists six gates. Implementing six `interrupt_before` points gives you six
resume contracts and six places to get the scope re-validation wrong. Instead:
**one `human_gate` node**, reached by routing, carrying a typed reason.

```
GateReason = Literal[
    "publish",            # a slate is about to be marked approved
    "policy_override",    # long-tail quota or diversity floor would be breached
    "low_confidence",     # calibrated confidence below threshold
    "repeat_failure",     # same error class twice
]
```

That is four of the six. The other two are **not graph interrupts**, and saying so
is more honest than pretending:

- **Export** — the agent has no export scope at all (§13.2, `never`). There is
  nothing for the graph to interrupt. The export gate lives at the API boundary
  (`POST /segments/export`, merchandiser+, explicit confirmation, audited). Calling
  it a graph gate would be theatre.
- **Out-of-allowlist crawl** — crawling is a build-time pipeline
  (`pipelines/07_crawl_corpus_d.py`) against a pinned snapshot. There is no live
  crawler in the request path, so this is a CLI confirmation in the pipeline, not a
  runtime interrupt. It still closes the SSRF path §13.5 names, because the runtime
  has no fetch capability whatsoever.

Being able to say "four of my six gates are graph interrupts, one is an API
boundary, one is a build-time prompt, and here is why each is where it is" is a
better viva answer than six identical-looking interrupts.

### Interrupt mechanics

`interrupt()` inside `human_gate`, resumed via `Command(resume=GateResolution(...))`.
The resume payload is validated against `pending_gate.gate_id` — a resolution for a
stale gate is rejected, not applied. Every gate open/resolve pair is written to the
audit log with the caller who resolved it, which is the §13.1 "every agent run
logged and replayable" requirement made concrete.

### The critic's adversarial separation, mechanically

§7.4 requires the critic not share state with the proposer. In LangGraph, all nodes
read the same state object, so this has to be *constructed*, not assumed. The critic
node receives a **projection**:

```
CriticView = (candidate_slate, claims, evidence[resolved only], policy_corpus_C, granted_scopes)
```

It does **not** see `plan`, the supervisor's reasoning, prior rejections it authored,
or the retriever's working context. That projection is what makes the separation
real rather than nominal, and it is a five-line function that is easy to write at D10
and painful to retrofit.

---

## 4. Tool catalogue

Every tool: Pydantic input model, Pydantic output model, required scope, owning
node, call cap, determinism flag. Registered in a frozen dict; **no dynamic
registration** (§13.4 LLM07). Bound to the API with `strict: true`, which makes
argument-schema validity an API-level guarantee rather than a runtime check — this
is a direct mechanism for the `slate_schema_validity = 1.00` gate.

| Tool | Node | Scope required | Cap | Det. | Notes |
|---|---|---|---|:--:|---|
| `recommend(customer_id \| article_id, k, model)` | Merchandiser | `recs:read` | 4 | ✅ | `model ∈ {popularity, content, collab, hybrid}`; `llm_reranker` deliberately **not** exposed — it is a benchmark arm, not a production path (§6.1) |
| `optimise_slots(candidate_ids, k, constraints)` | Merchandiser | `merch:simulate` | 3 | ✅ | Returns `Slate` + `OptimiserReport` (binding constraints, shadow prices) |
| `project_lift(slate_id, baseline)` | Analyst | `eval:read` | 3 | ✅ | Output field is literally named `projected_incremental_revenue` |
| `eval_report(run_id \| latest, metric)` | Analyst | `eval:read` | 4 | ✅ | Corpus B |
| `clv(cohort_spec)` | Analyst | `segments:read:agg` | 3 | ✅ | **Cohort only, never an individual.** Enforces §13.2 agent `—` on individual records |
| `rfm_segment(cohort_spec)` | Analyst | `segments:read:agg` | 3 | ✅ | Same |
| `graph_traverse(start, relations, depth, filters)` | Retriever | `corpus:a:read` | 6 | ✅ | `depth ≤ 3`, result cap 200 nodes |
| `hybrid_search(query, corpus, k)` | Retriever | `corpus:{b,d}:read` | 6 | ✅ | Corpus D results always arrive `trust="untrusted"` |
| `load_policy()` | Critic, Merchandiser | `corpus:c:read` | 2 | ✅ | Whole corpus, no retrieval (§8.2) |
| `request_human_approval(slate_id, rationale)` | Gate | — | 1 | n/a | Interrupt |

**Scope names are namespaced verbs, not role names.** `recs:read`, not `analyst`.
That is what makes the §13.3 intersection computable:

```
effective = caller_scopes & AGENT_ROLE_SCOPES & task_scopes
```

where `AGENT_ROLE_SCOPES` is a module-level `frozenset` containing **no** scope
ending in `:write`, `:approve`, `:export` or `:override` — asserted by a unit test
that iterates the full scope enum. That test is the mechanical form of the three
`never` rows, and it fails if someone later adds a write scope to the agent role.

`task_scopes` is derived deterministically from the parsed brief at supervisor time
(a brief naming a cohort yields `segments:read:agg`, never `segments:read:individual`)
— the narrowing in §13.3 point 2.

Enforcement is at the tool boundary in `core/rbac.py`, **and** re-asserted post-hoc
by critic criterion 9 against `state.tool_calls`. Belt and braces, because the first
is the control and the second is the evidence.

---

## 5. Taxonomy graph schema (corpus A)

### Storage: Postgres, not a graph database

15k nodes / ~200k edges is small. `WITH RECURSIVE` over an indexed edge table
handles depth-3 traversal in single-digit milliseconds. Adding Neo4j buys a
container, a driver, a query language, and a deploy dependency, for nothing at this
scale. If the graph ever exceeds ~10M edges, revisit — and write that threshold
down, the same way §8.2 writes down the 500-page RAG threshold.

### Node types

| Type | Count (est.) | Key |
|---|---|---|
| `Article` | ~15k | `article_id` |
| `ArticleType` | ~130 | slug |
| `SubCategory` | ~45 | slug |
| `MasterCategory` | ~7 | slug |
| `Colour` | ~50 | slug (H&M native + FPI-predicted, kept distinct) |
| `Season` | 4 | slug (predicted) |
| `Usage` | ~8 | slug (predicted) |
| `Gender` | ~5 | slug |

### Edge types and thresholds

| Edge | From → To | Threshold | Directed | Weight |
|---|---|---|:--:|---|
| `is_a` | Article → ArticleType → SubCategory → MasterCategory | none — structural | ✅ | 1.0 |
| `has_colour` | Article → Colour | native: none. predicted: classifier p ≥ 0.70 | ✅ | p |
| `has_season` | Article → Season | predicted, p ≥ 0.65 | ✅ | p |
| `has_usage` | Article → Usage | predicted, p ≥ 0.65 | ✅ | p |
| `co_purchased_with` | Article ↔ Article | support ≥ 20 baskets **and** lift ≥ 1.5, top-30 per article | ↔ | lift |
| `visually_near` | Article ↔ Article | CLIP cosine ≥ 0.82, k = 20, **mutual-kNN only** | ↔ | cosine |
| `substitutes_for` | Article ↔ Article | `visually_near` ∧ same `ArticleType` ∧ ¬`co_purchased_with` | ↔ | cosine |

Three notes that matter:

- **Predicted edges carry `source="predicted"` and the classifier's confidence.**
  §3 requires "state clearly" that enriched attributes are model-predicted; putting
  it in the edge row rather than a README paragraph means the "why this?" overlay
  can render it and the critic can see it. A path that runs through a predicted edge
  is weaker evidence than one that does not, and the graph should say so.
- **Mutual-kNN on `visually_near`** removes hub articles (plain black tees) that
  appear in everyone's neighbour list and quietly dominate every traversal.
  A one-line filter that materially improves path quality.
- **`substitutes_for` is an addition** and is the merchandising-useful one:
  "looks alike but is not bought alongside" is a substitute; "bought alongside" is a
  complement. A slot optimiser that cannot distinguish them will fill a page with
  eight versions of the same shirt. Cheap to derive from edges you already have.

Thresholds go in `pipelines/manifests/graph_v1.json` with the resulting edge counts,
so a reviewer can see that `lift ≥ 1.5` produced N edges and judge whether it was
tuned to flatter a metric.

---

## 6. Retrieval router decision function

`rag/router.py`. The critical property: **the model classifies the query's shape;
code maps shape to strategy.** The model never picks a strategy. That is what keeps
routing inside "models do routing" as §0.1 means it — routing as extraction, not
routing as decision — and it is what makes `retrieval_routing_accuracy` a
well-defined classification metric rather than a vibe.

```
def route(query: str, ctx: BriefContext) -> RouteDecision:

    # ── Stage 1 · deterministic pre-emption. No model call. ──────────────
    if POLICY_LEXICON.search(query):
        return RouteDecision(shape="policy", strategy="load_full_corpus",
                             corpus="C", decided_by="rule:policy_lexicon",
                             confidence=1.0)
    if ARTICLE_CODE_RE.search(query):        # ^\d{9,10}$ — H&M article ids
        return RouteDecision(shape="taxonomic", strategy="graph_node_lookup",
                             corpus="A", decided_by="rule:article_code",
                             confidence=1.0)
    if RUN_REF_RE.search(query):             # "run 12", "run_id=..."
        return RouteDecision(shape="numeric_historical",
                             strategy="structured_plus_dense", corpus="B",
                             decided_by="rule:run_ref", confidence=1.0)

    # ── Stage 2 · model classifies SHAPE ONLY. Constrained decode. ───────
    shape, confidence = classify_shape(query, ctx)     # 6-way + confidence

    # ── Stage 3 · deterministic table. Frozen dict in code. ──────────────
    if confidence < TAU_ROUTE:               # TAU_ROUTE = 0.60, tuned at D11
        return RouteDecision(shape=shape, strategy="fanout_rrf",
                             corpus="A+B", decided_by="fallback:low_confidence",
                             confidence=confidence)

    if shape == "market_context" and not ctx.allow_external:
        return RouteDecision(shape=shape, strategy="refuse",
                             corpus=None, decided_by="rule:external_default_deny",
                             confidence=confidence)

    return SHAPE_TABLE[shape].with_confidence(confidence)
```

`SHAPE_TABLE` is the §8.1 table, frozen:

| shape | strategy | corpus |
|---|---|---|
| `taxonomic` | `graph_traverse` | A |
| `multi_hop_relational` | `graph_path` | A |
| `visual_semantic` | `dense_clip` | A |
| `numeric_historical` | `structured_plus_dense` | B |
| `policy` | `load_full_corpus` | C |
| `market_context` | `hybrid_rerank_untrusted` | D |

Three deliberate behaviours:

- **Stage 1 exists so the highest-precision routes never depend on a model.**
  "What is the long-tail quota" must reach corpus C every single time. A lexicon
  match is 1.0 accurate; a classifier is not.
- **Low confidence fans out to A+B and RRF-fuses**, rather than guessing. Cost of a
  fan-out is one extra retrieval; cost of a wrong route is a wrong answer with
  confident citations. `max_retrieval_fanouts = 2` in Budget caps the damage.
- **Corpus D is default-deny.** The router will not route to untrusted external
  content unless the brief explicitly asked for market context. The cheapest defence
  against a corpus-D injection is not retrieving corpus D.

`decided_by` is recorded in `state.route_decisions`, so the eval can separate
"the rule fired" from "the classifier was right" — those are different numbers and
reporting them merged would inflate routing accuracy with free wins.

---

## 7. The subsampling rule, as executable criteria

`pipelines/01_subsample.py`. Ordered, deterministic, asserted. §3 says "document the
subsampling rule; it's a methodological choice." This is that document.

```
INPUTS   data/raw/hm/{transactions_train,articles,customers}.csv, images/
OUTPUTS  data/processed/{articles,customers,transactions_train,transactions_test}.parquet
         pipelines/manifests/subsample_v1.json
```

```
R1  T_end   := max(transactions.t_dat)
    T_start := T_end - 84 days                    # 12 weeks, inclusive
    txns    := transactions[T_start ≤ t_dat ≤ T_end]

R2  articles := articles[has_image ∧ image_decodes]     # CLIP + atlas require it

R3  ── fixed-point support filter, max 3 iterations ──
    repeat:
        keep_articles  := {a : count(txns[article=a]) ≥ 20}
        txns           := txns[article ∈ keep_articles]
        keep_customers := {c : count(distinct t_dat in txns[customer=c]) ≥ 3}
        txns           := txns[customer ∈ keep_customers]
    until (keep_articles, keep_customers) unchanged   or   3 iterations

R4  split_date := T_start + 70 days               # 10 weeks train / 2 weeks test
    train := txns[t_dat <  split_date]
    test  := txns[t_dat ≥ split_date]

R5  cold_start_users := customers in test with < 3 purchases in train
                        (retained, NOT filtered — they are the §9 cold-start strata)

R6  freeze: write parquet + manifest {counts, thresholds, T_start, T_end,
            split_date, iterations_to_fixpoint, sha256 of each output}
```

**Assertions that fail the build:**

```
A1  max(train.t_dat) < min(test.t_dat)               # §6. The leak assertion.
A2  every article in test ∈ keep_articles            # no unseen-article leakage
A3  10_000 ≤ |keep_articles| ≤ 20_000                # §3 says ~10–15k
A4  30_000 ≤ |keep_customers| ≤ 80_000
A5  every article has a decodable image
A6  iterations_to_fixpoint ≤ 3
A7  manifest sha256 matches the parquet actually on disk
```

**The subtlety most implementations miss, and why R3 loops.** Filtering articles to
≥20 purchases *removes transactions*, which pushes some customers below 3. Filtering
those customers removes more transactions, which pushes some articles below 20. A
single pass leaves the stated rule violated in the output. The fixed point takes
2–3 iterations and is cheap. If it has not converged in 3, that is a signal the
thresholds are wrong for this window — fail loudly rather than accept iteration 3.

**If A3/A4 fail**, adjust the thresholds, re-record them in the manifest, and note
the change in the README. Never silently widen a bound to make an assertion pass —
the manifest is the audit trail for exactly this.

**Test-set users must appear in train** for the warm-start evaluation; cold-start
users are evaluated as a separate stratum (R5), not dropped. Dropping them would
make the §9 cold-start curve measure nothing.

---

## 8. Atlas generation pipeline

`pipelines/04_build_atlas.py`. **Build-time only. Produces static assets.** Runs
once per catalogue version; the runtime never sees an image decoder.

### The single canonical index

Everything the 3D scene reads is indexed by one integer, `atlas_index`, assigned by
sorting `article_id` ascending. Positions, colours, atlas UVs and metadata all share
it. This is the invariant that makes the scene correct, and it gets its own test:

```
assert len(positions) == len(colours) == len(atlas_entries) == n_articles
assert atlas_entries[i].article_id == sorted_article_ids[i]   ∀ i
```

Get this wrong and every product in the gallery shows the wrong photograph — a bug
that looks like a rendering problem and is actually an indexing problem, and one
that is very hard to spot when 15,000 garments all look plausible.

### Pipeline

```
1  order      sort article_ids → atlas_index (stable, recorded in manifest)
2  decode     load image, EXIF-strip, convert to sRGB
3  crop       centre-crop to square, Lanczos resize to 64×64 (desktop)
                                                    32×32 (mobile variant)
4  colour     extract dominant colour in the SAME pass (k-means k=3, take the
              largest non-background cluster) → colours[i]
5  pack       row-major into 4096×4096 sheets → 64×64 tiles/side = 4096 per sheet
              15k articles → 4 sheets desktop, 1 sheet mobile
6  compress   KTX2 / ETC1S (basisu). ~3 MB per sheet → ~12 MB desktop total
7  emit       atlas_{0..3}.ktx2
              positions.bin        Float32Array[n*3]   from UMAP (step 03)
              colours.bin          Uint8Array[n*3]     from step 4
              atlas_manifest.json  {version, tile_px, sheet_px, tiles_per_sheet,
                                    n, layers, article_ids[]}
8  verify     the index assertions above + every tile is non-blank
```

### Two decisions worth defending

**64px tiles, not 128px.** 128px gives 1024 tiles/sheet → 15 sheets → ~45MB and a
messy sampler story. At 64px the whole catalogue is 4 sheets and ~12MB, and a
64×64 thumbnail on a plane you are flying past is indistinguishable from 128px. The
product detail route loads the real CDN image; the atlas is for the *cloud*, not for
inspection. This is the difference between a scene that loads in 2s and one that
doesn't load on mobile at all.

**`THREE.DataArrayTexture` (WebGL2 `sampler2DArray`), not four bound samplers.**
Four samplers means branching in the fragment shader on a per-instance sheet index,
which is ugly and costs you the single-draw-call claim in practice. A texture array
is one sampler, one draw call, per-instance `layer` attribute. WebGL2-core, so no
extension check. Fallback to the 2D grid (§12.5) if WebGL2 is absent — not to a
four-sampler path, which would be a second renderer to maintain.

### The signal colour falls out of this pipeline

Step 4 computes the dominant colour of all 15,000 garments. Histogram those in
HSV, find the emptiest high-chroma hue bin, and pick the signal colour from it.
§12.2 asks for "a single high-chroma value that appears nowhere in typical garment
photography" — that is a measurable property, and this pipeline already has the
data. Cyan (~185°) is the expected winner in a fashion catalogue; the point is to
**verify it rather than assert it**, and to be able to say in the viva that the
selection colour was chosen by measurement.

---

## 9. Design tokens

`apps/web/styles/tokens.css`. Light on bare `:root`, dark under `.dark`, per §12.4 —
never a colour defined only inside a media query.

### Colour

| Token | Light (paper) | Dark (void) | Use |
|---|---|---|---|
| `--ground` | `#FAF8F5` | `#0B0A09` | page |
| `--surface` | `#F2EEE8` | `#141210` | cards, panels |
| `--surface-raised` | `#FFFFFF` | `#1C1917` | popovers, hover |
| `--hairline` | `#E3DCD2` | `#2A2624` | 1px borders |
| `--text` | `#17140F` | `#F5F1EA` | body |
| `--text-muted` | `#6B635A` | `#9A918A` | metadata |
| `--text-faint` | `#9A9188` | `#635C56` | axis labels, timestamps |
| `--signal` | *measured, §8* | *measured, §8* | selection + active only |
| `--signal-dim` | signal @ 18% | signal @ 22% | focus ring, hover trace |
| `--tail` | `#8A8578` | `#7D786C` | long-tail / coverage viz |
| `--reject` | `#8C6D4F` | `#B08A63` | critic rejections — never red-alarm |
| `--scene-bg` | `#F2EEE8` | `#080807` | R3F clear colour |
| `--scene-line` | `#00000026` | `#FFFFFF1F` | neighbour lines |

Light is authored, not inverted: warm greys with a yellow bias, not the dark
palette's greys flipped. Contrast ratios are measured with a script at D16 and the
actual numbers go in the README (§12.4 requires the numbers, not "we checked").

### Non-colour

```
--space-1..8       4 8 12 16 24 32 48 64        4px base
--radius-sm/md/lg  4px 8px 14px                 no fully-round chrome
--font-ui          "Inter Variable", system-ui   opsz on, tracking -0.011em
--font-mono        "JetBrains Mono Variable"     font-variant-numeric: tabular-nums
--step--1..4       clamp() fluid scale, ratio 1.24
--ease-out         cubic-bezier(.16,1,.3,1)
--dur-fast/base/slow  120ms 220ms 420ms
--flyto-dur        900ms                          the camera transition
```

**Rules enforced in review, not just intended:**

- `--signal` appears in exactly one place per view. A grep for `var(--signal)` that
  returns more than a handful of hits is a bug.
- All spacing and border properties use logical forms (`padding-inline`,
  `border-inline-start`, `margin-block`) from the first component. Not "later".
  A stylelint rule blocks `margin-left` / `padding-right` / `left:` in CI.
- Recharts reads every colour from CSS variables via `getComputedStyle` on theme
  change. Zero hex literals in chart code.
- `color-scheme: light dark` set per theme so scrollbars and the caret follow.
- **Product textures are never tinted.** The instanced material uses the atlas
  sample unmodified; only `--scene-bg`, `--scene-line` and the LOD point colours
  rebind on theme change. Asserted by a visual test: same garment, both themes,
  pixel-identical texture region.

---

## 10. API contract

FastAPI. All responses envelope errors as
`{"error": {"code": <enum>, "message": str, "detail": any, "request_id": str}}`.
Auth via httpOnly JWT cookie. Every route declares its required scope in a decorator
that the RBAC test suite enumerates — a route with no declared scope fails CI.

### Catalogue and scene

```
GET  /catalogue/articles?cursor&limit&filter        → CursorPage[ArticleSummary]
GET  /catalogue/articles/{article_id}               → ArticleDetail
GET  /space/manifest                                → SpaceManifest
     { version, n, positions_url, colours_url,
       atlas: { sheets[], tile_px, sheet_px, tiles_per_sheet },
       variant: "desktop" | "mobile" }
```

`/space/manifest` is immutable per version and CDN-cached with a long max-age;
`positions.bin`, `colours.bin` and the `.ktx2` sheets are static assets, never
served from Python.

### Recommendations

```
GET  /recs/article/{article_id}?model&k             → RecList
GET  /recs/customer/{pseudo_id}?model&k             → RecList     [merchandiser+]
GET  /recs/article/{article_id}/why?k               → WhyThis
     { neighbours: [ { article_id, visual, collaborative,
                       taxonomy_path: [node_id], total } ] }
```

`WhyThis` returns the three signal contributions **separately** — the overlay
renders them, it does not compute them. `taxonomy_path` is the actual traversed
path from corpus A, which is what §8.3 means by "the explanation is the traversed
path, not a post-hoc narrative."

### Evaluation, segments, merchandise

```
GET  /evaluate/runs                                 → [RunManifest]        [analyst+]
GET  /evaluate/runs/{run_id}                        → EvalReport
GET  /evaluate/frontier                             → [{model, ndcg10, coverage, gini}]
GET  /evaluate/coldstart                            → [{bucket, n, ndcg10, ...}]
GET  /evaluate/agent                                → AgentGateTable
GET  /evaluate/calibration                          → ReliabilityCurve

GET  /segments/rfm                                  → RFMAggregates        [analyst+]
GET  /segments/clv                                  → CLVDistribution + holdout fit
POST /segments/export                               → 202 ExportJob   [merchandiser+,
                                                       confirmed, audited]

POST /merchandise/simulate                          → SimulationResult
     { slate, baseline_slate, projected_delta, coverage_cost, optimiser_report }
GET  /merchandise/policy                            → PolicyDocument (corpus C)
```

### Agent — the interesting one

```
POST /agent/runs            {brief}                 → 202 { run_id }        [analyst+]
GET  /agent/runs/{run_id}                           → MerchandisingRun (redacted)
GET  /agent/runs/{run_id}/events                    → text/event-stream
POST /agent/runs/{run_id}/resume  {gate_id, decision, note}
                                                    → 202  [merchandiser+ for publish]
POST /agent/runs/{run_id}/cancel                    → 202
GET  /agent/runs/{run_id}/trace                     → [Span]
GET  /audit?actor&action&from&to                    → CursorPage[AuditEntry]  [admin]
```

**SSE event schema.** Every event `{seq, run_id, ts, type, payload}`; the client
reconnects with `Last-Event-ID` and the server replays from `seq`. Types:

```
run.started · step.started · route.decided · tool.called · tool.returned
evidence.added · claim.added · slate.proposed · critic.rejected
gate.opened · gate.resolved · budget.warning · run.completed · run.failed
```

`critic.rejected` is a first-class streamed event, not something the client
discovers by polling the final state. The rejection panel (§12.6) renders as
rejections happen — which is the demo moment in §1.

**The resume endpoint is the interrupt contract.** It requires the *publish* scope
for a publish gate, re-validated against the caller's current role at resume time,
not against the checkpoint. See §2.

**PII.** `pseudo_id` is a per-deployment HMAC of the H&M customer id, never the raw
id, and never in a URL path that gets logged. `pii_leak_rate = 0.00` is a gate; the
test greps every response body in the golden-set run for raw-id patterns.

---

## 11. Build order

Revised for the golden-set move and the two gaps in §0. 22 slots, ~142h with the
reductions in §13 applied.

| # | Session | Hrs | Why here |
|---|---|---:|---|
| D0 | Architecture + this plan | 1.5 | done |
| D1 | Data ingest, subsample, temporal split (§7) | 5 | everything depends on the frozen parquet |
| **D1.5** | **Corpus C — author the policy** | **3** | **[GAP]** critic + golden set both need it to exist |
| D2 | CLIP embeddings, UMAP, **atlas + colour histogram + signal colour** | 8 | atlas is build-time; signal colour falls out |
| D3 | Baseline, content, collaborative, hybrid | 9 | |
| D4 | Ranking + beyond-accuracy + cold start | 7 | the graded core |
| D5 | RFM, CLV, slots, lift | 6 | slot optimiser is the tool the agent calls |
| D6 | Taxonomy graph (+ attribute classifier, **[GAP]**) | 9 | corpus A |
| D7 | Retrieval router, hybrid search, corpus D crawl + snapshot | 7 | |
| **D8** | **GOLDEN SET — 60 briefs + tool signatures + criteria 3/4/8 thresholds** | **8** | **moved. defines what "correct" means before anything can pass** |
| D9 | Agent state, graph topology, tool implementations | 9 | catalogue now derived from D8, not imagined |
| D10 | Critic, gates, budgets, Postgres checkpointing | 7 | thresholds already exist from D8 |
| D11 | Eval harness + CI gates **+ the 4-vs-3 node measurement (D0-1)** | 6 | reduced — briefs already written |
| D12 | LLM re-ranker arm + benchmark (batched, 500-user sample) | 5 | |
| D13 | Observability — reasoning spans, trace store | 3 | reduced, see §13 |
| D14 | FastAPI service **+ verify host free-tier terms (D0-3)** | 5 | early enough to change host if needed |
| D15 | Auth + RBAC + down-scoping | 4 | reduced, see §13 |
| D16 | Web scaffold, tokens, theme system | 7 | |
| D17 | 3D space — instancing, atlas, LOD, picking | 11 | |
| D18 | flyTo, neighbour lines, "why this?" | 6 | |
| D19 | Merchandise + segments + evaluate + agent console | 9 | |
| D20 | Injection red-team + security review | 5 | |
| D21 | Deploy, polish, recording | 5 | |

**The dependency the move creates:** D8 must come after D5 (the slot optimiser must
exist in signature form to write a brief about it) and after D1.5 (you cannot write
a quota-conflict brief without a quota). It must come before D9. That is exactly
where it now sits.

**The risk the move creates, and its mitigation:** writing briefs against tools that
do not exist yet risks specifying a capability you then cannot build in budget.
Mitigation is that D8's second output is the tool *signatures* — you write the brief
and the signature together, and if a brief needs a tool you cannot cost, you change
the brief at D8 when it costs an hour, not at D14 when it costs a week.

---

## 12. Verification

Per layer, because §7.1 says each layer is tested differently.

**Deterministic core** — `pytest tests/unit tests/property`. 100% of decision paths.
Property tests where the maths has invariants worth stating: the slot optimiser
never returns more than `k` items, never violates a hard constraint it reported as
satisfiable, and is monotone in the revenue coefficient. `lift.py` returns 0 for
identical slates.

**Invariants** — `pytest tests/invariants`. The temporal-split leak assertion (A1),
the atlas index alignment, the agent-role-has-no-write-scope assertion, and the
"every route declares a scope" enumeration. These are one-line tests guarding
architectural claims, and they are the ones that catch a regression six weeks later.

**Orchestration** — `pytest tests/integration`. Graph reaches every node; critic
caps at 2 rounds; a run interrupted at a gate resumes from the checkpoint with the
same state; a resume with a stale `gate_id` is rejected; budget exhaustion produces
a clean `run.failed`, not a hang.

**Agent + RAG** — `python eval/run.py`. Prints the §10.5 table, writes it into the
README, writes failures by name into `eval/failures/<date>/`. Non-zero exit if any
of the five gates fails. This is the CI job.

**End to end, by hand** — `docker compose up`, run the pipelines in order, start the
API, `npm run dev`, then: load `/`, confirm the cloud renders and flyTo works; open
`/agent`, submit the demo brief from §1, watch the SSE trace, confirm at least one
rejection renders in the panel, approve at the gate, confirm the explanation cites
resolvable evidence ids. Toggle the theme and confirm a garment's pixels are
unchanged.

---

## 13. Pushback — what is overengineered, and what I would cut

You asked to hear this now rather than at D16, so I am not softening it.

### The headline: this is not a 140-hour build

My estimate for the architecture as written is **185–200 hours**. The specific
places §14 is optimistic:

- **D2 at 8h.** CLIP extraction is ~1h of compute and 2h of code. The atlas — KTX2
  toolchain, basisu, packing, manifest, the index invariant, a mobile variant — is
  5–6h on its own. D2 is a 12h session.
- **D11 at 9h for harness + 60 briefs + CI.** Hand-writing 60 stratified briefs
  *with labels*, including 8 that must escalate and 6 that must refuse, is 8–10h
  before a line of harness code. This is the strongest argument for your move: it
  was never a sub-task of D11, it was always its own session.
- **D16 at 11h.** Instanced 15k + texture array + LOD + GPU picking + progressive
  streaming + a 2D fallback. GPU picking alone is most of a day when the render
  target and the instance id encoding fight you.
- **D9 at 7h** for critic + gates + budgets + Postgres checkpointing. That is four
  integration surfaces. Realistically 10h.

So the "if I lost 30 hours" question is not hypothetical. **Plan the cuts now.**

### Overengineered as specified

1. **Cross-encoder rerank (§8.4).** It applies only to corpora B (~50 manifests) and
   D (~100 documents). BM25 + dense with RRF over 100 documents is already at
   ceiling; a cross-encoder adds a model, a latency budget and ~2h for no measurable
   gain. **Cut it — and report that you tested it and it did not move context
   precision on a corpus this size.** That is a finding, and it is the same
   intellectual move as §8.2's decision not to build RAG over corpus C. It makes
   your "knowing when not to" argument stronger, not weaker.

2. **Serendipity (§9).** It has no agreed formula, you will spend an hour choosing
   between three, and coverage / Gini / popularity-bias / diversity / novelty is
   already a distinction-grade set. Cut serendipity, keep novelty.

3. **The OTel collector (D12).** The *reasoning span structure* is the valuable
   part and you should keep it. Running Jaeger or Tempo in your deploy is a
   container, a config and an ongoing cost for a demo nobody will open. Write
   structured spans to Postgres and render them in the agent console you are
   building anyway at D19. Saves ~3h and a deploy dependency, and the console is a
   better demo than a Jaeger UI. Keep the OTel *semantic conventions* for the span
   attribute names so the claim "OTel GenAI conventions" stays true.

4. **Full OAuth2 + refresh rotation (D14, 7h).** What §13.2 and §13.3 get graded on
   is the permission matrix and `effective_scopes = ∩`. Nobody in a viva will ask
   about refresh-token rotation. Credential login issuing an httpOnly JWT, plus the
   scope intersection and its unit tests, is 4h. Saves 3h at zero cost to the
   defensible claim.

5. **`lifetimes` for BG/NBD (D5) is a live dependency landmine.** It is effectively
   unmaintained and breaks against current numpy/scipy. This will silently eat 2h at
   D5. Use `pymc-marketing`'s CLV module, or write the BG/NBD likelihood directly —
   it is about 40 lines and you will understand it better in the viva. Decide this
   before D5, not during it.

6. **The Analyst / Merchandiser split.** §7.4's three grounds are strong for
   Retriever (context economy — taxonomy dumps are genuinely large) and overwhelming
   for Critic (adversarial separation). They are **weak** for Analyst vs
   Merchandiser: both are thin wrappers that call a deterministic function with typed
   args, neither holds retrieval, and their combined tool surface is five tools. An
   examiner who probes "why four?" will get a great answer twice and a thin one once.

   **Resolved (D0-1):** build four, measure both ways at D11, merge if the split
   buys nothing. The node bodies must therefore be written so a merge is a
   graph-wiring change, not a rewrite.

7. **`cost_per_brief_usd ≤ 0.05` (§10.5) is not reachable as specified.** A run is
   7–9 model calls; corpus C is ~10k tokens and gets loaded twice (critic and
   merchandiser); with evidence and traces, input is 60–100k tokens per brief. On
   `claude-opus-5` at $5/MTok input that is **$0.30–0.50 per brief, 6–10× the stated
   target**.

   **Resolved (D0-2):** cache corpus C, tier the models, batch the single-shot
   evals, sample the D12 benchmark. Target holds at $0.05 as a designed number.
   The cache-ordering constraint on `goal` is now part of the state design.

### If I lost 30 hours, in this order

| # | Cut | Saves | Damage |
|---|---|---:|---|
| 1 | OAuth2 + refresh rotation → JWT + scope intersection | 3h | none — the graded part is untouched |
| 2 | OTel collector → spans in Postgres, rendered in the console | 3h | none; arguably a better demo |
| 3 | Cross-encoder rerank → RRF only, reported as a tested negative | 2h | negative — it becomes a finding |
| 4 | Serendipity metric | 1.5h | none |
| 5 | FPI→H&M attribute classifier → keep FPI **only** as the cold-start holdout | 5h | moderate. Loses `has_season`/`has_usage` edges and the "derived attribute" honesty story. The cold-start use was always the stronger of §3's two uses |
| 6 | "Why this?" in-scene line annotations → render contributions in the 2D side panel | 3h | low. The panel is more readable than floating labels anyway |
| 7 | Live merchandise simulator → static comparison rendered from precomputed eval artefacts | 4h | low for the grade, moderate for the demo |
| 8 | `/segments` route → fold RFM and CLV charts into `/evaluate` | 3h | low |
| 9 | Golden set 60 → 40: keep **all 20** hard-negative / unanswerable / adversarial, cut standard 24→16 and cold-start 8→4 | 5h | real, and must be stated — wider CIs on the tuning metrics. **Last resort, and only partial** |

≈ 29.5h.

### What I would not cut under any circumstances

The five hard gates. The critic and its nine criteria. The rejection panel. The
temporal split and its leak assertion. The accuracy–coverage frontier plot.
`must_be_grounded`. The scope intersection. The 20 adversarial / unanswerable /
hard-negative briefs.

Those are not features of DHAWQ. They **are** DHAWQ — they are the entire reason
this is a defensible project rather than a good-looking one. If the schedule
collapses, cut breadth: fewer views, fewer metrics, a simpler auth story, the 2D
fallback instead of the 3D scene. Do not cut the spine. A DHAWQ with five gates,
a working critic, a rejection panel and a 2D scatter plot is a stronger submission
than a DHAWQ with a beautiful 3D gallery and an eval harness that was never
finished.
