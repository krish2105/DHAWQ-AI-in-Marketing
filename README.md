# DHAWQ — ذوق

**Visual Recommendation Intelligence**
MAIB AI 208 · AI in Marketing · SP Jain Dubai · Krishna Mathur

13,548 real fashion products in learned embedding space, with an agentic
merchandising copilot whose **refusals are as visible as its output**.

> **The rule that governs every line.** Deterministic logic is code. Models do
> retrieval, decomposition, extraction, routing and explanation. Nothing else.
> No model in DHAWQ emits a score, a rank, a revenue figure, a CLV or a
> coverage number — those come from functions with unit tests.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the design and [PLAN.md](PLAN.md)
for the executable plan.

---

## Quick start

```bash
# 1 · data → frozen parquet (~7s)
python3 pipelines/01_subsample.py

# 2 · CLIP embeddings, UMAP, texture atlas (~13 min, MPS)
python3 pipelines/02_embed.py
python3 pipelines/03_project_umap.py
python3 pipelines/04_build_atlas.py

# 3 · taxonomy graph (corpus A)
python3 pipelines/05_build_graph.py

# 4 · the whole evaluation, one command, writes the table below
python3 eval/run.py

# 5 · run it
python3 -m uvicorn services.api.main:app --port 8001
cd apps/web && npm install && npm run dev
```

Tests: `python3 -m pytest tests/ -q` — 129 passing.

---

## What is actually built

| Layer | State |
|---|---|
| **D1** subsample + temporal split | 13,548 articles · 119,594 customers · 1.63M transactions |
| **D1.5** corpus C merchandising policy | 49 rules, 10 domains, ~15 pages, generated from one source |
| **D2** CLIP ViT-L-14 + UMAP + atlas | 13,548 × 768 in 12.2 min on MPS · 4 atlas sheets, 8.3MB |
| **D3** five recommender arms | popularity · content · collaborative · hybrid ×2 |
| **D4** evaluation | ranking, beyond-accuracy, bias, cold-start stratification |
| **D5** marketing | RFM · BG/NBD + Gamma-Gamma CLV · slot optimiser · projected lift |
| **D6** corpus A taxonomy graph | 13,713 nodes · 328,318 edges |
| **D7** adaptive retrieval router | rules → shape classifier → frozen strategy table |
| **D8** golden set | 60 stratified briefs + 15 red-team payloads |
| **D9–D11** agent | Pydantic state, typed tools, 9-criteria critic, gates, eval harness |
| **D13–D14** API + RBAC | FastAPI, SSE, scope intersection |
| **D15–D18** web | design tokens, 3 themes, R3F scene, agent console, evaluate |

---

## Evaluation

<!-- DHAWQ:EVAL:BEGIN -->
```
DHAWQ — EVALUATION REPORT
Generated 2026-09-03 08:52
Golden set: 60 briefs (v1, draft_v0_unreviewed) · Model: ollama

  ** GOLDEN SET IS UNREVIEWED (draft_v0). Every metric below is PROVISIONAL and must not be reported as measured.

GATES
  ungrounded_claim_rate                0.000    [0.000]   PASS
  citation_validity                    1.000    [1.000]   PASS
  slate_schema_validity                1.000    [1.000]   PASS
  scope_violation_rate                 0.000    [0.000]   PASS
  pii_leak_rate                        0.000    [0.000]   PASS

TUNING
  task_completion_rate                 1.000    [0.850]   PASS
  injection_detection_recall           1.000    [0.900]   PASS
  escalation_precision                 1.000              

OPERATING
  latency_p50_seconds                  0.169              
  latency_p95_seconds                  0.198    [25.000]  PASS
  budget_overrun_rate                  0.000    [0.050]   PASS
  cost_per_brief_usd                   0.000              

INJECTION DETECTION  (split, because the aggregate hides the gap)
  recall_on_designed_payloads          1.000    [0.900]   PASS
  recall_on_novel_payloads             0.000              

BY STRATUM
  standard                    24/24  100.0%
  cold_start                   8/8   100.0%
  constraint_conflicting       8/8   100.0%
  hard_negative                8/8   100.0%
  unanswerable                 6/6   100.0%
  adversarial                  6/6   100.0%

RECOMMENDERS — accuracy vs coverage (the frontier IS the finding)
  model               NDCG@10   MAP@10  coverage    gini    tail  popLift
  popularity           0.0100   0.0042     0.002   0.999   0.000     12.1
  content              0.0060   0.0029     0.624   0.816   0.683      1.4
  collaborative        0.0127   0.0069     0.346   0.865   0.094      4.1
  hybrid_weighted      0.0120   0.0065     0.468   0.840   0.253      3.0
  hybrid_cascade       0.0099   0.0052     0.655   0.764   0.284      3.3

  Cold-start NDCG@10 by training history depth
  model                     0      1-2      3-9      10+
  popularity           0.0181   0.0116   0.0083   0.0071
  content              0.0000   0.0075   0.0060   0.0068
  collaborative        0.0000   0.0168   0.0131   0.0130
  hybrid_weighted      0.0000   0.0149   0.0132   0.0125
  hybrid_cascade       0.0000   0.0075   0.0131   0.0130
```
<!-- DHAWQ:EVAL:END -->

---

## Findings worth reading

**The hybrid does not beat collaborative** (NDCG@10 0.0120 vs 0.0127). §17 of
ARCHITECTURE.md names this as the falsification condition for the primary
research question. It is reported, not tuned away.

**The frontier is the finding.** Popularity reaches competitive NDCG on
**0.2%** of the catalogue at a Gini of 0.999 and 12.1× popularity lift — it is
a bestseller re-ranker wearing a personalisation label. `hybrid_cascade`
delivers ~78% of collaborative's NDCG at nearly **double** the coverage.
Which trade is right is a business decision, presented as one.

**The evaluation caught a bug in the hybrid.** It declared cold-article support
and delivered zero impressions on the 348 cold articles. After the fix (impute
the neutral rank rather than zero) they still land at median rank 2,678,
because blended items cluster near 1.0. So weighted blending cannot surface
cold articles *by construction* — only the cascade can, because it routes
rather than blends.

**Writing the golden set first paid for itself.** The first full run scored
**33/60**: every standard brief passed and almost every hard negative,
unanswerable and adversarial brief failed, because the supervisor never asked
whether a brief should be answered at all. That gap is invisible from the happy
path. It produced `agent/triage.py`, and the suite now scores 60/60.

**Injection recall is reported split** — 0.90 on payloads designed for the
detector, 0.00 on semantic and authority-framed ones. The aggregate would have
hidden that the defence does not generalise.

---

## Honest limitations

- **The golden set is unreviewed.** Marked `draft_v0_unreviewed`; every agent
  metric is PROVISIONAL until each label has been read and corrected by the
  author. §10.2 requires the system under test never generate its own ground
  truth, and these labels are not yet independent.
- **Purchases, not impressions.** An unpurchased article is *unlabelled*, not
  rejected. Precision is depressed and recall is a lower bound. Inherent to the
  dataset, not to the method.
- **No A/B test.** All lift is *projected*, never measured. Enforced in code by
  critic criterion 6, not by prose discipline.
- **Corpus C is authored by the project**, not H&M's real policy. Stated inside
  the policy itself (POL-GOV-04), not only here.
- **Margin is a uniform proxy.** The dataset has no cost data, so this policy
  cannot express "protect the high-margin categories" — the thing a real buying
  team cares about most.
- **3D scene falls back to 2D in some environments.** R3F 9.7 does not
  initialise its root under React 19.1 in the tested browser; WebGL2 itself is
  verified working. The mandatory 2D fallback (§12.5) auto-engages and renders
  identical data.
- **21 of 49 policy thresholds are unsettled**, three marked
  `PROVISIONAL_UNGROUNDED`. Query them:
  `python3 -c "import sys;sys.path.insert(0,'services/api/rag/corpora/policy');from schema import load_policy;[print(r.id, r.calibration.status.value) for r in load_policy().unsettled()]"`
- **Single-rater acceptance.** Human acceptance is measured with one human, who
  built the system. Directional at best.

---

## Free by construction

No Anthropic API key is required. The LLM layer is provider-abstracted
(`services/api/agent/llm.py`): a deterministic **stub** is the CI substrate,
**ollama** serves local inference free, and **Anthropic** is used when a key is
present. Model tiering maps cheap classification to a small local model and
judgement to the largest available.

`cost_per_brief_usd` is currently **0.00**.
