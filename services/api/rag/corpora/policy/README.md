# Corpus C — Merchandising Policy

Authored at **D1.5**. Identified as a `[GAP]` in `PLAN.md` §0: the critic cannot
evaluate criteria 3, 4 and 5 without a policy that states a quota, a floor and a
seasonality rule, and the golden set cannot contain quota-conflict briefs
without a quota to conflict with.

## Files

| File | Role |
|---|---|
| `policy.yaml` | **Source of truth.** Every rule, threshold and rationale. |
| `schema.py` | Pydantic shape + whole-policy integrity checks. `load_policy()` lives here. |
| `render.py` | Generates `POLICY.md` and `manifest.json`. `--check` verifies sync. |
| `POLICY.md` | **Generated.** The document that loads into the critic's context. |
| `manifest.json` | **Generated.** Version, hashes, counts, size against the §8.2 threshold. |

`POLICY.md` and `manifest.json` are generated. Never edit them by hand.

## Why the policy is generated, not written

There are two readers with different needs. The critic model reads prose. The
slot optimiser and criteria 3/4/5/9 read numbers. Maintaining those as two
hand-written artefacts means that sooner or later the engine enforces 0.20 while
the document says 0.25 — and every rejection the critic issues is then citing a
rule that does not describe what actually happened.

Generating the document from the parameters removes that by construction rather
than by discipline. `render.py --check` runs in CI.

## Why it is loaded, not retrieved

ARCHITECTURE.md §8.2. At ~15 pages and ~11k tokens it fits in context with room
to spare, so chunking it would add retrieval misses and an extra failure mode
and buy nothing. A critic that reads the whole policy every time cannot miss a
rule because chunk 7 did not rank.

This is enforced by omission: `schema.py` exposes `load_policy()` and no
`load_section()` or `search_policy()`. There is no partial-load API to reach
for. `POL-GOV-03` records the threshold at which that decision is revisited
(200k tokens / 500 pages), and `manifest.json` records the current size, so the
decision is monitored rather than assumed.

## Rule ids

`POL-<DOMAIN>-<NN>`. Ten domains, in precedence order:

`GOV` → `ESC` → `SEG` → `AVL` → `LT` → `DIV` → `PRC` → `BRD` → `SLT` → `CLM` → objective

The optimiser objective is last. It never overrides a constraint.

Cross-references between rules are validated at load: a rule citing a rule id
that does not exist fails at import. This is the policy-layer version of
citation validity, and the whole system rests on citations resolving.

## Severity

- `hard` — slate rejected, dropped, never downgraded
- `escalate` — human gate; the agent may not resolve it
- `soft` — proceeds with a recorded, rendered warning
- `advisory` — not machine-checked; never grounds for rejection

`advisory` exists so the boundary of what is actually enforced is explicit. A
rule with no checker that looks enforced is worse than no rule.

## Calibration

Every rule carries a `calibration` block naming its status, when it is revisited
and what it depends on. **21 of 49 thresholds are unsettled**, and three are
marked `PROVISIONAL_UNGROUNDED` — invented, with no empirical basis yet.

That is not a defect at D1.5; the numbers that will ground them do not exist
until D2 (embeddings) and D8 (golden set). It is recorded in the policy so that
"which numbers did you make up?" is a query, not a question for the author:

```bash
python3 -c "
import sys; sys.path.insert(0,'services/api/rag/corpora/policy')
from schema import load_policy
for r in load_policy().unsettled():
    print(f'{r.id:14} {r.calibration.status.value:24} revisit at {r.calibration.revisit_at}')
"
```

## Regenerate

```bash
python3 services/api/rag/corpora/policy/render.py
```
