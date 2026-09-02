<!-- GENERATED FROM policy.yaml BY render.py — DO NOT EDIT BY HAND. -->

# DHAWQ — Merchandising Policy

**Corpus C** · version `1.0.0` · effective 2026-09-03 · authored by Krishna Mathur

---

## Status of this document

This policy is authored by the project author as a plausible merchandising
rule set for a fashion retailer. It is NOT H&M's actual buying or
merchandising policy, and no part of it should be represented as such.
It exists so that the constraint layer of DHAWQ has something real and
specific to enforce, and so that policy conflicts are reproducible.
ARCHITECTURE.md §3 (Honest limitations) requires this to be stated; it is
stated here, inside the policy itself, rather than only in a README.

This document is loaded **whole** into context. It is not chunked, embedded
or retrieved (ARCHITECTURE.md §8.2). A critic that reads the entire policy
every time cannot miss a rule because a chunk failed to rank.

## How to read a rule

Every rule has an id of the form `POL-<DOMAIN>-<NN>`. **Cite the id.** A
rejection that does not name a rule id is not actionable, and a rule id
that does not appear in this document does not exist.

Each rule carries a severity:

- **`hard`** — The slate is rejected. It is dropped, never silently downgraded into the
output (ARCHITECTURE.md §7.6). A rejection record is persisted with the
rule id and rendered in the agent console.

- **`escalate`** — The slate is not rejected and not approved. The run halts at a human gate
with the rule id, the observed value and the required value. The agent
never resolves an escalation itself.

- **`soft`** — The slate proceeds. A warning carrying the rule id is attached to the run
record and rendered alongside the slate. Soft breaches are counted and
reported; they are not silent.

- **`advisory`** — Not machine-checked at MVP. Recorded here so that the boundary of what is
enforced is explicit rather than implied. An advisory rule must never be
cited by the critic as grounds for rejection.

## Precedence

Where two rules cannot both be satisfied, the earlier domain wins. The
optimiser objective is last: it never overrides a constraint.

`GOV` → `ESC` → `SEG` → `AVL` → `LT` → `DIV` → `PRC` → `BRD` → `SLT` → `CLM` → `objective`

## Definitions

Every term the rules depend on. A rule resting on an undefined term is not
machine-checkable.

### reference window

The 12-week (84-day) transaction window produced by the subsampling
pipeline, bounded by T_start and T_end as recorded in
pipelines/manifests/subsample_v1.json. Every recency and support figure
in this policy is computed within this window and nowhere else.

*depends_on_artifact:* pipelines/manifests/subsample_v1.json

### head / long tail

Rank all articles in the reference window by units sold, descending.
The HEAD is the smallest set of articles accounting for the top 20% of
article COUNT (not the top 20% of units). The LONG TAIL is every article
not in the head. The split is by article count so the quota does not
move when total demand moves.

*params:*

```json
{
  "head_share_of_catalogue": 0.2
}
```

*note:* Computed once per catalogue version and frozen in the catalogue manifest.
It is not recomputed per request — a quota that drifts between two runs
on the same catalogue is not a quota.

### intra-list diversity (ILD)

1 minus the mean pairwise cosine similarity of the CLIP image embeddings
of every article in the slate. Range 0 to 2 in principle; in practice
0.05 to 0.60 for fashion catalogues. Higher is more diverse.

*formula:* ILD(S) = 1 - (2 / (|S| * (|S| - 1))) * sum_{i<j} cos(e_i, e_j)

*requires:* CLIP ViT-L-14 embeddings, laion2b_s32b_b82k, as produced at D2

### active season

Derived from the campaign date by calendar month. The mapping below is
Northern-hemisphere and matches the origin of the H&M transaction data.
DHAWQ is positioned for the UAE market, where the practical retail
calendar is closer to a long hot season and a short mild one; that
divergence is a stated limitation, not an oversight, and the mapping is
a single parameter here so it can be replaced without touching a rule.

*mapping:*

```json
{
  "winter": [
    12,
    1,
    2
  ],
  "spring": [
    3,
    4,
    5
  ],
  "summer": [
    6,
    7,
    8
  ],
  "autumn": [
    9,
    10,
    11
  ]
}
```

*adjacency:*

```json
{
  "winter": [
    "autumn",
    "spring"
  ],
  "spring": [
    "winter",
    "summer"
  ],
  "summer": [
    "spring",
    "autumn"
  ],
  "autumn": [
    "summer",
    "winter"
  ]
}
```

### margin proxy

The H&M dataset contains a normalised `price` field and NO cost data, so
true margin is unobservable. Wherever this policy refers to margin it
means the proxy: price multiplied by a fixed assumed gross margin rate,
uniform across the catalogue. This makes margin monotone in price and
nothing more. Any statement derived from it must be labelled as a proxy.

*params:*

```json
{
  "assumed_gross_margin_rate": 0.55
}
```

*honesty:* A uniform margin rate means this policy cannot express "protect the
high-margin categories", which a real buying team would care about most.
That is a genuine limitation of the data, and it is recorded rather than
hidden behind a plausible-looking number.

### evidence coverage

The proportion of claims in an explanation for which every asserted fact
resolves to at least one evidence_id present in the run state, where that
evidence was produced by a tool call recorded in the same run.

*formula:* coverage = |claims fully resolved| / |claims|

### cohort

A set of customers selected by a stored, re-executable specification
(RFM band, CLV decile, purchase-count stratum, recency bucket, or a
conjunction of these). A cohort is never a list of customer ids pasted
into a brief.

---

## Rules

### LT — Long-tail exposure

The commercial argument for personalisation is that it lifts revenue per
session. The commercial cost is that it concentrates impressions on
articles that already sell. A merchandiser who allows that concentration
to run unchecked ends up with dead inventory and a catalogue that is
wide on paper and narrow in practice. These rules put a floor under
exposure. They are the rules most likely to conflict with a revenue
brief, and that conflict is intentional — it is the decision a human
should be making, so the system surfaces it rather than resolving it.

#### POL-LT-01 — Minimum long-tail share of a slate

**Severity:** `hard` · **Scope:** `slate` · **Critic criterion:** 3

At least 20% of the slots in a slate must be filled by long-tail
articles, rounded up to the next whole slot. For a 12-slot page that
is 3 slots; for a 10-slot page, 2 slots.

*Evaluated in code by* `long_tail_share_at_least`

```json
{
  "min_share": 0.2,
  "rounding": "ceil",
  "applies_when_k_at_least": 5
}
```

**Why:** 20% is chosen to be materially above the level a revenue-maximising
optimiser reaches unaided (typically under 5% on a catalogue with
strong collaborative signal) while remaining achievable without
destroying the objective. It is a constraint with teeth, which is the
only kind worth having.

> **Calibration — provisional.** Revisit at **D8** · depends on golden set constraint-conflicting stratum, D3 hybrid ranker output.
> The 8 constraint-conflicting briefs define the conflict this number
> is supposed to create. If a brief demanding maximum revenue at k=12
> can satisfy 20% without escalating, the quota is too low to be
> interesting and the stratum tests nothing.

#### POL-LT-02 — Quota waiver on very short slates

**Severity:** `hard` · **Scope:** `slate` · **Critic criterion:** 3

Slates of fewer than 5 slots are exempt from POL-LT-01. A quota
cannot be meaningfully allocated across 4 slots without dominating
the page.

*Evaluated in code by* `quota_waived_below_k`

```json
{
  "k_threshold": 5
}
```

**Why:** Without this, a 4-slot module would owe a full slot to the tail —
25% — which is a stricter quota than the 12-slot page it sits on.
Waivers must be written down; an unwritten waiver is a bug.

#### POL-LT-03 — New articles are not long-tail

**Severity:** `hard` · **Scope:** `article` · **Critic criterion:** 3

An article with fewer than 28 days of sales history inside the
reference window is classified as NEW, not long-tail, and does not
count toward the POL-LT-01 quota.

*Evaluated in code by* `exclude_new_from_tail_quota`

```json
{
  "min_history_days": 28
}
```

**Why:** A new arrival has low sales because it is new, not because it is
unwanted. Counting it as tail lets the quota be satisfied entirely
with new-season stock the buyer was always going to push, which
defeats the purpose of the rule.

> **Calibration — provisional.** Revisit at **D1** · depends on reference window length after subsampling.
> 28 days is a third of the 84-day window. If the window changes,
> this must change with it or it silently becomes a different rule.

#### POL-LT-04 — Tail slots must still be sellable

**Severity:** `hard` · **Scope:** `slate` · **Critic criterion:** 3

Articles used to satisfy the POL-LT-01 quota must independently
satisfy every rule in the AVL domain. The quota may not be met with
articles that are out of season or no longer selling.

*Evaluated in code by* `tail_slots_satisfy_availability`

```json
{}
```

**Why:** Otherwise the cheapest way to satisfy a long-tail quota is to dump
dead stock into the page, which is worse for the customer and worse
for the retailer than not having the quota at all. This rule is what
stops POL-LT-01 being gamed.

#### POL-LT-05 — Campaign-level impression concentration

**Severity:** `soft` · **Scope:** `campaign` · **Critic criterion:** —

Across a campaign (a set of slates published together), no more than
60% of total impressions may fall on the top 10% of articles by
projected impression count.

*Evaluated in code by* `campaign_concentration_at_most`

```json
{
  "max_share": 0.6,
  "top_articles_share": 0.1
}
```

**Why:** Slate-level quotas do not prevent campaign-level concentration —
twelve compliant slates can still show the same three tail articles.
This is the campaign-scope version of the same concern.

> **Calibration — provisional.** Revisit at **D8** · depends on whether any golden brief is campaign-scoped.
> SCOPE WARNING for the critic: this rule cannot be evaluated against
> a single slate. If the run produced one slate, this rule is not
> applicable and must not be cited as a breach. Marked soft partly
> for this reason.

### DIV — Intra-list diversity

Ten near-identical black t-shirts is a technically excellent and
commercially useless page. Diversity rules exist because ranking metrics
reward relevance and are blind to redundancy — the model is not doing
anything wrong when it produces a monotonous page, it is doing exactly
what it was optimised for. The constraint has to come from outside the
objective.

#### POL-DIV-01 — Intra-list diversity floor

**Severity:** `hard` · **Scope:** `slate` · **Critic criterion:** 4

For slates of 6 or more slots, intra-list diversity must be at least
0.35 as defined in the definitions section.

*Evaluated in code by* `intra_list_diversity_at_least`

```json
{
  "min_ild": 0.35,
  "applies_when_k_at_least": 6
}
```

**Why:** Below roughly 0.30 a fashion slate reads as a single product in
several colours. 0.35 sits above that with margin.

> **Calibration — ⚠︎ UNGROUNDED.** Revisit at **D8** · depends on D2 CLIP embeddings, observed ILD distribution, golden set.
> This is the least defensible number in the policy today. It is a
> plausible value for CLIP cosine distributions in general, not a
> measured value for THIS catalogue with THESE embeddings. It cannot
> be calibrated before D2 produces embeddings, and should not be
> fixed before D8 confirms which briefs are supposed to fail it.
> Until then, treat any DIV-01 rejection as suspect.

#### POL-DIV-02 — Maximum articles per product type

**Severity:** `hard` · **Scope:** `slate` · **Critic criterion:** 4

No more than ceil(k / 4) articles in a slate may share the same
product type, with a floor of 2. For a 12-slot page that is 3.

*Evaluated in code by* `max_per_attribute`

```json
{
  "attribute": "product_type_name",
  "formula": "ceil(k / 4)",
  "floor": 2
}
```

**Why:** A structural diversity rule that does not depend on embeddings, and
therefore does not depend on POL-DIV-01's uncalibrated threshold.
If DIV-01 turns out to be wrong, this rule still holds the page
together.

> **Calibration — provisional.** Revisit at **D8** · depends on product type cardinality after subsampling.
> Divisor 4 is a judgement; verify against real slates at D8.

#### POL-DIV-03 — Maximum articles per colour group

**Severity:** `hard` · **Scope:** `slate` · **Critic criterion:** 4

No more than ceil(k / 3) articles in a slate may share the same
colour group, with a floor of 2. For a 12-slot page that is 4.

*Evaluated in code by* `max_per_attribute`

```json
{
  "attribute": "colour_group_name",
  "formula": "ceil(k / 3)",
  "floor": 2
}
```

**Why:** Looser than product type because a coherent colour story is a
legitimate merchandising choice in a way that six identical dresses
is not. Uses H&M's native colour field, not the predicted fine
colour, so it does not inherit classifier error.

> **Calibration — provisional.** Revisit at **D8** · depends on colour group cardinality after subsampling.
> Divisor 3 is a judgement.

#### POL-DIV-04 — No adjacent near-duplicates

**Severity:** `hard` · **Scope:** `slate` · **Critic criterion:** 4

Two articles occupying adjacent slots must not have CLIP cosine
similarity of 0.93 or above.

*Evaluated in code by* `no_adjacent_pair_above_similarity`

```json
{
  "max_cosine": 0.93
}
```

**Why:** A slate can pass an aggregate diversity floor and still place two
near-identical garments side by side, which is the version a customer
actually notices. Adjacency is a positional property that an
aggregate metric cannot see.

> **Calibration — ⚠︎ UNGROUNDED.** Revisit at **D2** · depends on D2 CLIP embeddings, observed near-duplicate cosine distribution.
> 0.93 is a guess at where "same garment, different colourway" sits
> in this embedding space. Measure it at D2 by sampling known
> colourway pairs; do not carry this number forward unmeasured.

#### POL-DIV-05 — Minimum distinct subcategories

**Severity:** `hard` · **Scope:** `slate` · **Critic criterion:** 4

Slates of 8 or more slots must contain articles from at least 2
distinct subcategories.

*Evaluated in code by* `min_distinct_attribute_values`

```json
{
  "attribute": "sub_category",
  "min_distinct": 2,
  "applies_when_k_at_least": 8
}
```

**Why:** A deliberately weak floor. Its job is to catch the degenerate case
where an entire page is one subcategory, not to force breadth that
a focused campaign brief legitimately does not want.

#### POL-DIV-06 — Gender coherence for gendered cohorts

**Severity:** `soft` · **Scope:** `slate` · **Critic criterion:** 4

Where a cohort specification names a target gender, at least 80% of
the slate must be articles indexed to that gender or to unisex.

*Evaluated in code by* `attribute_share_at_least`

```json
{
  "attribute": "index_group_name",
  "min_share": 0.8,
  "allow_values_from": "cohort_spec.target_gender + ['unisex']"
}
```

**Why:** Soft rather than hard because cross-gender recommendations are
sometimes correct — gifting, households, and genuinely unisex
product. An 80% floor catches incoherence without forbidding it.

> **Calibration — provisional.** Revisit at **D8** · depends on whether any golden brief targets a gendered cohort.
> If no brief exercises this, it is untested and should be said so.

### AVL — Availability and seasonality

The single most embarrassing failure a recommender can produce is a
beautiful page of things the customer cannot buy. DHAWQ has no stock
feed, so availability is inferred from recent sales — an honest proxy
with a real failure mode, stated here rather than papered over. The
seasonality rules carry an additional burden: they act on PREDICTED
attributes, and a rule that enforces a prediction as though it were a
fact is a rule that manufactures confident errors.

#### POL-AVL-01 — Recent-sale availability proxy

**Severity:** `hard` · **Scope:** `article` · **Critic criterion:** 5

An article may appear in a slate only if it recorded at least one
purchase within the trailing 28 days of the reference window.

*Evaluated in code by* `purchased_within_trailing_days`

```json
{
  "trailing_days": 28,
  "min_purchases": 1
}
```

**Why:** The dataset has no stock table. Recent sale is the strongest
available evidence that an article is still in assortment.

**Known limitation:** This proxy has two known errors. It will admit an article that sold
three weeks ago and has since sold out, and it will exclude a
genuinely available article that happens not to have sold recently —
which biases against exactly the long-tail articles POL-LT-01 is
trying to protect. That tension between AVL-01 and LT-01 is real,
it is not resolved here, and POL-LT-04 makes AVL win. A production
system would replace this rule with a stock feed on day one.

> **Calibration — provisional.** Revisit at **D1** · depends on reference window, article sales sparsity after subsampling.
> At 28 days this may exclude a large share of the tail. Measure the
> exclusion rate at D1. If it removes more than about a third of tail
> articles, the proxy is fighting the quota and one of them must move.

#### POL-AVL-02 — Discontinued articles excluded

**Severity:** `hard` · **Scope:** `article` · **Critic criterion:** 5

An article flagged discontinued in the catalogue manifest may never
appear in a slate, regardless of any other rule.

*Evaluated in code by* `not_flagged`

```json
{
  "flag": "discontinued"
}
```

**Why:** An explicit exclusion list that does not depend on a proxy. Empty at
MVP, but the mechanism exists so that a known-bad article can be
removed without editing code.

#### POL-AVL-03 — Season compatibility

**Severity:** `hard` · **Scope:** `article` · **Critic criterion:** 5

An article's predicted season must be the active season for the
campaign date, an adjacent season within the POL-AVL-04 allowance,
or all-season. Articles whose season is unknown are permitted.

*Evaluated in code by* `season_compatible`

```json
{
  "allow_all_season": true,
  "allow_unknown": true
}
```

**Why:** Promoting winter coats in July is the canonical merchandising error
and the clearest case for criterion 5 existing at all.

> **Calibration — provisional.** Revisit at **D6** · depends on FPI attribute classifier, season label distribution.
> Cannot be enforced at all until the D6 classifier exists. Before
> D6 every article is season_unknown and this rule admits everything.

#### POL-AVL-04 — Shoulder-season tolerance

**Severity:** `hard` · **Scope:** `slate` · **Critic criterion:** 5

Up to 25% of a slate may be filled with articles from a season
adjacent to the active season, rounded down.

*Evaluated in code by* `adjacent_season_share_at_most`

```json
{
  "max_share": 0.25,
  "rounding": "floor"
}
```

**Why:** Real retail transitions. Selling nothing but the current season in
late August is as wrong as selling coats in July, in the other
direction.

> **Calibration — provisional.** Revisit at **D8** · depends on whether a golden brief sits on a season boundary.
> Worth one deliberate boundary-dated brief at D8.

#### POL-AVL-05 — Predicted attributes may not be enforced below confidence

**Severity:** `hard` · **Scope:** `article` · **Critic criterion:** 5

A season-based exclusion under POL-AVL-03 may only be applied when
the attribute classifier's confidence for that article is at least
0.65. Below that threshold the article is treated as season-unknown
and is not excluded.

*Evaluated in code by* `predicted_attribute_confidence_gate`

```json
{
  "attribute": "season",
  "min_confidence": 0.65,
  "below_threshold_behaviour": "treat_as_unknown"
}
```

**Why:** ARCHITECTURE.md §3 requires that enriched attributes be treated as
model-predicted rather than ground truth. This rule is what makes
that requirement operational instead of decorative. Without it the
critic would reject real slates on the strength of a coin-flip
classifier output, and the rejection would look authoritative
because it cites a policy id.

**Known limitation:** The failure direction is deliberate. A low-confidence prediction
causes the rule to ADMIT rather than exclude, so classifier error
produces a slightly stale page rather than a confidently wrong
rejection. Admitting a bad article is recoverable; rejecting a good
slate with a citation is not.

> **Calibration — provisional.** Revisit at **D6** · depends on classifier reliability curve, per-class calibration.
> 0.65 must be replaced with a value read off the classifier's own
> reliability curve at D6 — the confidence at which observed accuracy
> crosses an acceptable bar. Do not keep this number by default.

#### POL-AVL-06 — Markdown share in a full-price campaign

**Severity:** `soft` · **Scope:** `slate` · **Critic criterion:** 5

Where a campaign is not designated a sale campaign, at most 30% of
the slate may be articles currently in markdown.

*Evaluated in code by* `markdown_share_at_most`

```json
{
  "max_share": 0.3,
  "applies_when": "campaign.type != 'sale'"
}
```

**Why:** A page dominated by markdown trains customers to wait for markdown.

**Known limitation:** Markdown status is not directly observable in the dataset. It is
inferred from a fall in an article's realised price relative to its
trailing median. That inference is weak, which is why this rule is
soft rather than hard.

> **Calibration — provisional.** Revisit at **D1** · depends on price series behaviour in the reference window.
> Verify a markdown signal exists at all before relying on it.

### SLT — Slate structure

Structural rules about the shape of a page. They are cheap to check,
almost never contentious, and catch a class of bug that is otherwise
only visible to a human looking at the rendered page.

#### POL-SLT-01 — Slate size bounds

**Severity:** `hard` · **Scope:** `slate` · **Critic criterion:** —

A slate must contain between 4 and 24 slots inclusive.

*Evaluated in code by* `k_within_bounds`

```json
{
  "k_min": 4,
  "k_max": 24
}
```

**Why:** Below 4 there is no allocation problem to solve. Above 24 the page
is a grid, not a merchandised slate, and the optimiser's constraints
stop being the interesting part.

#### POL-SLT-02 — No duplicate articles

**Severity:** `hard` · **Scope:** `slate` · **Critic criterion:** —

An article may occupy at most one slot in a slate.

*Evaluated in code by* `no_duplicate_article_ids`

```json
{}
```

**Why:** Trivially true and worth asserting, because an optimiser bug produces exactly this.

#### POL-SLT-03 — Never pad an under-filled slate

**Severity:** `hard` · **Scope:** `slate` · **Critic criterion:** —

If the optimiser cannot fill k slots without breaching a hard rule,
it must return fewer slots and name the binding constraint. It must
never pad the remaining slots with articles that breach a rule, and
it must never silently reduce k without reporting why.

*Evaluated in code by* `underfill_is_reported`

```json
{}
```

**Why:** This is the rule that keeps the constraint layer honest. A quietly
padded slate looks identical to a compliant one, which means the
quota was never enforced at all — it was just reported as enforced.

#### POL-SLT-04 — Hero slot recency

**Severity:** `hard` · **Scope:** `slate` · **Critic criterion:** 5

The article in slot 1 must have recorded at least one purchase within
the trailing 14 days of the reference window — a stricter bar than
POL-AVL-01.

*Evaluated in code by* `slot_purchased_within_trailing_days`

```json
{
  "slot_index": 1,
  "trailing_days": 14,
  "min_purchases": 1
}
```

**Why:** The hero slot carries disproportionate impressions. If exactly one
article on the page must be certainly available, it is that one.

> **Calibration — provisional.** Revisit at **D1** · depends on sales sparsity.
> If 14 days is too sparse to admit tail articles, the hero slot becomes head-only by construction — check this.

#### POL-SLT-05 — Deterministic tie-breaking

**Severity:** `hard` · **Scope:** `slate` · **Critic criterion:** —

Where two candidate slates score equally on the objective, the
optimiser must prefer, in order: higher long-tail share; then lower
mean price; then lower article id at the first differing slot.

*Evaluated in code by* `tie_break_order_respected`

```json
{
  "order": [
    "long_tail_share_desc",
    "mean_price_asc",
    "article_id_asc"
  ]
}
```

**Why:** Unbounded non-determinism is measured as a defect (§10.4 stability).
A fully specified tie-break makes the deterministic core actually
deterministic, which is the difference between a stability metric
that means something and one that measures tie-breaking noise.

### PRC — Price and margin coherence

Every rule here rests on the margin proxy defined above, and is therefore
weaker than it looks. They are included because price incoherence is a
real merchandising failure that a relevance objective will produce
happily, and because a policy with no commercial rules at all would be
conspicuously thin.

#### POL-PRC-01 — Price range coherence

**Severity:** `soft` · **Scope:** `slate` · **Critic criterion:** —

The ratio of the highest to the lowest price in a slate must not
exceed 6.0.

*Evaluated in code by* `price_ratio_at_most`

```json
{
  "max_ratio": 6.0
}
```

**Why:** A page mixing a premium coat with a multipack of socks reads as
incoherent regardless of how relevant each item is individually.

> **Calibration — provisional.** Revisit at **D1** · depends on price distribution in the reference window.
> 6.0 is a guess against an unfamiliar normalised price scale. Measure the real spread at D1 before trusting it.

#### POL-PRC-02 — Cohort price band

**Severity:** `soft` · **Scope:** `slate` · **Critic criterion:** —

For a cohort-targeted slate, at least 60% of articles must fall
between 0.5x and 2.0x the cohort's median historical item price.

*Evaluated in code by* `cohort_price_band_share_at_least`

```json
{
  "min_share": 0.6,
  "lower_multiple": 0.5,
  "upper_multiple": 2.0
}
```

**Why:** Recommending predominantly out-of-band product to a cohort is a
common and expensive personalisation failure.

> **Calibration — provisional.** Revisit at **D5** · depends on RFM and CLV cohort price distributions.
> Cannot be sensible before cohorts exist at D5.

#### POL-PRC-03 — Margin floor against baseline

**Severity:** `soft` · **Scope:** `slate` · **Critic criterion:** —

The mean margin proxy of a slate must be at least 85% of the mean
margin proxy of the popularity baseline slate for the same k.

*Evaluated in code by* `margin_at_least_share_of_baseline`

```json
{
  "min_ratio": 0.85,
  "baseline": "popularity"
}
```

**Why:** Guards the case where personalisation lifts conversion by steering
customers toward cheaper product — a real revenue increase that is a
margin decrease, and one a projected-revenue objective will not see.

**Known limitation:** Uniform margin rate means this reduces to a price-floor rule. Stated, not hidden.

> **Calibration — provisional.** Revisit at **D5** · depends on baseline slate construction, margin proxy.
> Meaningless until the popularity baseline is built at D3 and lift at D5.

#### POL-PRC-04 — No price claims in explanations

**Severity:** `hard` · **Scope:** `explanation` · **Critic criterion:** 6

An explanation may not assert a margin, profit or markup figure,
because none is observable. It may state the margin proxy only when
it is labelled as a proxy and the assumed rate is named.

*Evaluated in code by* `no_unlabelled_margin_claim`

```json
{
  "banned_unqualified_terms": [
    "margin",
    "profit",
    "markup",
    "gross margin"
  ],
  "required_qualifier": "proxy"
}
```

**Why:** The fastest way to lose credibility in a viva is to state a profit
number derived from a constant you invented.

### BRD — Presentation standards

Two enforced rules and one honest admission of what is not enforced.

#### POL-BRD-01 — Every article must have a usable image

**Severity:** `hard` · **Scope:** `article` · **Critic criterion:** —

An article may appear in a slate only if it has an image present in
the texture atlas manifest for the current catalogue version.

*Evaluated in code by* `article_in_atlas_manifest`

```json
{}
```

**Why:** DHAWQ is a gallery. An article with no photograph cannot be
merchandised, and the atlas manifest is the authoritative record of
which articles have one.

#### POL-BRD-02 — Article must resolve in the catalogue

**Severity:** `hard` · **Scope:** `article` · **Critic criterion:** —

Every article id in a slate must resolve to a row in the frozen
catalogue for the pinned catalogue version.

*Evaluated in code by* `article_resolves_in_catalogue`

```json
{}
```

**Why:** A hallucinated article id is the recommendation-system equivalent of
a hallucinated citation, and it is caught here rather than by a
404 in the UI.

#### POL-BRD-03 — Visual consistency of imagery

**Severity:** `advisory` · **Scope:** `slate` · **Critic criterion:** —

Slates should present consistent image treatment — comparable crop,
background and model framing across slots.

*Not machine-checked.* This rule must never be cited as grounds for rejection.

**Why:** A real merchandising team enforces this and it visibly affects page
quality.

**Known limitation:** NOT machine-checked at MVP. It would need an image-property model
that does not exist in this build. Recorded as advisory so that the
boundary of enforcement is explicit. The critic must never cite this
rule as grounds for rejection.

### SEG — Segment and targeting

Rules about who a slate is for, and whether it is permitted to exist.
These are the rules with a privacy dimension, and they are the ones that
bind most directly to the RBAC matrix in ARCHITECTURE.md §13.2.

#### POL-SEG-01 — Minimum cohort size

**Severity:** `escalate` · **Scope:** `cohort` · **Critic criterion:** —

A slate may not be produced for a cohort of fewer than 100 customers.
A brief naming a smaller cohort escalates to a human.

*Evaluated in code by* `cohort_size_at_least`

```json
{
  "min_customers": 100
}
```

**Why:** Two reasons that point the same way. Small cohorts produce unstable
estimates that look precise, and a sufficiently small cohort stops
being a segment and starts being an identifiable person.

> **Calibration — provisional.** Revisit at **D8** · depends on cohort sizes produced by D5 segmentation.
> 100 is a common k-anonymity-flavoured floor rather than a derived
> one. Confirm at D8 that at least one golden brief names a cohort
> below it, or the escalation path is never exercised.

#### POL-SEG-02 — No published slate for an individual

**Severity:** `hard` · **Scope:** `cohort` · **Critic criterion:** 9

A publishable merchandising slate is always cohort-scoped. Individual
customer recommendations are a runtime personalisation surface and
are never a publishable artefact.

*Evaluated in code by* `slate_scope_is_cohort`

```json
{}
```

**Why:** This is the policy expression of the ARCHITECTURE.md §13.2 row
denying the agent access to individual customer records. The agent
has no scope to read them; this rule ensures there is also no
legitimate reason for it to want to.

#### POL-SEG-03 — Cohorts must be specified, not enumerated

**Severity:** `hard` · **Scope:** `cohort` · **Critic criterion:** 9

A cohort must be expressed as a stored, re-executable specification.
A slate targeted at a pasted list of customer identifiers is rejected.

*Evaluated in code by* `cohort_is_specification`

```json
{}
```

**Why:** A specification is reproducible and auditable; a pasted list is
neither, and is also the most likely route for raw customer ids to
enter the system and then leave it in an output. The PII gate
(pii_leak_rate = 0.00) depends on this rule holding.

#### POL-SEG-04 — Relaxed quota for cold-start cohorts

**Severity:** `hard` · **Scope:** `cohort` · **Critic criterion:** 3

For cohorts defined by fewer than 3 prior purchases, the POL-LT-01
long-tail quota is relaxed from 20% to 10%.

*Evaluated in code by* `long_tail_share_at_least`

```json
{
  "min_share": 0.1,
  "rounding": "ceil",
  "applies_when_cohort": "purchase_count_lt_3"
}
```

*Overrides* `POL-LT-01` where both apply.

**Why:** Cold-start cohorts are served largely by content-based similarity,
which has weaker signal for judging whether a tail article is
genuinely a good match. Holding them to the full quota trades a real
relevance cost for an exposure gain the customer did not ask for.

> **Calibration — provisional.** Revisit at **D8** · depends on D4 cold-start curve, golden set cold-start stratum.
> The relaxation should be justified by the measured cold-start
> relevance gap at D4, not asserted. If the gap turns out to be
> small, this rule is an unearned exception and should be removed.

### CLM — Claims and language

These rules govern the explanation, not the slate. They exist because the
explanation is where an otherwise honest system starts making claims its
evidence does not support — and because ARCHITECTURE.md commits, in
several places, to saying "projected" and never "measured".

#### POL-CLM-01 — Lift and revenue figures are projected

**Severity:** `hard` · **Scope:** `explanation` · **Critic criterion:** 6

Every revenue, lift or uplift figure must be described as projected,
estimated or modelled. Causal and past-tense claims of measured
effect are prohibited.

*Evaluated in code by* `no_banned_causal_language`

```json
{
  "banned_terms": [
    "increased revenue",
    "drove",
    "caused",
    "resulted in",
    "measured lift",
    "proved",
    "demonstrated that",
    "led to",
    "generated an uplift",
    "delivered a lift"
  ],
  "required_qualifier_any_of": [
    "projected",
    "estimated",
    "modelled",
    "would be expected"
  ]
}
```

**Why:** There is no live A/B test anywhere in DHAWQ. Every lift number is an
offline estimate under strong assumptions. Enforcing the vocabulary
in code is more reliable than asking a model to be careful.

> **Calibration — provisional.** Revisit at **D8** · depends on explanation phrasing observed at D9.
> The banned-term list is a denylist and denylists leak. Expect to
> extend it once real explanations exist. Every leak found becomes a
> permanent golden-set regression case per §10.4.

#### POL-CLM-02 — No causal claim without an experiment

**Severity:** `hard` · **Scope:** `explanation` · **Critic criterion:** 6

A causal claim requires a cited experiment identifier. DHAWQ runs no
experiments, so no causal claim can be made.

*Evaluated in code by* `causal_claim_requires_experiment_id`

```json
{
  "experiments_available": false
}
```

**Why:** Stated as a general rule with a currently-empty precondition rather
than a flat prohibition, so that the rule remains correct if an
experiment is ever added.

#### POL-CLM-03 — Numeric claims cite their producing tool call

**Severity:** `hard` · **Scope:** `explanation` · **Critic criterion:** 1

Every numeric claim must carry an evidence_id resolving to the tool
call that produced the number.

*Evaluated in code by* `numeric_claims_have_tool_evidence`

```json
{}
```

**Why:** The policy expression of the central rule in ARCHITECTURE.md §0.1 —
no model emits a number. If a number appears in an explanation and
cannot be traced to a deterministic tool call, a model produced it,
and the run has violated the one rule the system is built around.

#### POL-CLM-04 — Predicted attributes labelled when cited

**Severity:** `hard` · **Scope:** `explanation` · **Critic criterion:** 6

Where an explanation cites season, usage or fine colour, it must
identify the attribute as model-predicted.

*Evaluated in code by* `predicted_attributes_labelled`

```json
{
  "predicted_attributes": [
    "season",
    "usage",
    "fine_colour"
  ]
}
```

**Why:** ARCHITECTURE.md §3 names passing predictions off as metadata as the
failure mode of the enrichment approach. This is where that is caught.

#### POL-CLM-05 — Confidence requires evidence coverage

**Severity:** `escalate` · **Scope:** `explanation` · **Critic criterion:** 8

An explanation may state a confidence level only when evidence
coverage is at least 0.80. Below that the slate may still be
produced, but confidence is suppressed and the run is flagged.

*Evaluated in code by* `confidence_requires_coverage`

```json
{
  "min_coverage": 0.8,
  "below_threshold_behaviour": "suppress_confidence_and_flag"
}
```

**Why:** Directly implements criterion 8. Suppressing confidence rather than
rejecting the slate is the calibration-honest response — the system
may still be right, it just has no basis for saying how sure it is.

> **Calibration — ⚠︎ UNGROUNDED.** Revisit at **D8** · depends on 6 unanswerable golden briefs, D11 calibration curve, Brier score.
> 0.80 is invented. The 6 unanswerable briefs are the only thing that
> can define it: they are the cases where coverage SHOULD fall below
> the line. Set this to the value that separates them from the
> answerable briefs, then verify against the reliability curve at D11.

### ESC — Escalation

When the system must stop and ask. ARCHITECTURE.md §7.7 places a gate
wherever the cost of being wrong exceeds the cost of asking; this domain
is the policy-side expression of those gates. The refusal path gets the
same design care as the happy path.

#### POL-ESC-01 — Unresolvable quota or diversity breach escalates

**Severity:** `escalate` · **Scope:** `run` · **Critic criterion:** —

Where the optimiser cannot satisfy POL-LT-01 or POL-DIV-01 within the
candidate set, the run escalates to a human. The agent may not relax
the constraint, and may not return a breaching slate as though it
complied.

*Evaluated in code by* `unresolvable_constraint_escalates`

```json
{
  "rules": [
    "POL-LT-01",
    "POL-DIV-01"
  ]
}
```

**Why:** This is the policy-override gate in §7.7. Whether to trade exposure
for revenue is a commercial decision, and it belongs to a person.

#### POL-ESC-02 — Brief-versus-policy conflict escalates

**Severity:** `escalate` · **Scope:** `run` · **Critic criterion:** —

Where an explicit instruction in a brief cannot be satisfied without
breaching a hard rule, the run escalates, stating both the brief's
requirement and the rule it conflicts with. The agent does not
silently choose between them.

*Evaluated in code by* `brief_policy_conflict_escalates`

```json
{}
```

**Why:** This rule is what the constraint-conflicting stratum of the golden
set tests. A brief demanding maximum revenue at 12 slots with a 20%
tail quota is a genuine conflict, and the correct behaviour is to
surface it, not to quietly pick a side and present the result as
though nothing was traded away.

#### POL-ESC-03 — Sub-threshold cohort escalates

**Severity:** `escalate` · **Scope:** `run` · **Critic criterion:** —

A brief naming a cohort below POL-SEG-01 escalates rather than proceeding.

*Evaluated in code by* `escalate_on_rule`

```json
{
  "rule": "POL-SEG-01"
}
```

**Why:** Privacy and stability, per POL-SEG-01.

#### POL-ESC-04 — Low evidence coverage flags rather than blocks

**Severity:** `escalate` · **Scope:** `run` · **Critic criterion:** 8

Where evidence coverage falls below POL-CLM-05, the slate is produced
with confidence suppressed and the run is flagged for review. It is
not rejected.

*Evaluated in code by* `escalate_on_rule`

```json
{
  "rule": "POL-CLM-05"
}
```

**Why:** Thin evidence is a reason to stop claiming certainty, not a reason to
refuse to answer. Conflating the two produces a system that refuses
constantly and is therefore ignored.

#### POL-ESC-05 — Repeated failure escalates

**Severity:** `escalate` · **Scope:** `run` · **Critic criterion:** —

Where the same class of tool or validation failure occurs twice in a
run, the run escalates rather than retrying a third time.

*Evaluated in code by* `repeat_failure_escalates`

```json
{
  "max_same_class_failures": 2
}
```

**Why:** The repeat-failure gate in §7.7, and the budget defence against a
retry loop that burns the whole allowance on one broken call.

#### POL-ESC-06 — Override authority is never the agent's

**Severity:** `hard` · **Scope:** `run` · **Critic criterion:** 9

Waiving any rule in this policy requires the policy override scope.
The agent role does not hold that scope under any circumstances,
including when the caller does.

*Evaluated in code by* `override_requires_scope`

```json
{
  "required_scope": "policy:override",
  "agent_may_hold": false
}
```

**Why:** The policy expression of the intersection rule in §13.3 — an admin
submitting a brief does not lend the agent admin authority. A policy
that the executing agent can waive is not a policy.

### GOV — Governance

How this policy is versioned, cited, waived and outgrown.

#### POL-GOV-01 — Policy version pinned in every run

**Severity:** `hard` · **Scope:** `run` · **Critic criterion:** —

Every run record stores the policy version and content hash that was
in force. A rejection citing a rule id is only interpretable against
the version that produced it.

*Evaluated in code by* `run_records_policy_version`

```json
{}
```

**Why:** Without this, a rejection from last week cannot be reproduced once a
threshold moves, and the D8 recalibration would invalidate every
historical run silently.

#### POL-GOV-02 — Waivers are recorded

**Severity:** `hard` · **Scope:** `run` · **Critic criterion:** —

A waiver records the rule id, the human actor, a reason, and a
timestamp. Waivers are per-run and never persist to later runs.

*Evaluated in code by* `waiver_is_fully_recorded`

```json
{
  "required_fields": [
    "rule_id",
    "actor_id",
    "reason",
    "timestamp"
  ],
  "persists": false
}
```

**Why:** A waiver that carries forward silently becomes a policy change nobody
approved.

#### POL-GOV-03 — Corpus C is loaded, not retrieved

**Severity:** `hard` · **Scope:** `system` · **Critic criterion:** —

This policy is loaded whole into context. It is not chunked, embedded
or retrieved. If it exceeds 200,000 tokens or roughly 500 pages it
graduates to retrieval and this rule is revisited.

*Evaluated in code by* `corpus_c_within_context_budget`

```json
{
  "max_tokens": 200000,
  "max_pages": 500
}
```

**Why:** ARCHITECTURE.md §8.2. A critic that reads the entire policy every
time cannot miss a rule because a chunk failed to rank. The current
size is recorded in manifest.json so the threshold is monitored
rather than assumed.

#### POL-GOV-04 — This policy is not authoritative

**Severity:** `hard` · **Scope:** `explanation` · **Critic criterion:** 6

No output may represent this policy as H&M's actual merchandising
policy, or as sourced from a real buying team. Where an explanation
cites a rule, it cites it as a DHAWQ project policy rule.

*Evaluated in code by* `no_false_policy_attribution`

```json
{
  "banned_attributions": [
    "H&M policy",
    "retailer policy",
    "industry standard",
    "buying team policy"
  ]
}
```

**Why:** Required by the honest-limitations list in ARCHITECTURE.md §3. Placed
in the enforced rules rather than the preamble because a limitation
that is only stated in a README is a limitation that will be
contradicted by an explanation nobody checked.

#### POL-GOV-05 — Precedence on conflict

**Severity:** `hard` · **Scope:** `system` · **Critic criterion:** —

Where two rules cannot both be satisfied, the domain earlier in the
declared precedence order wins. The optimiser objective is last and
never overrides a constraint. Where two rules within one domain
conflict, the more specific scope wins (article over slate over
campaign); if still tied, the run escalates.

*Evaluated in code by* `precedence_respected`

```json
{
  "order_ref": "precedence"
}
```

**Why:** An unordered rule set is not machine-checkable — the first conflict
makes the outcome depend on evaluation order, which is a bug that
looks like a judgement call.
