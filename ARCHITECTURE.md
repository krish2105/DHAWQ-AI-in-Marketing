# DHAWQ — ذوق
**Visual Recommendation Intelligence**
MAIB AI 208 · AI in Marketing · SP Jain Dubai · Krishna Mathur

---

## 0. CTO framing — read before the architecture

The instinct with a fashion dataset is to build "a recommender." That's a
model, not a product, and AI 208 is a *marketing* subject — it grades
marketing science, not cosine similarity.

The framing that makes this a marketing project:

> **A merchandiser has a finite number of slots on a page. Which products go
> in them, for whom, and how much incremental revenue does that choice
> create versus showing everyone the bestsellers?**

That reframes every component. The recommender is the engine; the graded
contribution is the *evaluation* — popularity bias, coverage, cold start,
and incremental lift over a business-as-usual baseline. A model that
achieves NDCG@10 of 0.31 means nothing to a CMO. "Personalisation lifts
projected revenue per session by X% but concentrates 60% of impressions on
4% of the catalogue" is a business decision.

**The number you defend in the viva:** incremental revenue per session over
a popularity baseline, with the long-tail exposure cost stated alongside it.

---

## 1. The 30-second demo

44,000 real fashion products floating in 3D space, clustered by learned
visual and semantic similarity — you can see the shape of the catalogue.
Pick a shirt. The camera **flies** to it, its neighbours illuminate, and
recommendations appear as real product photographs. Toggle **"why this?"**
and the embedding distances render as lines with their contributions.

Then switch to the merchandiser view: a simulated page of slots, filled by
your model versus the popularity baseline, with the projected revenue delta
and the catalogue-coverage cost side by side.

---

## 2. Research questions

**Primary (graded):**
Does a hybrid recommender combining visual embeddings with collaborative
signal beat both a content-only and a popularity baseline on ranking
quality — and what does it cost in catalogue coverage and long-tail
exposure?

**Secondary (the marketing claim):**
At a fixed page budget of *k* slots, how much projected incremental revenue
per session does personalisation generate over showing bestsellers to
everyone, and how does that gap change for cold-start users with fewer than
3 prior purchases?

Both are testable offline. Both produce numbers a CMO would act on.

---

## 3. Data

### Primary: H&M Personalized Fashion Recommendations
| Field | Detail |
|---|---|
| Transactions | ~31.8M purchase records |
| Customers | ~1.37M |
| Articles | ~105k, with metadata |
| Images | Real product photography for most articles |
| Period | Two years, dated — enables temporal splits |

`kaggle competitions download -c h-and-m-personalized-fashion-recommendations`

**Why this and not Fashion Product Images:** the 44k Fashion Product Images
dataset has *no user-item interactions*. Without them you cannot do
collaborative filtering, cold start, CLV, or incremental lift — you'd be
building a visual similarity search and calling it a recommender. H&M has
real purchases by real customers over real time.

**Subsample deliberately.** Full H&M is ~25GB. Take the most recent 12
weeks, articles with ≥ 20 purchases, customers with ≥ 3 transactions. That's
~10–15k articles and ~50k customers — enough for every method, small enough
to iterate on a laptop. **Document the subsampling rule**; it's a
methodological choice, not a convenience.

### Supplement: Fashion Product Images — **confirmed in scope**
`kaggle datasets download -d paramaggarwal/fashion-product-images-small`

44k products with richer attribute metadata than H&M's taxonomy: `gender`,
`masterCategory`, `subCategory`, `articleType`, `baseColour`, `season`,
`year`, `usage`. H&M's `product_type_name` and `colour_group_name` are
thinner.

**How the two join.** They are *different catalogues* — there is no shared
product key, so do not attempt a row-level join. Use Fashion Product Images
two ways instead:

1. **Attribute vocabulary.** Train a lightweight attribute classifier on
   Fashion Product Images (it has clean labels), then run it over H&M
   imagery to enrich H&M articles with `season`, `usage` and finer colour
   than H&M provides. Document this as a derived, predicted attribute — not
   ground truth.
2. **Cold-start stress test.** Hold out Fashion Product Images items as
   genuinely-unseen articles with zero interaction history. Your
   content-based recommender should still place them sensibly in the
   embedding space. That's a clean, honest cold-start evaluation using real
   images from outside the training catalogue.

**State clearly:** enriched attributes on H&M articles are model-predicted,
with the classifier's accuracy reported. Passing predictions off as metadata
would be the failure mode here.

### Honest limitations
- Purchases, not views. You observe conversions, never impressions — so
  "the user didn't buy it" ≠ "the user rejected it"
- No price experiments, no real A/B test. Lift is *projected* from offline
  evaluation, not measured causally. Say "projected" everywhere
- Position bias in the original data is unobservable

---

## 4. Architecture

```
┌───────────────────────────────────────────────────────────────┐
│  apps/web — Next.js 16                                        │
│  3D embedding space · recommendations · merchandiser sim      │
└──────────────────────────┬────────────────────────────────────┘
                           │ httpOnly JWT
┌──────────────────────────▼────────────────────────────────────┐
│  services/api — FastAPI                                       │
│  ┌────────┬──────────┬──────────┬──────────┬───────────────┐  │
│  │ recs   │ embed    │ evaluate │ segments │ merchandise   │  │
│  └───┬────┴────┬─────┴────┬─────┴────┬─────┴──────┬────────┘  │
└──────┼─────────┼──────────┼──────────┼────────────┼───────────┘
       │         │          │          │            │
 ┌─────▼───┐ ┌───▼─────┐ ┌──▼──────┐ ┌─▼────────┐ ┌─▼─────────┐
 │ Hybrid  │ │  CLIP   │ │ Ranking │ │ RFM +    │ │ Slot      │
 │ ranker  │ │ + UMAP  │ │ metrics │ │ BG/NBD   │ │ optimiser │
 │         │ │ 3D proj │ │ + bias  │ │ CLV      │ │           │
 └─────────┘ └─────────┘ └─────────┘ └──────────┘ └───────────┘
```

### Repo layout

```
dhawq/
├── apps/web/
│   ├── app/
│   │   ├── page.tsx                3D embedding space
│   │   ├── product/[id]/           detail + recommendations + "why this"
│   │   ├── merchandise/            slot simulator, baseline comparison
│   │   ├── segments/               CLV / RFM cohorts
│   │   └── evaluate/               metrics, bias, coverage
│   ├── components/space/           R3F embedding scene
│   ├── components/product/         cards, grids, explanation overlay
│   └── components/ui/
├── services/api/
│   ├── routers/
│   ├── models/
│   │   ├── content.py              CLIP embedding similarity
│   │   ├── collaborative.py        implicit ALS / item-item
│   │   ├── hybrid.py               weighted / cascade blend
│   │   └── baseline.py             popularity + recency
│   ├── embed/
│   │   ├── extract.py              open_clip, local, batched
│   │   ├── project.py              UMAP → 3D, cached
│   │   └── index.py                pgvector / FAISS
│   ├── evaluate/
│   │   ├── ranking.py              precision@k, recall@k, NDCG, MAP
│   │   ├── beyond_accuracy.py      coverage, diversity, novelty, serendipity
│   │   ├── bias.py                 popularity bias, Gini, long-tail exposure
│   │   └── coldstart.py            stratified by user history depth
│   ├── marketing/
│   │   ├── rfm.py                  segmentation
│   │   ├── clv.py                  BG/NBD + Gamma-Gamma
│   │   ├── slots.py                page budget optimiser
│   │   └── lift.py                 projected incremental revenue
│   └── core/security.py
└── data/
```

---

## 5. Layer 1 — Embeddings

**Model:** `open_clip` **ViT-L-14**, weights `laion2b_s32b_b82k`, run
**locally on MPS** (Apple Silicon). Free, no API.

**Why L/14 and not B/32 on this machine:** encoding is a one-time cost. On an
M-series Mac with MPS, ViT-L/14 processes the ~15k subsampled articles in
roughly 10–15 minutes at batch size 32–64. B/32 would take ~3 minutes but
produces measurably weaker retrieval on fine-grained visual distinctions —
and fine-grained is the whole point when the catalogue is 15k garments that
differ by cut, texture and pattern. Twelve extra minutes, once, for better
embeddings across every downstream metric.

**Fallback:** if MPS runs out of memory, drop batch size to 16 before
dropping to B/32. Cache embeddings to disk as `.npy` immediately after
extraction — never re-encode.

**Why CLIP and not ResNet:** CLIP's joint image-text space means "sleeveless
navy midi dress" and a photograph of one land near each other. That enables
natural-language search as a near-free feature and makes the "why this?"
explanation legible.

**Projection:** UMAP to 3D. **Fit once, cache the coordinates.** Never
recompute per request — UMAP is not deterministic across runs and the space
must stay stable between sessions or the demo breaks.

Store: full-dimension vectors in **pgvector** for retrieval; the 3D
coordinates as a plain cached table for the scene.

---

## 6. Layer 2 — The recommenders

| Model | Role |
|---|---|
| **Popularity + recency** | The business-as-usual baseline. Every claim is measured against this, not against nothing |
| **Content-based** (CLIP kNN) | Solves cold start for new articles — no interaction history needed |
| **Collaborative** (implicit ALS or item-item) | Captures "bought together" signal invisible to images |
| **Hybrid** | Weighted blend, or cascade: collaborative where history is sufficient, content where it isn't |

**Temporal split, never random.** H&M is dated. Train on weeks 1–10, test on
11–12. Assert `max(train_date) < min(test_date)` in a test. A random split
lets the model see the future and inflates every metric.

---

## 7. Layer 3 — Evaluation (this is what AI 208 grades)

### Ranking quality
Precision@k, Recall@k, **NDCG@k**, MAP@k, MRR — at k = 5, 10, 20.

### Beyond-accuracy — the part most projects skip
| Metric | Why a marketer cares |
|---|---|
| **Catalogue coverage** | % of articles ever recommended. Low coverage means dead inventory |
| **Gini / long-tail exposure** | How concentrated are impressions? |
| **Popularity bias** | Does the model just re-rank bestsellers? Measure it explicitly |
| **Intra-list diversity** | Ten near-identical black t-shirts is a bad page |
| **Novelty** | Are recommendations surprising relative to popularity? |
| **Serendipity** | Relevant *and* unexpected |

**The tension is the finding.** Accuracy and coverage trade off. Plot the
frontier. A model that wins NDCG while collapsing coverage to 4% of the
catalogue is a merchandising problem, and naming that is a distinction-level
observation.

### Cold start
Stratify every metric by user history depth: 0 purchases, 1–2, 3–9, 10+.
Report the curve. Personalisation that only works for heavy buyers is a
known and important limitation.

---

## 8. Layer 4 — The marketing layer

**RFM segmentation** — recency, frequency, monetary. Standard, expected,
cheap.

**CLV** — BG/NBD for purchase frequency, Gamma-Gamma for monetary value,
via `lifetimes`. Holdout-validated: fit on the first period, predict the
second, plot predicted vs actual.

**Slot optimiser** — given *k* page slots and a customer, choose the set
maximising projected revenue subject to a diversity constraint and a minimum
long-tail quota. This is where the model becomes a merchandising decision.

**Projected incremental lift**
```
lift = Σ (P(purchase | recommended) × margin)  —  same for baseline
```
Say **projected**, never "measured." Without a live A/B test this is an
offline estimate, and the honest framing is what makes it credible.

---

## 9. Frontend — design direction and 3D

### 9.1 The design thesis — deliberately unlike your other projects

RASID is a severity console. HISBAH is a control room. MASAR is a map.
**DHAWQ is a gallery.**

The product photography *is* the colour. 44,000 real garment images will
supply every hue on screen — so the interface must recede almost entirely.
Deep neutral ground, near-monochrome chrome, generous negative space, and
one saturated signal colour that appears only on selection.

This is not a stylistic preference; it's a functional requirement. A colourful
UI around a fashion catalogue fights the merchandise and makes colour-based
recommendations impossible to judge visually.

### 9.2 Palette

```
--void        near-black, slightly warm     #0B0A09  (dark)
--paper       warm off-white, not pure      #FAF8F5  (light)
--surface     one step from ground
--hairline    1px borders, very low contrast
--text        high contrast
--text-muted  ~60% — metadata only
--signal      ONE saturated accent, selection + active state only
--tail        muted secondary for long-tail / coverage viz
```

**Deliberately avoided:** purple-blue gradients, cream-and-terracotta
serifs, acid green on black, glassmorphism everywhere, untouched shadcn
defaults, emoji icons, three-feature-card rows.

**Signal colour suggestion:** a single high-chroma value that appears
nowhere in typical garment photography — an electric cyan or a hot magenta
— so selection never gets visually confused with a product's own colour.
Pick one, use it nowhere else.

### 9.3 Type

- **UI:** one variable sans, tight tracking, optical sizing on
- **Numbers:** mono with `tabular-nums` — metrics update live, must not
  jitter
- **Product names:** slightly larger, generous leading. This is the one
  place editorial typography is appropriate
- Fluid `clamp()` sizing, no breakpoint jumps

### 9.4 Theme toggle

Dark designed **first** — a gallery at night, images glowing off deep
ground. Light is separately authored: warm paper, not inverted greys.
`next-themes` class strategy, inline head script, no FOUC. Both ≥ 4.5:1 on
body text. Measure and report the actual ratios.

### 9.5 The 3D embedding space — the signature moment

**What it renders:** every product as a **textured plane** (its actual
photograph) positioned at its UMAP coordinate. You are literally flying
through the catalogue's latent structure. Clusters are real — dresses drift
from footwear, colours gradient across regions.

**Stack:** React Three Fiber + `@react-three/drei`.

| Concern | Approach |
|---|---|
| Geometry | `InstancedMesh` of planes. 15k instances, one draw call |
| Textures | **Texture atlas** — pack thumbnails into a few 4096² sheets, index by instance UV offset. 15k individual texture loads will kill the browser |
| LOD | Distant products render as coloured points (dominant colour); textures resolve only within a camera radius |
| Camera | `OrbitControls` with damping. **`flyTo` on selection** — smooth eased transition, not a jump cut. This is the moment |
| Interaction | GPU picking or raycast against instances → hover card with product name, category, price |
| Neighbours | On select, draw `Line` segments to top-k neighbours, opacity ∝ similarity |
| "Why this?" | Toggle: lines annotate with contribution — visual similarity vs collaborative signal |
| Post-processing | None by default. Optional very subtle vignette. No bloom — it would wash out product colour, which is the content |

**Loading:** progressive. Points first, textures stream in. Never a blank
canvas with a spinner.

**Mobile — non-negotiable:**
- `dpr={[1, 1.5]}`
- Reduce to ~3k instances below 768px
- Smaller atlas resolution
- Larger touch targets
- 30fps mobile / 60fps desktop

**2D fallback toggle — mandatory.** A 2D scatter or a plain grid rendering
identical data. Accessibility, WebGL failure, and your own debugging escape
hatch.

### 9.6 Views

| View | Content |
|---|---|
| **Space** | The 3D embedding scene. Search bar drives a `flyTo` |
| **Product** | Large image, recommendations as a filmstrip, "why this?" overlay |
| **Merchandise** | Side-by-side page simulation: your model vs popularity baseline, projected revenue delta, coverage cost |
| **Segments** | RFM cohorts, CLV distribution, cold-start curve |
| **Evaluate** | Ranking metrics, the accuracy-coverage frontier, popularity-bias plots |

### 9.7 Performance budget

- LCP < 2.5s on the product route
- 3D canvas lazy-mounted, never blocks first paint
- UMAP coordinates and texture atlases precomputed and CDN-cached
- Virtualised grids (`@tanstack/react-virtual`)
- R3F code-split out of the initial bundle
- Lighthouse accessibility ≥ 95

### 9.8 Stack

Next.js 16 (App Router) · React 19.2.4+ · TypeScript · Tailwind v4 ·
shadcn/ui (restyled) · `next-themes` · `motion/react` · React Three Fiber +
drei · Recharts · `@tanstack/react-virtual`.

---

## 10. Security

| Control | Implementation |
|---|---|
| Auth | OAuth2 → JWT in **httpOnly cookies** |
| Hashing | `pwdlib` Argon2 |
| Tokens | Short access + refresh rotation |
| RBAC | `viewer` / `merchandiser` (run simulations) / `admin` |
| Rate limiting | `slowapi` — the recommendation endpoint is the expensive one |
| CORS | Explicit allowlist, never `["*"]` with credentials |
| Headers | CSP, HSTS, X-Frame-Options |
| Customer data | H&M customer IDs are pseudonymous — never expose raw IDs in URLs; no PII in embeddings |
| Images | Served from CDN with signed URLs; never hotlink |
| Audit | Segment exports and simulation runs logged |

OWASP API Security Top 10 framing.

---

## 11. Sessions

| # | Session | Model | Mode | Effort | Hrs |
|---|---|---|---|---|---|
| D0 | Kickoff / architecture | **Opus** | Plan | high | 1.5 |
| D1 | Data ingest + subsample + temporal split | Sonnet | Accept | medium | 5 |
| D2 | CLIP embeddings + UMAP + atlas generation | **Opus** | Plan→Accept | high | 8 |
| D3 | Baselines + content + collaborative + hybrid | **Opus** | Plan→Accept | high | 9 |
| D4 | Evaluation: ranking + beyond-accuracy + cold start | **Opus** | Plan→Accept | high | 7 |
| D5 | Marketing layer: RFM, CLV, slots, lift | Sonnet | Accept | medium | 6 |
| D6 | FastAPI service | Sonnet | Accept | medium | 5 |
| D7 | Web scaffold + gallery design system + theme | Sonnet | Accept | medium | 6 |
| D8 | 3D embedding space (instancing, atlas, LOD) | **Opus** | Plan→Accept | high | 11 |
| D9 | flyTo, neighbours, "why this?" overlay | **Opus** | Plan→Accept | high | 6 |
| D10 | Merchandise simulator + segments + evaluate views | Sonnet | Accept | medium | 7 |
| D11 | Auth + security | Sonnet | Accept | medium | 4 |
| D12 | Deploy + polish + recording | Sonnet | Accept | medium | 4 |

**Total ≈ 80 hours** with the full 3D scene.
**≈ 55 hours** if D8/D9 collapse to a 2D UMAP scatter with a good grid UI.

---

## 12. Deliverables

- Live deployed URL, phone-usable, gallery theme with dark/light toggle
- 3D embedding space: 15k textured product planes, flyTo, neighbour lines
- "Why this?" explanation overlay separating visual from collaborative signal
- Four recommenders benchmarked: popularity, content, collaborative, hybrid
- Temporal-split evaluation with a leak assertion in tests
- Ranking metrics: precision@k, recall@k, NDCG, MAP, MRR
- Beyond-accuracy: coverage, Gini, popularity bias, diversity, novelty,
  serendipity
- **The accuracy–coverage frontier plot** — the finding
- Cold-start curve stratified by user history depth
- RFM segments + holdout-validated CLV
- Merchandiser slot simulator with projected incremental revenue vs baseline
- Full auth, RBAC, audit
- README with honest limitations
- 90-second recording + ~12-slide deck

---

## 13. Viva Q&A

**Q: Why CLIP rather than a plain CNN?**
Because CLIP's joint image-text space lets a natural-language query and a
photograph occupy the same neighbourhood. That gives free text search and
makes the "why this?" explanation legible to a merchandiser, who thinks in
words, not feature maps.

**Q: Your model beats the baseline on NDCG. So what?**
On its own, nothing. The number a merchandiser acts on is projected
incremental revenue per session at a fixed slot budget — and the cost side
is catalogue coverage. My frontier plot shows the hybrid gains X% NDCG while
concentrating impressions on Y% of articles. Whether that trade is worth it
is a business decision, and I present it as one.

**Q: Why is it "projected" lift and not measured?**
Because there is no live A/B test. Offline evaluation on held-out purchases
estimates what *would* have happened under strong assumptions — no position
bias, no interference. I state those assumptions rather than laundering an
estimate into a claim.

**Q: The data has purchases but no impressions. What does that break?**
Everything about negatives. I never observe "shown and rejected," only
"bought." So unpurchased items are unlabelled, not negative, and every
ranking metric inherits that. It's the single biggest limitation and it's
inherent to the dataset, not to my method.

**Q: What would falsify your hypothesis?**
If the hybrid failed to beat content-only and collaborative-only under the
temporal split, the blend would be adding complexity for nothing and the
right engineering call would be the simpler model. That's a real possible
outcome and I'd report it.
