"""D7 — the adaptive retrieval router (ARCHITECTURE.md §8.1, PLAN.md §6).

THE PROPERTY THAT MATTERS: the model classifies the query's SHAPE; CODE maps
shape to strategy. The model never picks a strategy.

That split is what keeps routing inside "models do routing" as §0.1 means it —
routing as extraction, not routing as decision — and it is what makes
`retrieval_routing_accuracy` a well-defined classification metric rather than a
vibe. SHAPE_TABLE is a frozen dict; if it were a model output there would be
nothing to score against.

Three deliberate behaviours:

  Stage 1 pre-empts the model entirely for the highest-precision routes.
  "What is the long-tail quota" must reach corpus C every single time; a
  lexicon match is 1.0 accurate and a classifier is not.

  Low confidence FANS OUT and fuses rather than guessing. One extra retrieval
  is cheaper than a wrong answer delivered with confident citations.

  Corpus D is DEFAULT-DENY. The cheapest defence against an injection in
  crawled content is not retrieving crawled content.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

QueryShape = Literal[
    "taxonomic", "multi_hop_relational", "visual_semantic",
    "numeric_historical", "policy", "market_context",
]

Strategy = Literal[
    "graph_traverse", "graph_node_lookup", "graph_path", "dense_clip",
    "structured_plus_dense", "load_full_corpus", "hybrid_rerank_untrusted",
    "fanout_rrf", "refuse",
]

Corpus = Literal["A", "B", "C", "D", "A+B", None]

TAU_ROUTE = 0.60          # below this, fan out rather than commit


# ── Stage 1 · deterministic pre-emption ──────────────────────────────────────

POLICY_LEXICON = re.compile(
    r"\b(polic\w+|quota|long[- ]tail|diversity floor|rule|compliance|"
    r"allowed|permitted|breach|override|guardrail|POL-[A-Z]{2,4}-\d{2})\b",
    re.IGNORECASE,
)
ARTICLE_CODE_RE = re.compile(r"\b\d{9,10}\b")
RUN_REF_RE = re.compile(r"\b(run[_ ]?(id)?[ _]?\d+|run_\d{8}_\d{6})\b", re.IGNORECASE)


# ── Stage 3 · the frozen table. CODE, not a model output. ────────────────────

SHAPE_TABLE: dict[str, tuple[str, str]] = {
    "taxonomic":           ("graph_traverse", "A"),
    "multi_hop_relational": ("graph_path", "A"),
    "visual_semantic":     ("dense_clip", "A"),
    "numeric_historical":  ("structured_plus_dense", "B"),
    "policy":              ("load_full_corpus", "C"),
    "market_context":      ("hybrid_rerank_untrusted", "D"),
}


@dataclass(frozen=True)
class BriefContext:
    allow_external: bool = False       # corpus D is opt-in, never default
    cohort_scoped: bool = False
    has_article_ref: bool = False


@dataclass(frozen=True)
class RouteDecision:
    shape: str
    strategy: str
    corpus: str | None
    decided_by: str
    confidence: float
    fanout: tuple[str, ...] = field(default_factory=tuple)

    @property
    def by_rule(self) -> bool:
        """Rule-fired routes are free wins. Reporting them merged with the
        classifier's accuracy would inflate routing accuracy, so the evaluation
        separates them on this flag."""
        return self.decided_by.startswith("rule:")

    def as_dict(self) -> dict:
        return {"shape": self.shape, "strategy": self.strategy, "corpus": self.corpus,
                "decided_by": self.decided_by, "confidence": round(self.confidence, 3),
                "fanout": list(self.fanout), "by_rule": self.by_rule}


class ShapeClassification(BaseModel):
    """The ONLY thing the model is asked for. Note there is no `strategy`
    field — the model is structurally unable to choose one."""
    shape: QueryShape
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(max_length=200)


CLASSIFY_SYSTEM = """You classify the SHAPE of a merchandising retrieval query.
You do not choose a retrieval strategy and you do not answer the query.

Shapes:
- taxonomic: about categories, types or hierarchy ("what else in this subcategory")
- multi_hop_relational: relationships across entities ("what co-sells with what this cohort bought")
- visual_semantic: about appearance ("something like this but lighter")
- numeric_historical: about past evaluation runs or metrics ("why did coverage drop")
- policy: about rules, quotas or what is permitted
- market_context: about external trends or the market outside our data

Reply with ONLY a JSON object:
{"shape": "<one of the six>", "confidence": <0.0-1.0>, "reason": "<short>"}"""


def classify_shape(query: str, provider=None) -> ShapeClassification:
    """Stage 2. Extraction, not decision."""
    from services.api.agent.llm import LLMError, Message, for_task, parse_structured

    provider = provider or for_task("classify")
    try:
        resp = provider.complete(CLASSIFY_SYSTEM, [Message("user", query)],
                                 max_tokens=150, temperature=0.0)
        return parse_structured(resp.text, ShapeClassification)
    except (LLMError, Exception):
        # A classifier failure must not take the run down. Return
        # low confidence and let stage 3 fan out — degrading to a broader
        # search is the correct failure mode for a router.
        return ShapeClassification(shape="taxonomic", confidence=0.0,
                                   reason="classifier unavailable; fanning out")


def route(query: str, ctx: BriefContext | None = None, provider=None) -> RouteDecision:
    ctx = ctx or BriefContext()

    # ── Stage 1 · rules. No model call. ──────────────────────────────────────
    if POLICY_LEXICON.search(query):
        return RouteDecision("policy", "load_full_corpus", "C",
                             "rule:policy_lexicon", 1.0)
    if ARTICLE_CODE_RE.search(query):
        return RouteDecision("taxonomic", "graph_node_lookup", "A",
                             "rule:article_code", 1.0)
    if RUN_REF_RE.search(query):
        return RouteDecision("numeric_historical", "structured_plus_dense", "B",
                             "rule:run_ref", 1.0)

    # ── Stage 2 · model classifies shape only ────────────────────────────────
    c = classify_shape(query, provider)

    # ── Stage 3 · deterministic mapping ──────────────────────────────────────
    if c.confidence < TAU_ROUTE:
        return RouteDecision(c.shape, "fanout_rrf", "A+B",
                             "fallback:low_confidence", c.confidence,
                             fanout=("graph_traverse", "dense_clip",
                                     "structured_plus_dense"))

    if c.shape == "market_context" and not ctx.allow_external:
        # Default-deny. The brief did not ask for external context, so we do
        # not reach for untrusted content.
        return RouteDecision(c.shape, "refuse", None,
                             "rule:external_default_deny", c.confidence)

    strategy, corpus = SHAPE_TABLE[c.shape]
    return RouteDecision(c.shape, strategy, corpus, "model:shape_classifier", c.confidence)
