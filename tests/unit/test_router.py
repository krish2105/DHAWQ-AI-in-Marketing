"""Router tests — PLAN.md §6.

THE INVARIANT UNDER TEST: the model classifies shape; code maps shape to
strategy. Several of these assert on the ABSENCE of a capability, which is the
point — the model is structurally unable to pick a strategy, so there is no
path by which a bad classification becomes an arbitrary retrieval.
"""

from __future__ import annotations

import pytest

from services.api.rag.router import (
    SHAPE_TABLE, BriefContext, RouteDecision, ShapeClassification, route,
)


class FakeClassifier:
    """Stands in for the model. Lets us drive stage 2 to any (shape,
    confidence) and assert what stage 3 does with it."""

    def __init__(self, shape: str, confidence: float):
        self.payload = ShapeClassification(shape=shape, confidence=confidence,
                                           reason="test")

    def complete(self, system, messages, **kw):
        from services.api.agent.llm import LLMResponse
        return LLMResponse(text=self.payload.model_dump_json(),
                           provider="fake", model="fake")


# ── Stage 1: rules pre-empt the model ────────────────────────────────────────

@pytest.mark.parametrize("q", [
    "what is the long-tail quota",
    "does this breach the diversity floor",
    "show me POL-LT-01",
    "what is our policy on markdown",
])
def test_policy_queries_always_reach_corpus_c_without_a_model(q):
    """Must be 1.0 accurate. A classifier is not, and a policy question routed
    anywhere else means the critic reasons from no rules at all."""
    d = route(q, provider=FakeClassifier("visual_semantic", 0.99))
    assert d.corpus == "C" and d.strategy == "load_full_corpus"
    assert d.decided_by == "rule:policy_lexicon" and d.confidence == 1.0


def test_article_code_short_circuits_to_a_node_lookup():
    d = route("something like 0663713001", provider=FakeClassifier("policy", 0.99))
    assert d.decided_by == "rule:article_code" and d.corpus == "A"


def test_run_reference_routes_to_eval_artefacts():
    d = route("why did coverage drop between run 12 and run 14",
              provider=FakeClassifier("policy", 0.99))
    assert d.decided_by == "rule:run_ref" and d.corpus == "B"


# ── Stage 3: code owns the mapping ───────────────────────────────────────────

@pytest.mark.parametrize("shape,expected", list(SHAPE_TABLE.items()))
def test_every_shape_maps_through_the_frozen_table(shape, expected):
    strategy, corpus = expected
    ctx = BriefContext(allow_external=True)
    d = route("neutral phrasing", ctx, provider=FakeClassifier(shape, 0.95))
    assert (d.strategy, d.corpus) == (strategy, corpus)


def test_shape_classification_schema_has_no_strategy_field():
    """The model is structurally unable to choose a strategy. If this field
    ever appears, routing has become a model decision and
    retrieval_routing_accuracy stops being a classification metric."""
    assert "strategy" not in ShapeClassification.model_fields
    assert set(ShapeClassification.model_fields) == {"shape", "confidence", "reason"}


# ── Low confidence and default-deny ──────────────────────────────────────────

def test_low_confidence_fans_out_rather_than_guessing():
    d = route("ambiguous", provider=FakeClassifier("taxonomic", 0.2))
    assert d.strategy == "fanout_rrf"
    assert d.decided_by == "fallback:low_confidence"
    assert len(d.fanout) > 1


def test_classifier_failure_degrades_to_fanout_not_a_crash():
    """A router that dies on a model error takes the whole run with it.
    Degrading to a broader search is the correct failure mode."""
    class Broken:
        def complete(self, *a, **k):
            raise RuntimeError("model down")
    d = route("anything", provider=Broken())
    assert d.strategy == "fanout_rrf" and d.confidence == 0.0


def test_corpus_d_is_default_deny():
    """The cheapest defence against injection in crawled content is not
    retrieving crawled content."""
    d = route("what is trending", provider=FakeClassifier("market_context", 0.99))
    assert d.strategy == "refuse" and d.corpus is None


def test_corpus_d_reachable_only_on_explicit_opt_in():
    d = route("what is trending", BriefContext(allow_external=True),
              provider=FakeClassifier("market_context", 0.99))
    assert d.corpus == "D" and d.strategy == "hybrid_rerank_untrusted"


# ── Evaluation support ───────────────────────────────────────────────────────

def test_rule_fired_routes_are_flagged_separately_from_model_routes():
    """Free wins must not inflate the reported classifier accuracy."""
    assert route("what is the quota", provider=FakeClassifier("policy", 0.9)).by_rule
    assert not route("neutral", BriefContext(allow_external=True),
                     provider=FakeClassifier("taxonomic", 0.9)).by_rule


def test_decision_serialises_for_the_run_record():
    """state.route_decisions must be persistable — retrieval_routing_accuracy
    is unmeasurable unless the run records what it decided."""
    d = route("what is the quota", provider=FakeClassifier("policy", 0.9))
    payload = d.as_dict()
    assert set(payload) >= {"shape", "strategy", "corpus", "decided_by", "confidence"}
