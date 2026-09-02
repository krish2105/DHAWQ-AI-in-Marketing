"""Property tests for the slot optimiser.

ARCHITECTURE.md §7.1 requires 100% of decision paths in the deterministic core
covered. This is the most consequential function in that core: it is what the
agent CALLS, and every constraint it enforces is a numbered corpus C rule that
a critic rejection will cite. A bug here produces a slate that looks compliant,
cites a policy id, and is not.
"""

from __future__ import annotations

import numpy as np
import pytest

from services.api.marketing.slots import (
    Candidate, max_per_attribute, optimise_slots, required_long_tail_slots,
)


def make(n=40, tail_from=20, types=("tee", "dress", "shoe", "bag"),
         colours=("black", "blue", "red", "green")):
    return [
        Candidate(
            article_id=f"A{i:04d}", score=1.0 - i / 100,
            price=10.0 + (i % 7), product_type=types[i % len(types)],
            colour_group=colours[i % len(colours)], is_long_tail=i >= tail_from,
        )
        for i in range(n)
    ]


# ── quota arithmetic (POL-LT-01 / POL-LT-02) ─────────────────────────────────

@pytest.mark.parametrize("k,expected", [(12, 3), (10, 2), (5, 1), (20, 4), (24, 5)])
def test_quota_rounds_up(k, expected):
    assert required_long_tail_slots(k, 0.20, 5) == expected


@pytest.mark.parametrize("k", [4])
def test_quota_waived_on_short_slates(k):
    """POL-LT-02. Without the waiver a 4-slot module owes a full slot — 25% —
    a stricter quota than the 12-slot page it sits on."""
    assert required_long_tail_slots(k, 0.20, 5) == 0


def test_attribute_cap_has_a_floor():
    """POL-DIV-02/03. Without the floor, k=4 gives a cap of 1 and the optimiser
    cannot fill a page from a narrow catalogue."""
    assert max_per_attribute(4, 4, 2) == 2
    assert max_per_attribute(12, 4, 2) == 3


# ── core invariants ──────────────────────────────────────────────────────────

def test_never_returns_more_than_k():
    for k in (4, 8, 12, 24):
        slate, _ = optimise_slots(make(), k)
        assert len(slate) <= k


def test_never_returns_duplicates():
    """POL-SLT-02."""
    slate, _ = optimise_slots(make(), 12)
    assert len(slate) == len(set(slate))


def test_meets_the_long_tail_quota_when_supply_allows():
    """POL-LT-01, the rule most likely to conflict with a revenue brief."""
    cands = make(n=60, tail_from=20)
    slate, rep = optimise_slots(cands, 12)
    tail = {c.article_id for c in cands if c.is_long_tail}
    assert sum(1 for a in slate if a in tail) >= rep.long_tail_required
    assert "POL-LT-01" not in rep.binding_constraints


def test_respects_product_type_cap():
    """POL-DIV-02. All-one-type input must not produce a 12-slot monoculture."""
    cands = [
        Candidate(f"A{i}", 1.0 - i / 100, 10.0, "tee", f"c{i % 4}", is_long_tail=i >= 20)
        for i in range(40)
    ]
    slate, _ = optimise_slots(cands, 12)
    assert len(slate) <= max_per_attribute(12, 4, 2)


def test_availability_precedes_quota():
    """Corpus C precedence: AVL before LT. An unavailable article must never be
    used to satisfy the tail quota — POL-LT-04 exists precisely so the cheapest
    way to comply is not to dump dead stock."""
    cands = [
        Candidate(f"H{i}", 1.0 - i / 100, 10.0, "tee", "black", is_long_tail=False)
        for i in range(20)
    ] + [
        Candidate(f"T{i}", 0.5, 10.0, "dress", "blue", is_long_tail=True, available=False)
        for i in range(20)
    ]
    slate, rep = optimise_slots(cands, 12)
    assert not any(a.startswith("T") for a in slate)
    assert "POL-LT-01" in rep.binding_constraints, "unmet quota must be reported"


def test_out_of_season_articles_excluded():
    """POL-AVL-03."""
    cands = make(n=40)
    cands = [Candidate(**{**c.__dict__, "season_ok": c.article_id != "A0000"})
             for c in cands]
    slate, _ = optimise_slots(cands, 12)
    assert "A0000" not in slate


# ── POL-SLT-03: never pad ────────────────────────────────────────────────────

def test_underfill_is_reported_never_padded():
    """THE rule that keeps the constraint layer honest. A quietly padded slate
    is indistinguishable from a compliant one, which means the quota was never
    enforced — only reported as enforced."""
    slate, rep = optimise_slots(make(n=5, tail_from=2), 12)
    assert len(slate) < 12
    assert rep.underfilled is True
    assert rep.binding_constraints, "under-fill must name what bound"


def test_size_bounds_rejected_cleanly():
    """POL-SLT-01."""
    for k in (0, 3, 25, 100):
        slate, rep = optimise_slots(make(), k)
        assert slate == []
        assert "POL-SLT-01" in rep.binding_constraints


# ── POL-SLT-05: determinism ──────────────────────────────────────────────────

def test_identical_input_gives_identical_slate():
    """§10.4 measures stability. Unbounded non-determinism is a defect, and a
    fully specified tie-break is what makes the stability number mean
    something rather than measure sort internals."""
    runs = [optimise_slots(make(), 12)[0] for _ in range(5)]
    assert all(r == runs[0] for r in runs)


def test_tie_break_prefers_long_tail_then_lower_price():
    """POL-SLT-05, exact order."""
    # Isolated: HEAD and TAIL share the top score and each sits in its own
    # product type, so neither is displaced by a diversity cap. Filler scores
    # strictly below both so the only thing separating HEAD and TAIL is the
    # tie-break itself.
    cands = [
        Candidate("HEAD", 0.9, 5.0, "outerwear", "black", is_long_tail=False),
        Candidate("TAIL", 0.9, 9.0, "knitwear", "blue", is_long_tail=True),
    ] + [
        Candidate(f"F{i:02d}", 0.5 - i / 1000, 10.0,
                  ("tee", "dress", "shoe", "bag")[i % 4],
                  ("red", "green", "grey", "navy")[i % 4], is_long_tail=i >= 10)
        for i in range(20)
    ]
    slate, _ = optimise_slots(cands, 12)
    assert "HEAD" in slate and "TAIL" in slate
    assert slate.index("TAIL") < slate.index("HEAD"), (
        "POL-SLT-05: on equal score the long-tail article takes the earlier slot"
    )


# ── diversity floor ──────────────────────────────────────────────────────────

def test_ild_breach_is_reported():
    """POL-DIV-01. Its threshold is PROVISIONAL_UNGROUNDED until D8, so the
    optimiser reports the breach rather than silently rejecting — the number
    is not yet trustworthy enough to drop a slate on."""
    cands = make(n=40)
    emb = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (40, 1))
    index = {f"A{i:04d}": i for i in range(40)}
    _, rep = optimise_slots(cands, 12, embeddings=emb, index=index, min_ild=0.35)
    assert "POL-DIV-01" in rep.binding_constraints


def test_report_names_every_rule_it_applied():
    """A rejection the merchandiser cannot trace to a policy line is not
    actionable."""
    _, rep = optimise_slots(make(), 12)
    assert all(r.startswith("POL-") for r in rep.rules_applied)
    assert "POL-LT-01" in rep.rules_applied
