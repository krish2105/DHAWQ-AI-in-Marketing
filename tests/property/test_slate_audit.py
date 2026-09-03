"""The auditor duplicates the optimiser's rules on purpose. These tests are
what stop that being an accident.

An auditor that called optimise_slots() could only ever agree with it, and
would be useless for the question it exists to answer: how often does a slate
built WITHOUT these constraints breach the policy? So the rules are written
twice. The risk of writing them twice is that they drift, and the mitigation is
here: whatever the optimiser produces, the auditor must find compliant, or one
of the two is wrong.
"""

import random

import pytest

from services.api.marketing.slate_audit import Violation, audit, revenue_ranked
from services.api.marketing.slots import Candidate, optimise_slots

TYPES = ["Trousers", "T-shirt", "Dress", "Sweater", "Jacket", "Skirt"]
COLOURS = ["Black", "Blue", "White", "Red", "Green"]


def pool(n=300, seed=0, tail_every=7, price_lo=10, price_hi=60):
    rng = random.Random(seed)
    return [
        Candidate(
            article_id=f"{i:010d}", score=1.0 - i / n,
            price=rng.uniform(price_lo, price_hi),
            product_type=rng.choice(TYPES), colour_group=rng.choice(COLOURS),
            is_long_tail=(i % tail_every == 0),
        )
        for i in range(n)
    ]


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("k", [8, 12, 20])
def test_the_optimisers_output_is_compliant_or_the_optimiser_said_why(seed, k):
    """The invariant that keeps the duplication honest.

    Either the slate passes the audit, or every rule it breaks appears in
    binding_constraints — POL-ESC-01: a breach that cannot be resolved
    escalates, it is never shipped silently.
    """
    p = pool(seed=seed)
    chosen, report = optimise_slots(p, k)
    slate = [c for c in p if c.article_id in set(chosen)]
    declared = set(report.binding_constraints)
    for v in audit(slate, k):
        assert v.rule_id in declared, (
            f"undeclared breach {v}: binding_constraints={sorted(declared)}")


def test_a_revenue_ranked_slate_breaches_and_that_is_the_whole_comparison():
    p = pool(seed=3, tail_every=11)
    v = audit(revenue_ranked(p, 12), 12)
    assert v, "if the ungoverned slate is compliant there is nothing to measure"
    assert any(x.rule_id == "POL-LT-01" for x in v)


def test_underfill_is_always_declared_not_only_when_nothing_else_bound():
    """The defect the audit found: six of 210 slates shipped underfilled with
    POL-SLT-03 absent from binding_constraints, because POL-LT-01 had already
    bound and the old guard skipped it. Silent non-compliance is the one
    failure the escalation path exists to make impossible."""
    thin = pool(n=14, seed=1, tail_every=99)          # cannot fill 12 after caps
    chosen, report = optimise_slots(thin, 12)
    if len(chosen) < 12:
        assert "POL-SLT-03" in report.binding_constraints
        assert report.underfilled


def test_audit_finds_each_rule_it_claims_to():
    k = 12
    base = pool(n=200, seed=5)

    dup = base[:11] + [base[0]]
    assert any(v.rule_id == "POL-SLT-02" for v in audit(dup, k))

    one_type = [Candidate(c.article_id, c.score, c.price, "Trousers",
                          c.colour_group, c.is_long_tail) for c in base[:k]]
    assert any(v.rule_id == "POL-DIV-02" for v in audit(one_type, k))

    one_colour = [Candidate(c.article_id, c.score, c.price, c.product_type,
                            "Black", c.is_long_tail) for c in base[:k]]
    assert any(v.rule_id == "POL-DIV-03" for v in audit(one_colour, k))

    no_tail = [Candidate(c.article_id, c.score, c.price, c.product_type,
                         c.colour_group, False) for c in base[:k]]
    assert any(v.rule_id == "POL-LT-01" for v in audit(no_tail, k))

    unavailable = [Candidate(c.article_id, c.score, c.price, c.product_type,
                             c.colour_group, c.is_long_tail, available=False)
                   for c in base[:k]]
    assert any(v.rule_id == "POL-AVL-01" for v in audit(unavailable, k))

    spread = [Candidate(c.article_id, c.score, 5.0 if i else 500.0,
                        c.product_type, c.colour_group, c.is_long_tail)
              for i, c in enumerate(base[:k])]
    assert any(v.rule_id == "POL-PRC-01" for v in audit(spread, k))


def test_short_slates_waive_the_quota_in_both_implementations():
    # POL-LT-02. If the auditor missed the waiver it would report a breach on
    # every 4-slot page and the whole measurement would be inflated.
    no_tail = [Candidate(f"{i:010d}", 1.0 - i, 20.0, TYPES[i % 6],
                         COLOURS[i % 5], False) for i in range(4)]
    assert not [v for v in audit(no_tail, 4) if v.rule_id == "POL-LT-01"]


def test_violation_carries_a_rule_id_that_can_actually_be_cited():
    from pathlib import Path
    import sys
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "services/api/rag/corpora/policy"))
    from schema import load_policy                      # noqa: E402

    known = {r.id for r in load_policy().rules}
    p = pool(seed=9, tail_every=13)
    seen: set[str] = set()
    for k in (8, 12, 24):
        seen |= {v.rule_id for v in audit(revenue_ranked(p, k), k)}
    assert seen, "no violations produced; the test is not exercising anything"
    for rid in seen:
        assert rid in known, f"audit cites {rid}, which is not in corpus C"
    assert isinstance(next(iter(seen)), str)
    assert Violation("POL-LT-01", "x").rule_id == "POL-LT-01"
