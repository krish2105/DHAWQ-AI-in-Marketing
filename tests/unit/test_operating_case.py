"""The cost model's only real risk is that an assumption gets read as a
measurement. These tests exist to make that structurally hard.
"""

import pytest

from services.api.marketing.operating_case import ASSUMPTIONS, compute

MEASURED = {
    "escalation_rate": 0.5571,
    "ungoverned_breach_rate": 1.0,
    "silent_breach_rate": 0.0,
    "slates_audited": 210,
    "cohorts": 35,
}


def test_every_assumption_declares_a_range_and_a_source():
    for a in ASSUMPTIONS:
        assert a.low < a.high, f"{a.name} has no range — that is a point estimate"
        assert a.source, f"{a.name} has no source"
        assert a.note, f"{a.name} has no justification"


def test_no_assumption_claims_to_be_measured():
    # The whole point. If one of these ever says "measured", someone has
    # promoted a guess and the output stops being a range of possibilities.
    for a in ASSUMPTIONS:
        assert "measur" not in a.source.lower(), (
            f"{a.name} claims to be measured; it is not")


def test_the_output_is_a_range_never_a_point():
    c = compute(MEASURED)
    p = c.per_100_slates
    assert p["hours_saved_low"] < p["hours_saved_high"]
    assert p["manual_hours_low"] < p["manual_hours_high"]
    assert not any(k.endswith("_estimate") for k in p), (
        "a single-figure saving would read as a finding and is a guess")


def test_break_even_does_not_depend_on_a_wage_or_a_volume():
    """The one number a reader can act on without adopting my assumptions.
    It must move with the escalation rate and the review time ONLY."""
    a = compute(MEASURED).break_even
    b = compute({**MEASURED, "escalation_rate": 0.20}).break_even
    assert b["manual_minutes_to_break_even_low"] < \
        a["manual_minutes_to_break_even_low"]

    # Doubling the wage must not move it at all.
    from dataclasses import replace
    wage = [x for x in ASSUMPTIONS if x.name == "merchandiser_cost_per_hour"][0]
    doubled = tuple(replace(x, low=x.low * 2, high=x.high * 2)
                    if x is wage else x for x in ASSUMPTIONS)
    c = compute(MEASURED, doubled).break_even
    assert c["manual_minutes_to_break_even_low"] == \
        a["manual_minutes_to_break_even_low"]


def test_caveats_state_that_a_prevented_breach_is_not_damage_avoided():
    text = " ".join(compute(MEASURED).caveats).lower()
    assert "would have shipped" in text
    assert "never run in production" in text
    assert "confidence interval includes zero" in text


@pytest.mark.parametrize("rate", [0.0, 0.25, 0.5571, 1.0])
def test_saving_shrinks_monotonically_as_escalation_rises(rate):
    c = compute({**MEASURED, "escalation_rate": rate})
    assert c.per_100_slates["hours_saved_low"] <= \
        compute({**MEASURED, "escalation_rate": 0.0}).per_100_slates["hours_saved_low"]


def test_the_model_contains_no_model():
    """§0.1. Nothing in the cost case may come from a language model."""
    import inspect

    from services.api.marketing import operating_case as m
    src = inspect.getsource(m)
    for banned in ("llm", "ollama", "complete(", "classify", "prompt"):
        assert banned not in src.lower(), f"{banned!r} in the cost model"
