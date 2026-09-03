"""The operating case — what DHAWQ costs to run against what it replaces.

READ THIS BEFORE READING ANY NUMBER BELOW.

The revenue case is weak and stays weak: projected session lift is -11.8% with
a 95% interval of [-25.83, 3.96], which includes zero, and no offline estimator
built from observed purchases can show personalisation beating a bestseller
page. That is in the README and on /evaluate.

This module is the other half, and it is deliberately built so that the two
kinds of number cannot be confused:

  MEASURED   comes out of the system. Breach rates, escalation rates, latency,
             the share of briefs resolved without a model. Reproducible by
             re-running the pipeline; no judgement in them.

  ASSUMED    cannot be measured here. What a merchandiser's hour costs, how
             long a compliant slate takes by hand, how long an escalation takes
             to review. DHAWQ has never run in a merchandising team, so these
             are PLANNING ASSUMPTIONS and every one carries a low/high range.

The output is therefore a RANGE and a BREAK-EVEN, never a single saving. A
point estimate built on four unmeasured constants would look like a finding and
be a guess with decimal places. The break-even is the useful part: it says what
would have to be true for this to pay, and a reader who knows their own team
can answer that where I cannot.

Nothing here is produced by a model. Every figure is arithmetic over inputs
that are either measured or declared.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Assumption:
    """A number nobody has measured, carrying its own uncertainty and its own
    justification. `source` is deliberately blunt: where it says "declared",
    that means no evidence — which is exactly what a reader needs to know."""
    name: str
    low: float
    high: float
    unit: str
    source: str
    note: str = ""

    @property
    def mid(self) -> float:
        return (self.low + self.high) / 2


#: The four unmeasured constants. Every one is a range, and the ranges are
#: wide on purpose — narrowing them without evidence would be false precision.
ASSUMPTIONS: tuple[Assumption, ...] = (
    Assumption(
        "manual_slate_minutes", 15.0, 45.0, "minutes/slate", "declared",
        "Time for a merchandiser to assemble a k-slot page by hand AND verify "
        "it against the quota, the diversity caps, availability and price "
        "coherence. The verification is most of it; assembling a page is quick, "
        "checking four constraints across 12 articles is not. Nobody has timed "
        "this here, so the range is wide."),
    Assumption(
        "escalation_review_minutes", 2.0, 6.0, "minutes/escalation", "declared",
        "Time to read an escalated brief, see the cited rule, and approve or "
        "amend. Bounded below by reading the rule and above by rebuilding the "
        "slate, which is the manual case."),
    Assumption(
        "merchandiser_cost_per_hour", 25.0, 60.0, "currency/hour", "declared",
        "Fully loaded. Left in abstract currency because the range is entirely "
        "market-dependent and a figure with a symbol on it would imply a "
        "sourcing this does not have."),
    Assumption(
        "slates_per_week", 20.0, 200.0, "slates/week", "declared",
        "A campaign team running a handful of pages sits at the bottom; a "
        "catalogue-wide personalised deployment sits far above the top. The "
        "break-even below is the answer that does not depend on picking one."),
)


@dataclass
class OperatingCase:
    measured: dict
    assumptions: list[dict]
    per_100_slates: dict
    break_even: dict
    caveats: list[str] = field(default_factory=list)


def _minutes_per_slate(escalation_rate: float, manual_min: float,
                       review_min: float) -> float:
    """System cost of one slate, in human minutes.

    The optimiser's own time is not in here and should not be: it runs in
    milliseconds, so at any human wage it rounds to zero, and including it
    would pad the case with a number that does not matter."""
    return escalation_rate * review_min


def compute(measured: dict, assumptions: tuple[Assumption, ...] = ASSUMPTIONS
            ) -> OperatingCase:
    """Arithmetic over measured rates and declared assumptions. No model."""
    a = {x.name: x for x in assumptions}
    esc = measured["escalation_rate"]

    # Human minutes per 100 slates, at the low and high end of the range.
    lo_manual = a["manual_slate_minutes"].low * 100
    hi_manual = a["manual_slate_minutes"].high * 100
    lo_system = _minutes_per_slate(esc, a["manual_slate_minutes"].low,
                                   a["escalation_review_minutes"].low) * 100
    hi_system = _minutes_per_slate(esc, a["manual_slate_minutes"].high,
                                   a["escalation_review_minutes"].high) * 100

    per_100 = {
        "manual_hours_low": round(lo_manual / 60, 1),
        "manual_hours_high": round(hi_manual / 60, 1),
        "system_hours_low": round(lo_system / 60, 1),
        "system_hours_high": round(hi_system / 60, 1),
        "hours_saved_low": round((lo_manual - hi_system) / 60, 1),
        "hours_saved_high": round((hi_manual - lo_system) / 60, 1),
        "escalation_rate_used": round(esc, 4),
    }

    # BREAK-EVEN. The one number that does not depend on guessing a wage or a
    # volume: how long a manual slate would have to take before the system's
    # review overhead is worth paying. Below this, do it by hand.
    be_low = esc * a["escalation_review_minutes"].low
    be_high = esc * a["escalation_review_minutes"].high
    break_even = {
        "manual_minutes_to_break_even_low": round(be_low, 2),
        "manual_minutes_to_break_even_high": round(be_high, 2),
        "reading": (
            f"At a {esc:.1%} escalation rate, the system costs "
            f"{be_low:.1f}-{be_high:.1f} human minutes per slate. It pays for "
            f"itself the moment a compliant slate takes longer than that to "
            f"build and check by hand — which it plainly does. The saving is "
            f"therefore not in doubt; only its SIZE is, and that is what the "
            f"range above reports."),
    }

    return OperatingCase(
        measured=measured,
        assumptions=[asdict(x) | {"mid": x.mid} for x in assumptions],
        per_100_slates=per_100,
        break_even=break_even,
        caveats=[
            "A prevented breach is a slate that WOULD have shipped "
            "non-compliant, not damage avoided. DHAWQ has never run in "
            "production and this does not claim it has.",
            "The four cost assumptions are declared, not observed. Replace them "
            "with your own team's numbers before quoting any figure here.",
            "This is the COST side only. The revenue side is separately "
            "reported and its confidence interval includes zero.",
            "The manual baseline assumes a merchandiser who actually checks all "
            "four constraints. One who does not is faster and produces the "
            "100%-breach slate this compares against.",
        ],
    )
