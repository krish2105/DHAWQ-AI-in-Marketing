"""Calibration — ARCHITECTURE.md §10.3, "the senior signal".

"Accuracy tells you how often the system is right. Calibration tells you
whether its confidence means anything. A system that is 70% accurate and knows
it is more useful than one that is 85% accurate and always says 99%."

Pure functions. Nothing here knows what a brief is.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Bin:
    lo: float
    hi: float
    n: int
    mean_confidence: float
    observed_accuracy: float

    @property
    def gap(self) -> float:
        """Positive = overconfident. This is the number that matters."""
        return self.mean_confidence - self.observed_accuracy


def brier_score(confidences: list[float], outcomes: list[bool]) -> float:
    """Mean squared error between stated confidence and what happened.

    0 is perfect. 0.25 is what you get by always saying 0.5. Above that the
    confidence is worse than useless — it is actively misleading.
    """
    if not confidences:
        return 0.0
    return sum((c - float(o)) ** 2 for c, o in zip(confidences, outcomes)) / len(confidences)


def reliability_curve(confidences: list[float], outcomes: list[bool],
                      n_bins: int = 5) -> list[Bin]:
    """Bucket by stated confidence, compare to observed accuracy per bucket.

    Empty bins are DROPPED rather than reported as 0% accurate — a bin nobody
    landed in says nothing about calibration, and showing it as a failure would
    invent evidence.
    """
    out: list[Bin] = []
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        idx = [j for j, c in enumerate(confidences)
               if (lo <= c < hi) or (i == n_bins - 1 and c == 1.0)]
        if not idx:
            continue
        out.append(Bin(
            lo=lo, hi=hi, n=len(idx),
            mean_confidence=sum(confidences[j] for j in idx) / len(idx),
            observed_accuracy=sum(outcomes[j] for j in idx) / len(idx),
        ))
    return out


def expected_calibration_error(bins: list[Bin]) -> float:
    """ECE — average |confidence − accuracy|, weighted by bin population."""
    total = sum(b.n for b in bins)
    if total == 0:
        return 0.0
    return sum(b.n * abs(b.gap) for b in bins) / total


def overconfidence(bins: list[Bin]) -> float:
    """Signed ECE. Positive means the system systematically overstates itself,
    which is the direction §10.3 says to act on: suppress confidence rather
    than inflate the accuracy claim."""
    total = sum(b.n for b in bins)
    if total == 0:
        return 0.0
    return sum(b.n * b.gap for b in bins) / total
