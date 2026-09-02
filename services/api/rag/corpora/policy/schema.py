"""Typed shape of corpus C.

A malformed policy fails here, at load, with a line-level error — not in the
critic at run time with a KeyError that looks like an agent bug.

This module is deliberately dependency-light (pydantic + pyyaml). It is
imported by the critic, by the slot optimiser and by render.py, so it must not
pull in the rest of the service.
"""

from __future__ import annotations

import re
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

POLICY_DIR = Path(__file__).parent
POLICY_YAML = POLICY_DIR / "policy.yaml"

RULE_ID_RE = re.compile(r"^POL-([A-Z]{2,4})-(\d{2})$")


class Severity(str, Enum):
    """What the critic does when the rule is breached."""

    HARD = "hard"          # slate rejected, dropped, never downgraded
    ESCALATE = "escalate"  # human gate, agent may not resolve it
    SOFT = "soft"          # proceeds with a recorded, rendered warning
    ADVISORY = "advisory"  # not machine-checked; never grounds for rejection


class CalibrationStatus(str, Enum):
    SETTLED = "settled"
    PROVISIONAL = "provisional"
    #  Screaming case is intentional: these are the numbers with no empirical
    #  basis at all yet. They should be visible when skimming the file.
    PROVISIONAL_UNGROUNDED = "PROVISIONAL_UNGROUNDED"


class Scope(str, Enum):
    ARTICLE = "article"
    SLATE = "slate"
    CAMPAIGN = "campaign"
    COHORT = "cohort"
    EXPLANATION = "explanation"
    RUN = "run"
    SYSTEM = "system"


class Check(BaseModel):
    """The binding between a policy rule and the function that evaluates it.

    `fn` is None only for advisory rules. It is a *name*, not a callable — the
    policy does not import executable code, and code does not live in YAML.
    """

    model_config = ConfigDict(extra="forbid")

    fn: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    overrides: str | None = None


class Calibration(BaseModel):
    """Provenance of every threshold: where it came from, when it is revisited.

    Present on every rule so that "which numbers are invented?" is a query
    against the policy rather than a question for its author.
    """

    model_config = ConfigDict(extra="forbid")

    status: CalibrationStatus
    revisit_at: str | None = None
    depends_on: list[str] = Field(default_factory=list)
    note: str = ""


class Rule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    severity: Severity
    critic_criterion: int | None = None
    scope: Scope
    statement: str
    check: Check
    rationale: str
    limitation: str | None = None
    calibration: Calibration

    @model_validator(mode="after")
    def _validate(self) -> "Rule":
        if not RULE_ID_RE.match(self.id):
            raise ValueError(f"{self.id!r} is not a well-formed rule id (POL-XX-NN)")

        if self.critic_criterion is not None and not 1 <= self.critic_criterion <= 9:
            raise ValueError(
                f"{self.id}: critic_criterion {self.critic_criterion} is outside "
                "the nine criteria in ARCHITECTURE.md §7.6"
            )

        # An advisory rule with a checker would be silently enforced; a
        # non-advisory rule without one is a rule that only looks enforced.
        if self.severity is Severity.ADVISORY and self.check.fn is not None:
            raise ValueError(f"{self.id}: advisory rules must not declare a check fn")
        if self.severity is not Severity.ADVISORY and self.check.fn is None:
            raise ValueError(f"{self.id}: non-advisory rules must declare a check fn")

        if self.calibration.status is not CalibrationStatus.SETTLED:
            if not self.calibration.revisit_at:
                raise ValueError(f"{self.id}: unsettled calibration needs revisit_at")

        return self

    @property
    def domain(self) -> str:
        m = RULE_ID_RE.match(self.id)
        assert m is not None  # guaranteed by _validate
        return m.group(1)


class Section(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    preamble: str
    rules: list[Rule]

    @model_validator(mode="after")
    def _rules_match_domain(self) -> "Section":
        for rule in self.rules:
            if rule.domain != self.id:
                raise ValueError(
                    f"{rule.id} is filed under section {self.id!r} but its id "
                    f"declares domain {rule.domain!r}"
                )
        return self


class Authority(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["non_authoritative"]
    statement: str


class Definition(BaseModel):
    model_config = ConfigDict(extra="allow")

    term: str
    definition: str


class Policy(BaseModel):
    """Corpus C, whole. There is no partial load."""

    model_config = ConfigDict(extra="forbid")

    policy_version: str
    effective_from: str
    authored_by: str
    supersedes: str | None
    authority: Authority
    severity_levels: dict[str, str]
    precedence: list[str]
    definitions: dict[str, Definition]
    sections: list[Section]

    # ── derived views ────────────────────────────────────────────────────────

    @property
    def rules(self) -> list[Rule]:
        return [r for s in self.sections for r in s.rules]

    def by_id(self, rule_id: str) -> Rule:
        for rule in self.rules:
            if rule.id == rule_id:
                return rule
        raise KeyError(f"no such rule: {rule_id}")

    def for_criterion(self, criterion: int) -> list[Rule]:
        """Every rule the given critic criterion is evaluated against."""
        return [r for r in self.rules if r.critic_criterion == criterion]

    def unsettled(self) -> list[Rule]:
        return [
            r for r in self.rules
            if r.calibration.status is not CalibrationStatus.SETTLED
        ]

    # ── whole-policy integrity ───────────────────────────────────────────────

    @model_validator(mode="after")
    def _validate(self) -> "Policy":
        rules = self.rules
        ids = [r.id for r in rules]

        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate rule ids: {sorted(dupes)}")

        section_ids = {s.id for s in self.sections}
        missing = section_ids - set(self.precedence)
        if missing:
            raise ValueError(f"domains absent from precedence order: {sorted(missing)}")

        # Every cross-reference must resolve. A rule citing a rule that does not
        # exist is the policy-layer version of a broken citation, and the whole
        # system is built on citations resolving.
        known = set(ids)
        for rule in rules:
            refs: list[str] = []
            if rule.check.overrides:
                refs.append(rule.check.overrides)
            for key in ("rule", "rules"):
                val = rule.check.params.get(key)
                if isinstance(val, str):
                    refs.append(val)
                elif isinstance(val, list):
                    refs.extend(v for v in val if isinstance(v, str))
            for ref in refs:
                if ref.startswith("POL-") and ref not in known:
                    raise ValueError(f"{rule.id} references unknown rule {ref}")

        return self


def load_policy(path: Path | None = None) -> Policy:
    """Load corpus C whole.

    There is deliberately no `load_section` or `search_policy`. See
    ARCHITECTURE.md §8.2 — the absence of a partial-load API is the design.
    """
    raw = yaml.safe_load((path or POLICY_YAML).read_text(encoding="utf-8"))
    return Policy.model_validate(raw)
