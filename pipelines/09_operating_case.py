"""D22 — the cost case, measured where it can be and declared where it cannot.

WHY THIS EXISTS. The revenue case for DHAWQ is honestly weak: the headline
session-lift estimate is negative with a confidence interval spanning zero, and
no offline estimator built from observed purchases can show personalisation
beating a bestseller page. That is reported and not going to improve without a
live A/B test.

The COST case does not need one. Two things are measurable from the system as
it stands:

  1. HOW OFTEN AN UNGOVERNED SLATE BREACHES THE POLICY. Build the slate a
     revenue-ranked module produces — top k by score, no constraints — and
     audit it against corpus C. Then audit the optimiser's slate for the same
     cohort. The difference is breaches prevented, per rule, and it is a
     measurement with no assumptions in it at all.

  2. HOW MUCH OF THE BRIEF QUEUE NEVER REACHES A HUMAN OR A MODEL. Measured in
     eval/artifacts/triage_cost.json.

Everything after that — what a merchandiser's hour costs, how long a compliant
slate takes to build by hand, what a policy breach costs when it ships — is an
ASSUMPTION. Those live in operating_case.py, are labelled as assumptions, are
reported with a sensitivity range rather than a point estimate, and are never
mixed into the measured numbers.

OUT  data/processed/cohorts/operating_case.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from services.api.core.artifacts import (                          # noqa: E402
    articles, catalogue_facts, cohort_candidates)
from services.api.marketing.slate_audit import audit, revenue_ranked  # noqa: E402
from services.api.marketing.slots import Candidate, optimise_slots    # noqa: E402

#: Slate sizes a merchandiser actually asks for. Not a sweep for its own sake:
#: the long-tail quota rounds up, so the constraint bites differently at each
#: size and a single k would hide that.
SLATE_SIZES = (8, 10, 12, 16, 20, 24)


def build_candidates(ids: list[str], meta: dict, prices: dict,
                     head: set) -> list[Candidate]:
    return [
        Candidate(
            article_id=a, score=1.0 - i / max(len(ids), 1),
            price=float(prices.get(a, 0.0) or 0.0),
            product_type=(meta.get(a) or (a, "unknown", "unknown"))[1] or "unknown",
            colour_group=(meta.get(a) or (a, "unknown", "unknown"))[2] or "unknown",
            is_long_tail=a not in head,
        )
        for i, a in enumerate(ids) if a in meta
    ]


def main() -> None:
    facts = catalogue_facts()
    head, prices = set(facts["head"]), facts["prices"]
    meta = {r[0]: r for r in articles().select(
        "article_id", "product_type_name", "colour_group_name").iter_rows()}

    cands = cohort_candidates()
    rows, ungoverned_rules, governed_rules = [], Counter(), Counter()
    silent_rules: Counter = Counter()

    for model, segments in cands.items():
        for segment, ids in segments.items():
            pool = build_candidates(ids, meta, prices, head)
            if len(pool) < max(SLATE_SIZES):
                continue
            for k in SLATE_SIZES:
                rr = revenue_ranked(pool, k)
                v_un = audit(rr, k)

                chosen, rep = optimise_slots(pool, k)
                keep = set(chosen)
                opt = [c for c in pool if c.article_id in keep]
                v_go = audit(opt, k)

                # THE SAME SPLIT THAT MATTERED FOR REFUSALS MATTERS HERE.
                # A slate that breaches and SAYS SO reaches a human under
                # POL-ESC-01 and never ships; a slate that breaches silently
                # is the actual defect. Counting them together made the
                # governed rate look like a 56% failure when most of it is
                # the escalation path working exactly as corpus C specifies.
                flagged = set(rep.binding_constraints)
                silent = [v for v in v_go if v.rule_id not in flagged]

                for v in v_un:
                    ungoverned_rules[v.rule_id] += 1
                for v in v_go:
                    governed_rules[v.rule_id] += 1
                for v in silent:
                    silent_rules[v.rule_id] += 1

                rows.append({
                    "model": model, "segment": segment, "k": k,
                    "ungoverned_violations": len(v_un),
                    "governed_violations": len(v_go),
                    "ungoverned_rules": sorted({v.rule_id for v in v_un}),
                    "governed_rules": sorted({v.rule_id for v in v_go}),
                    "silent_violations": len(silent),
                    "escalated": bool(v_go) and not silent,
                })

    # PER-MODEL, because the aggregate hides the most useful thing here.
    # Escalation load varies 40x across the five arms, and the cause is the
    # TAIL SHARE OF THE COHORT POOL — not catalogue coverage, which is measured
    # across all users and can be high while every individual cohort's
    # candidate list is head-heavy. hybrid_cascade has the best coverage on the
    # frontier plot (0.655) and 3.1% tail per cohort; hybrid_weighted has worse
    # coverage (0.468) and 36.9%. Only the second determines whether a page can
    # satisfy POL-LT-01 without a human.
    by_model: dict[str, dict] = {}
    for r in rows:
        b = by_model.setdefault(r["model"], {"n": 0, "escalated": 0})
        b["n"] += 1
        b["escalated"] += bool(r["governed_violations"])
    for m, b in by_model.items():
        pools = cands[m]
        shares = [sum(1 for a in ids if a not in head) / max(len(ids), 1)
                  for ids in pools.values()]
        b["escalation_rate"] = round(b["escalated"] / b["n"], 4)
        b["cohort_pool_tail_share"] = round(sum(shares) / len(shares), 4)

    by_k: dict[int, dict] = {}
    for r in rows:
        b = by_k.setdefault(r["k"], {"n": 0, "escalated": 0})
        b["n"] += 1
        b["escalated"] += bool(r["governed_violations"])
    for b in by_k.values():
        b["escalation_rate"] = round(b["escalated"] / b["n"], 4)

    n = len(rows)
    un_any = sum(1 for r in rows if r["ungoverned_violations"])
    go_any = sum(1 for r in rows if r["governed_violations"])
    silent_any = sum(1 for r in rows if r["silent_violations"])
    escalated = sum(1 for r in rows if r["escalated"])

    out = {
        "measured": {
            "slates_audited": n,
            "cohorts": len({(r["model"], r["segment"]) for r in rows}),
            "slate_sizes": list(SLATE_SIZES),
            "ungoverned_breach_rate": round(un_any / max(n, 1), 4),
            "governed_breach_rate": round(go_any / max(n, 1), 4),
            "governed_escalated_rate": round(escalated / max(n, 1), 4),
            "governed_silent_breach_rate": round(silent_any / max(n, 1), 4),
            "breaches_prevented": un_any - go_any,
            "breaches_prevented_or_escalated": un_any - silent_any,
            "mean_violations_per_ungoverned_slate": round(
                sum(r["ungoverned_violations"] for r in rows) / max(n, 1), 3),
            "ungoverned_by_rule": dict(ungoverned_rules.most_common()),
            "governed_by_rule": dict(governed_rules.most_common()),
            "silent_by_rule": dict(silent_rules.most_common()),
            "by_model": by_model,
            "by_slate_size": {str(k): v for k, v in sorted(by_k.items())},
        },
        "method": (
            "Ungoverned = top-k by score, which is exactly what a revenue-ranked "
            "module produces and exactly what the optimiser starts from. Governed "
            "= the same candidate pool through optimise_slots. Both audited by "
            "slate_audit.audit(), which reimplements corpus C against slate "
            "CONTENTS rather than reading the optimiser's own report — an auditor "
            "that called the optimiser could only ever agree with it."),
        "caveat": (
            "A breach here is a slate that WOULD have shipped non-compliant, not "
            "a breach that did ship. DHAWQ has never run in production, so this "
            "measures the failure rate of the alternative, not damage avoided."),
        "rows": rows,
    }

    dst = REPO / "data" / "processed" / "cohorts" / "operating_case.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2))

    m = out["measured"]
    print(f"slates audited        {m['slates_audited']} "
          f"({m['cohorts']} cohorts x {len(SLATE_SIZES)} sizes)")
    print(f"ungoverned breach     {m['ungoverned_breach_rate']:.1%}")
    print(f"governed breach       {m['governed_breach_rate']:.1%}"
          f"  (of which escalated: {m['governed_escalated_rate']:.1%})")
    print(f"governed SILENT       {m['governed_silent_breach_rate']:.1%}"
          f"  <- the only real defect rate")
    print(f"prevented or caught   {m['breaches_prevented_or_escalated']}/{n}")
    print(f"mean violations/slate {m['mean_violations_per_ungoverned_slate']}")
    print("\nungoverned, by rule:")
    for r, c in m["ungoverned_by_rule"].items():
        print(f"  {r:<14}{c:>5}")
    if m["governed_by_rule"]:
        print("\nafter optimisation, breached AND reported (escalates to a human):")
        for r, c in m["governed_by_rule"].items():
            print(f"  {r:<14}{c:>5}")
    print("\nbreached SILENTLY (would ship non-compliant — the real defect):")
    print("  none" if not m["silent_by_rule"] else "")
    for r, c in m["silent_by_rule"].items():
        print(f"  {r:<14}{c:>5}")
    print("\nescalation load by recommender (and why):")
    print(f"  {'model':<20}{'escalates':>10}{'cohort pool tail':>19}")
    for m, b in sorted(by_model.items(), key=lambda kv: kv[1]["escalation_rate"]):
        print(f"  {m:<20}{b['escalation_rate']:>9.1%}{b['cohort_pool_tail_share']:>19.1%}")
    print("  Catalogue coverage is measured across ALL users; this is tail")
    print("  availability WITHIN one cohort's candidate list. They are not the")
    print("  same number and only the second decides whether POL-LT-01 can be")
    print("  met without a human.")
    print(f"\nwrote {dst.relative_to(REPO)}")


if __name__ == "__main__":
    main()
