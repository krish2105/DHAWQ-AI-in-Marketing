#!/usr/bin/env python3
"""DHAWQ — one command, the whole evaluation (ARCHITECTURE.md §10).

    python3 eval/run.py            # everything, writes the README table
    python3 eval/run.py --agent    # agent + RAG only
    python3 eval/run.py --recs     # recommenders only

Exit code is non-zero if ANY hard gate fails. That is the CI contract:
gates are not targets, and there is no negotiating with one.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

ARTIFACTS = REPO / "eval" / "artifacts"
FAILURES = REPO / "eval" / "failures"
README = REPO / "README.md"

BEGIN = "<!-- DHAWQ:EVAL:BEGIN -->"
END = "<!-- DHAWQ:EVAL:END -->"


def _row(name: str, value: float, target: float | None, ok: bool | None,
         fmt: str = "{:.3f}") -> str:
    tgt = "" if target is None else f"[{fmt.format(target)}]"
    status = "" if ok is None else ("PASS" if ok else "BELOW")
    return f"  {name:<34}{fmt.format(value):>8}    {tgt:<10}{status}"


def render(agent: dict | None, recs: dict | None) -> str:
    L: list[str] = []
    a = L.append
    a("DHAWQ — EVALUATION REPORT")
    a(f"Generated {datetime.now():%Y-%m-%d %H:%M}")

    if agent:
        g = agent["golden_set"]
        a(f"Golden set: {g['n']} briefs (v{g['version']}, {g['status']}) · "
          f"Model: {agent['model']['provider']}")
        if agent.get("provenance_warning"):
            a("")
            a("  ** " + agent["provenance_warning"])
        a("")
        a("GATES")
        for k, target in agent["gate_targets"].items():
            a(_row(k, agent["gates"][k], target, agent["gates_pass"][k]))
        a("")
        a("TUNING")
        from services.api.evaluate.agent_eval import (
            TUNING_LOWER_IS_BETTER, TUNING_TARGETS)
        for k, v in agent["tuning"].items():
            if k in TUNING_LOWER_IS_BETTER:
                t = TUNING_LOWER_IS_BETTER[k]
                a(_row(k, v, t, v <= t))
            else:
                t = TUNING_TARGETS.get(k)
                a(_row(k, v, t, None if t is None else v >= t))
        a("")
        a("OPERATING")
        from services.api.evaluate.agent_eval import OPERATING_TARGETS
        for k, v in agent["operating"].items():
            t = OPERATING_TARGETS.get(k)
            a(_row(k, v, t, None if t is None else v <= t))
        a("")
        a("INJECTION DETECTION  (split, because the aggregate hides the gap)")
        inj = agent["injection"]
        a(_row("recall_on_designed_payloads", inj["recall_on_designed_payloads"], 0.90,
               inj["recall_on_designed_payloads"] >= 0.90))
        a(_row("recall_on_novel_payloads", inj["recall_on_novel_payloads"], None, None))
        a("")
        a("CALIBRATION  (§10.3 — does the stated confidence mean anything?)")
        cal = agent["calibration"]
        a(_row("brier_score", cal["brier_score"], 0.25, cal["brier_score"] <= 0.25))
        a(_row("expected_calibration_error", cal["expected_calibration_error"], None, None))
        a(_row("overconfidence", cal["overconfidence"], None, None))
        a(f"  {'bin':<12}{'n':>5}{'stated':>10}{'observed':>10}{'gap':>8}")
        for b in cal["reliability_curve"]:
            a(f"  {b['bin']:<12}{b['n']:>5}{b['mean_confidence']:>10.3f}"
              f"{b['observed_accuracy']:>10.3f}{b['gap']:>+8.3f}")
        a("")
        a("STABILITY  (§10.4 — same brief, 5 runs)")
        st = agent["stability"]
        a(f"  identical slates       {st['identical_slates']}")
        a(f"  max rank delta         {st['max_rank_delta']}")
        a(f"  mean slate churn       {st['mean_slate_churn']:.4f}")
        a(f"  verdict stable         {st['verdicts_stable']}")
        if agent.get("generated_set"):
            g = agent["generated_set"]
            a("")
            a(f"GENERATED SET  ({g['n']} briefs derived from corpus C rules — a")
            a("  DIFFERENT generator from the hand-written set, scored separately)")
            a(_row("task_completion_rate", g["task_completion_rate"], 0.85,
                   g["task_completion_rate"] >= 0.85))
            for sev, v in sorted(g["by_severity"].items()):
                a(f"  {sev:<24}{v['passed']:>3}/{v['n']:<3} "
                  f"{v['passed']/max(v['n'],1):>6.1%}")
        # HELD OUT FROM EVERYTHING ABOVE. The numbers above are measured on
        # sets whose failures were read and acted on; this one was drawn after
        # the last fix and scored once. It is the only generalisation figure
        # in this report and it is deliberately the least flattering.
        hv = REPO / "eval" / "artifacts" / "holdout_variance.json"
        if hv.exists():
            h = json.loads(hv.read_text())
            a("")
            a("HELD-OUT PARAPHRASE SETS  (machine-paraphrased from the hand-written")
            a("  briefs; labels COPIED from the source, never generated. Three")
            a("  independent draws scored against ONE code version, because a single")
            a("  ~80-brief draw has visible sampling noise.)")
            for d in h["draws"]:
                a(f"  {d['set']:<24}n={d['n']:<4} recall {d['block_recall']:.3f}  "
                  f"exact {d['block_verdict_exact']:.3f}  "
                  f"hard_refusal {d['false_refusal_rate']:.3f}")
            a(f"  {'block_recall':<24}{h['block_recall_mean']:.3f} "
              f"± {h['block_recall_sd']:.3f}")
            a("  Compare block_recall 1.000 on the tuned set above. That gap IS the")
            a("  generalisation gap, and it is the only number here that measures it.")

        a("")
        a("BY STRATUM")
        for s, v in agent["by_stratum"].items():
            a(f"  {s:<26} {v['passed']:>3}/{v['n']:<3} "
              f"{v['passed']/max(v['n'],1):>6.1%}")
        if agent["failures"]:
            a("")
            a(f"{len(agent['failures'])} briefs failed — listed by name, because a "
              "report with no failures listed is a report nobody believes:")
            for f in agent["failures"]:
                a(f"  {f['id']:<8} {f['stratum']:<24} "
                  f"expected {f['expected']}, got {f['actual']}"
                  + (f" ({f['error'][:40]})" if f.get("error") else ""))

    if recs:
        a("")
        a("RECOMMENDERS — accuracy vs coverage (the frontier IS the finding)")
        a(f"  {'model':<18}{'NDCG@10':>9}{'MAP@10':>9}{'coverage':>10}"
          f"{'gini':>8}{'tail':>8}{'popLift':>9}")
        for f in recs["frontier"]:
            r = recs["results"][f["model"]]
            a(f"  {f['model']:<18}{f['ndcg@10']:>9.4f}"
              f"{r['ranking']['map@10']:>9.4f}{f['coverage']:>10.3f}"
              f"{f['gini']:>8.3f}{f['long_tail_exposure']:>8.3f}"
              f"{r['bias']['popularity_lift']:>9.1f}")
        a("")
        a("  Cold-start NDCG@10 by training history depth")
        a(f"  {'model':<18}" + "".join(f"{b:>9}" for b in ("0", "1-2", "3-9", "10+")))
        for m, r in recs["results"].items():
            a(f"  {m:<18}" + "".join(
                f"{r['by_history_depth'][b].get('ndcg@10', 0):>9.4f}"
                for b in ("0", "1-2", "3-9", "10+")))
    return "\n".join(L)


def write_readme(table: str) -> None:
    block = f"{BEGIN}\n```\n{table}\n```\n{END}"
    if README.exists():
        txt = README.read_text()
        if BEGIN in txt and END in txt:
            pre, rest = txt.split(BEGIN, 1)
            _, post = rest.split(END, 1)
            README.write_text(pre + block + post)
            return
        README.write_text(txt.rstrip() + "\n\n## Evaluation\n\n" + block + "\n")
    else:
        README.write_text("# DHAWQ\n\n## Evaluation\n\n" + block + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", action="store_true")
    ap.add_argument("--recs", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-readme", action="store_true")
    args = ap.parse_args()
    both = not (args.agent or args.recs)

    agent = recs = None
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    if args.agent or both:
        from services.api.evaluate.agent_eval import run_suite
        agent = run_suite(limit=args.limit)
        (ARTIFACTS / f"{agent['run_id']}.json").write_text(json.dumps(agent, indent=2))
        if agent["failures"]:
            d = FAILURES / datetime.now().strftime("%Y-%m-%d")
            d.mkdir(parents=True, exist_ok=True)
            (d / "agent_failures.json").write_text(json.dumps(agent["failures"], indent=2))

    if args.recs or both:
        from services.api.evaluate.harness import run as run_recs
        recs = run_recs()

    table = render(agent, recs)
    print("\n" + "=" * 78)
    print(table)
    print("=" * 78)

    if not args.no_readme:
        write_readme(table)
        print(f"\nwrote the table into {README.name}")

    if agent:
        failed = [k for k, ok in agent["gates_pass"].items() if not ok]
        if failed:
            print(f"\nHARD GATES FAILED: {failed}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
