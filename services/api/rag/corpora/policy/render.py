"""Generate POLICY.md and manifest.json from policy.yaml.

POLICY.md is the document that loads whole into the critic's context. It is
generated rather than hand-written so that a threshold can never differ between
what the code enforces and what the model reads. That divergence — the engine
saying 0.20 while the document says 0.25 — is the classic way a policy layer
and its documentation rot apart, and generating one from the other removes it
by construction rather than by discipline.

    python3 services/api/rag/corpora/policy/render.py          # write
    python3 services/api/rag/corpora/policy/render.py --check  # verify in sync

--check is what CI runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from schema import POLICY_DIR, POLICY_YAML, CalibrationStatus, Policy, load_policy  # noqa: E402

POLICY_MD = POLICY_DIR / "POLICY.md"
MANIFEST = POLICY_DIR / "manifest.json"

# ARCHITECTURE.md §8.2 — above these, corpus C graduates to retrieval.
MAX_TOKENS = 200_000
MAX_PAGES = 500
WORDS_PER_PAGE = 450


def _rule_md(rule) -> str:
    out = [
        f"#### {rule.id} — {rule.title}",
        "",
        f"**Severity:** `{rule.severity.value}` · "
        f"**Scope:** `{rule.scope.value}` · "
        + (
            f"**Critic criterion:** {rule.critic_criterion}"
            if rule.critic_criterion
            else "**Critic criterion:** —"
        ),
        "",
        rule.statement.strip(),
        "",
    ]

    if rule.check.fn:
        params = json.dumps(rule.check.params, indent=2, ensure_ascii=False)
        out += [f"*Evaluated in code by* `{rule.check.fn}`", "", "```json", params, "```", ""]
        if rule.check.overrides:
            out += [f"*Overrides* `{rule.check.overrides}` where both apply.", ""]
    else:
        out += ["*Not machine-checked.* This rule must never be cited as grounds for rejection.", ""]

    out += [f"**Why:** {rule.rationale.strip()}", ""]

    if rule.limitation:
        out += [f"**Known limitation:** {rule.limitation.strip()}", ""]

    cal = rule.calibration
    if cal.status is not CalibrationStatus.SETTLED:
        marker = "⚠︎ UNGROUNDED" if cal.status is CalibrationStatus.PROVISIONAL_UNGROUNDED else "provisional"
        dep = f" · depends on {', '.join(cal.depends_on)}" if cal.depends_on else ""
        out += [
            f"> **Calibration — {marker}.** Revisit at **{cal.revisit_at}**{dep}.",
            *( ["> " + ln if ln.strip() else ">"
                for ln in cal.note.strip().splitlines()] if cal.note else [">"] ),
            "",
        ]

    return "\n".join(out)


def render(policy: Policy) -> str:
    L: list[str] = []
    a = L.append

    a("<!-- GENERATED FROM policy.yaml BY render.py — DO NOT EDIT BY HAND. -->")
    a("")
    a("# DHAWQ — Merchandising Policy")
    a("")
    a(f"**Corpus C** · version `{policy.policy_version}` · "
      f"effective {policy.effective_from} · authored by {policy.authored_by}")
    a("")
    a("---")
    a("")
    a("## Status of this document")
    a("")
    a(policy.authority.statement.strip())
    a("")
    a("This document is loaded **whole** into context. It is not chunked, embedded")
    a("or retrieved (ARCHITECTURE.md §8.2). A critic that reads the entire policy")
    a("every time cannot miss a rule because a chunk failed to rank.")
    a("")

    a("## How to read a rule")
    a("")
    a("Every rule has an id of the form `POL-<DOMAIN>-<NN>`. **Cite the id.** A")
    a("rejection that does not name a rule id is not actionable, and a rule id")
    a("that does not appear in this document does not exist.")
    a("")
    a("Each rule carries a severity:")
    a("")
    for name, desc in policy.severity_levels.items():
        a(f"- **`{name}`** — {desc.strip()}")
        a("")

    a("## Precedence")
    a("")
    a("Where two rules cannot both be satisfied, the earlier domain wins. The")
    a("optimiser objective is last: it never overrides a constraint.")
    a("")
    a(" → ".join(f"`{d}`" for d in policy.precedence))
    a("")

    a("## Definitions")
    a("")
    a("Every term the rules depend on. A rule resting on an undefined term is not")
    a("machine-checkable.")
    a("")
    for defn in policy.definitions.values():
        a(f"### {defn.term}")
        a("")
        a(defn.definition.strip())
        a("")
        extra = defn.model_extra or {}
        for key in ("formula", "params", "mapping", "adjacency", "requires",
                    "note", "honesty", "depends_on_artifact"):
            if key in extra:
                val = extra[key]
                if isinstance(val, (dict, list)):
                    a(f"*{key}:*")
                    a("")
                    a("```json")
                    a(json.dumps(val, indent=2, ensure_ascii=False))
                    a("```")
                else:
                    a(f"*{key}:* {str(val).strip()}")
                a("")

    a("---")
    a("")
    a("## Rules")
    a("")
    for section in policy.sections:
        a(f"### {section.id} — {section.title}")
        a("")
        a(section.preamble.strip())
        a("")
        for rule in section.rules:
            a(_rule_md(rule))
    return "\n".join(L).rstrip() + "\n"


def build_manifest(policy: Policy, text: str) -> dict:
    words = len(text.split())
    # Character-based estimate. The exact count comes from the Anthropic
    # `messages.count_tokens` endpoint once an API client exists at D9; this is
    # a standing approximation so the §8.2 threshold is monitored from day one
    # rather than assumed. It is labelled as an estimate deliberately.
    est_tokens = len(text) // 4
    est_pages = round(words / WORDS_PER_PAGE, 1)

    return {
        "corpus": "C",
        "name": "merchandising_policy",
        "policy_version": policy.policy_version,
        "generated_on": date.today().isoformat(),
        "source": "policy.yaml",
        "source_sha256": hashlib.sha256(POLICY_YAML.read_bytes()).hexdigest(),
        "rendered_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "counts": {
            "sections": len(policy.sections),
            "rules": len(policy.rules),
            "rules_by_severity": {
                s: sum(1 for r in policy.rules if r.severity.value == s)
                for s in ("hard", "escalate", "soft", "advisory")
            },
            "rules_by_criterion": {
                str(c): len(policy.for_criterion(c)) for c in range(1, 10)
                if policy.for_criterion(c)
            },
            "unsettled_thresholds": len(policy.unsettled()),
        },
        "size": {
            "chars": len(text),
            "words": words,
            "estimated_pages": est_pages,
            "estimated_tokens": est_tokens,
            "estimation_method": "chars // 4 — replace with messages.count_tokens at D9",
        },
        "retrieval": {
            "strategy": "load_whole_into_context",
            "chunked": False,
            "indexed": False,
            "rationale_ref": "ARCHITECTURE.md §8.2",
            "graduation_threshold": {"max_tokens": MAX_TOKENS, "max_pages": MAX_PAGES},
            "within_threshold": est_tokens <= MAX_TOKENS and est_pages <= MAX_PAGES,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="fail if the committed artefacts are stale")
    args = ap.parse_args()

    policy = load_policy()
    text = render(policy)
    manifest = build_manifest(policy, text)

    if args.check:
        if not POLICY_MD.exists():
            print("POLICY.md missing — run render.py", file=sys.stderr)
            return 1
        if POLICY_MD.read_text(encoding="utf-8") != text:
            print("POLICY.md is stale — policy.yaml changed. Run render.py",
                  file=sys.stderr)
            return 1
        committed = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if committed.get("source_sha256") != manifest["source_sha256"]:
            print("manifest.json is stale — run render.py", file=sys.stderr)
            return 1
        print("policy artefacts in sync")
        return 0

    POLICY_MD.write_text(text, encoding="utf-8")
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    s = manifest["size"]
    print(f"wrote POLICY.md  {s['words']:,} words · ~{s['estimated_pages']} pages "
          f"· ~{s['estimated_tokens']:,} est. tokens")
    print(f"wrote manifest.json  {manifest['counts']['rules']} rules · "
          f"{manifest['counts']['unsettled_thresholds']} unsettled thresholds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
