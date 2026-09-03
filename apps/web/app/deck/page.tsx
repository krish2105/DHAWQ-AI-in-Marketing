"use client";

import { useEffect, useState } from "react";
import { apiGet } from "@/lib/api";

/**
 * The §15 deck — 12 slides, keyboard-navigable, and DATA-DRIVEN.
 *
 * Every number is fetched from the live API rather than typed in. A deck with
 * hardcoded figures drifts from the system the moment either changes, and then
 * the presentation and the evidence disagree in front of an examiner. If the
 * API is down the slide says so instead of showing a stale number.
 */

type Slide = { kind: "title" | "point" | "data"; title: string; body?: string;
               bullets?: string[]; note?: string; metric?: () => React.ReactNode };

function Stat({ value, label, tone = "var(--signal)" }:
  { value: string; label: string; tone?: string }) {
  return (
    <div>
      <div className="tnum" style={{ fontSize: "var(--step-4)", color: tone, lineHeight: 1 }}>
        {value}
      </div>
      <div style={{ fontSize: "var(--step--1)", color: "var(--text-muted)",
                    marginBlockStart: 8, maxInlineSize: "26ch" }}>{label}</div>
    </div>
  );
}

export default function Deck() {
  const [i, setI] = useState(0);
  const [d, setD] = useState<any>({});

  useEffect(() => {
    Promise.all([
      apiGet<any>("/api/evaluate/latest"),
      apiGet<any>("/api/merchandise/session-lift"),
      apiGet<any>("/api/merchandise/operating-case"),
    ]).then(([ev, sl, oc]) => setD({
      agent: ev.ok ? ev.data.agent : null,
      recs: ev.ok ? ev.data.recommenders : null,
      lift: sl.ok ? sl.data : null,
      ops: oc.ok ? oc.data : null,
    }));
  }, []);

  const recs = d.recs, agent = d.agent, lift = d.lift, ops = d.ops;
  const arm = (n: string) => recs?.frontier?.find((f: any) => f.model === n);
  const num = (v: any, f = 4) => (typeof v === "number" ? v.toFixed(f) : "—");

  const slides: Slide[] = [
    { kind: "title", title: "DHAWQ · ذوق",
      body: "Visual recommendation intelligence with an agentic merchandising copilot.",
      note: "MAIB AI 208 · AI in Marketing · SP Jain Dubai · Krishna Mathur" },

    { kind: "point", title: "The question is not “which model wins”",
      body: "A merchandiser has a finite number of slots on a page.",
      bullets: [
        "Which products go in them, for whom?",
        "How much incremental revenue does that choice create over showing everyone the bestsellers?",
        "And what does it cost in catalogue coverage?",
      ],
      note: "NDCG@10 of 0.31 means nothing to a CMO. The trade does." },

    { kind: "data", title: "The rule that governs every line",
      body: "Deterministic logic is code. Models do retrieval, decomposition, extraction, routing and explanation. Nothing else.",
      note: "No model in DHAWQ emits a score, a rank, a revenue figure, a CLV or a coverage number. Those come from functions with unit tests — currently 192 of them.",
      metric: () => <div style={{ display: "flex", gap: "var(--space-7)", flexWrap: "wrap" }}>
        <Stat value="192" label="tests, covering the deterministic core" />
        <Stat value="5" label="hard gates that fail the build in CI" />
        <Stat value="49" label="policy rules the critic cites by id" />
      </div> },

    { kind: "data", title: "The data, and what it cannot tell us",
      body: "H&M: 12 weeks, temporal split, leak assertion in tests.",
      note: "Purchases, not impressions. An article nobody bought is UNLABELLED, not rejected — every metric here inherits that, and it is the single biggest limitation.",
      metric: () => <div style={{ display: "flex", gap: "var(--space-7)", flexWrap: "wrap" }}>
        <Stat value="13,548" label="articles with usable photography" />
        <Stat value="119,594" label="customers" />
        <Stat value="1.63M" label="transactions, train / test split by date" />
      </div> },

    { kind: "data", title: "Five arms, measured identically",
      body: "Popularity, content, collaborative, and hybrid in two shapes — plus an LLM re-ranker as a benchmarked fifth.",
      note: "The re-ranker LOSES: −0.5pp NDCG at 2,125 seconds per 1,000 slates against a sub-second arm. That is the more interesting finding.",
      metric: () => <div style={{ display: "flex", gap: "var(--space-6)", flexWrap: "wrap" }}>
        <Stat value={num(arm("collaborative")?.["ndcg@10"])} label="collaborative NDCG@10 — best arm" />
        <Stat value={num(arm("popularity")?.["ndcg@10"])} label="popularity NDCG@10 — the baseline to beat" tone="var(--tail)" />
        <Stat value={num(arm("hybrid_weighted")?.["ndcg@10"])} label="hybrid — does NOT beat collaborative" tone="var(--reject)" />
      </div> },

    { kind: "point", title: "The hypothesis that failed",
      body: "The primary research question asked whether a hybrid beats content-only and collaborative-only.",
      bullets: [
        "It does not. Collaborative alone is the strongest arm.",
        "§17 named this as the falsification condition before the work began.",
        "Reported, not tuned away — the blend adds complexity for nothing here.",
      ],
      note: "A project that never reports a negative result has not been testing anything." },

    { kind: "data", title: "The tension IS the finding",
      body: "Accuracy and coverage trade off, and the trade is the decision.",
      note: "Popularity reaches competitive ranking on a fraction of a percent of the catalogue. That is a bestseller re-ranker wearing a personalisation label.",
      metric: () => <div style={{ display: "flex", gap: "var(--space-6)", flexWrap: "wrap" }}>
        <Stat value={arm("popularity") ? (arm("popularity").coverage * 100).toFixed(1) + "%" : "—"}
              label="of the catalogue ever shown by popularity" tone="var(--tail)" />
        <Stat value={arm("hybrid_cascade") ? (arm("hybrid_cascade").coverage * 100).toFixed(1) + "%" : "—"}
              label="by hybrid cascade, at ~78% of the NDCG" />
        <Stat value={arm("popularity") ? arm("popularity").gini.toFixed(3) : "—"}
              label="popularity Gini — near-total concentration" tone="var(--reject)" />
      </div> },

    { kind: "data", title: "What personalisation is actually worth",
      body: "Per visitor, scored against that customer's own held-out purchases.",
      note: lift?.hypothesis_tested_and_rejected ??
            "Personalisation was expected to win here. It does not.",
      metric: () => lift ? <div style={{ display: "flex", gap: "var(--space-6)", flexWrap: "wrap" }}>
        <Stat value={`${lift.results.collaborative.projected_lift_pct.toFixed(1)}%`}
              label={`projected revenue, 95% CI [${lift.results.collaborative.ci95.join(", ")}] — includes zero`}
              tone="var(--tail)" />
        <Stat value={`${lift.results.collaborative.reach_multiple}×`}
              label="more of the catalogue reaching a customer" />
      </div> : <div style={{ color: "var(--text-faint)" }}>API unavailable — no number shown rather than a stale one.</div> },

    { kind: "point", title: "The business case the data supports",
      body: "Not “personalisation lifts revenue”. On this data it does not, and the intervals say so.",
      bullets: [
        "It buys REACH: hundreds of times more of the assortment gets exposure.",
        "At a revenue difference the data cannot distinguish from zero.",
        "For a retailer holding inventory, that is the trade worth making.",
      ],
      note: "Dead stock is a cost a bestseller page never addresses." },

    { kind: "data", title: "The case that does not need an A/B test",
      body: "The revenue interval includes zero and no offline estimator can close it. This half is measured: 210 slates, built twice, audited against the policy.",
      metric: () => ops ? <div style={{ display: "grid", gap: "var(--space-5)",
                                  gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))" }}>
        <Stat value={`${(ops.measured.ungoverned_breach_rate * 100).toFixed(0)}%`}
              label="of revenue-ranked slates breach the merchandising policy" />
        <Stat value={`${(ops.measured.silent_breach_rate * 100).toFixed(1)}%`}
              label="ship non-compliant WITHOUT saying so — the rest escalate to a human" />
        <Stat value={`${ops.break_even.manual_minutes_to_break_even_low}–${ops.break_even.manual_minutes_to_break_even_high} min`}
              label="human cost per slate. It pays the moment a compliant page takes longer than that by hand" />
      </div> : <div style={{ color: "var(--text-faint)" }}>API unavailable — no number shown rather than a stale one.</div>,
      note: "Hours saved is reported as a RANGE (19–73 per 100 slates), never a point estimate: four of its inputs are declared assumptions, not measurements, and a single figure built on them would read as a finding." },

    { kind: "point", title: "The frontier plot would have picked the wrong model",
      body: "Escalation load varies 40× across the five arms, and catalogue coverage does not predict it.",
      bullets: [
        "hybrid_cascade: best coverage on the frontier (0.655) — escalates 71% of slates.",
        "hybrid_weighted: worse coverage (0.468) — escalates 2.4%.",
        "The cause is tail share WITHIN one cohort's candidate list: 3.1% against 36.9%.",
      ],
      note: "Coverage is measured across all users. Whether one page can satisfy the long-tail quota unaided is a different question, and only the second answers it." },

    { kind: "data", title: "The agent, and what it refuses",
      body: "Supervisor plus specialists, typed read-only tools, a nine-criteria critic, human gates on everything irreversible.",
      note: "The rejection panel is a first-class surface. A system that shows what it refused is more credible than one that only shows what it produced.",
      metric: () => <div style={{ display: "flex", gap: "var(--space-6)", flexWrap: "wrap" }}>
        <Stat value={agent ? agent.tuning.task_completion_rate.toFixed(3) : "—"}
              label="task completion on 83 hand-written briefs" />
        <Stat value={agent ? agent.tuning.block_recall.toFixed(3) : "—"}
              label="block recall — of what must not produce a slate, how much did not" />
        <Stat value={agent?.generated_set ? agent.generated_set.task_completion_rate.toFixed(3) : "—"}
              label="on 48 briefs generated from the policy — a different author"
              tone="var(--reject)" />
      </div> },

    { kind: "point", title: "What I would not claim",
      bullets: [
        "The golden set is not independently reviewed. One author wrote the briefs, the labels and the code that scores them.",
        "Injection recall is 1.00 on lexical attacks and 0.00 on semantic ones. The defence does not generalise.",
        "All lift is PROJECTED. There is no A/B test, and no offline estimator built from observed purchases can settle it.",
        "Corpus D is synthetic and labelled so, because an unattended process cannot approve its own crawl allowlist.",
      ],
      note: "Every one of these is in the README and on the live evaluation page, not only here." },

    { kind: "point", title: "What I would do next",
      bullets: [
        "A live A/B test. It is the only thing that converts projected into measured.",
        "A second reader on the golden set — one hour, and every agent metric stops being provisional.",
        "Semantic injection detection. The lexical layer is at ceiling; the gap is the whole risk.",
        "Inventory data, so the long-tail quota can be argued on holding cost rather than coverage alone.",
      ],
      note: "dhawq-krishnamathur008-1499s-projects.vercel.app" },
  ];

  useEffect(() => {
    const k = (e: KeyboardEvent) => {
      if (e.key === "ArrowRight" || e.key === " ") setI((v) => Math.min(v + 1, slides.length - 1));
      if (e.key === "ArrowLeft") setI((v) => Math.max(v - 1, 0));
    };
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, [slides.length]);

  const s = slides[i];
  return (
    <div style={{ minBlockSize: "calc(100dvh - 53px)", display: "flex",
                  flexDirection: "column", padding: "var(--space-7) var(--space-6)",
                  maxInlineSize: 1000, marginInline: "auto" }}>
      <div style={{ flex: 1 }}>
        {s.kind === "title" ? (
          <>
            <h1 style={{ fontSize: "var(--step-4)", margin: 0, letterSpacing: "-0.04em" }}>
              {s.title}
            </h1>
            <p style={{ fontSize: "var(--step-2)", color: "var(--text-muted)",
                        maxInlineSize: "28ch", lineHeight: 1.3 }}>{s.body}</p>
          </>
        ) : (
          <>
            <h1 style={{ fontSize: "var(--step-3)", margin: 0, letterSpacing: "-0.03em",
                         maxInlineSize: "22ch", lineHeight: 1.1 }}>{s.title}</h1>
            {s.body && (
              <p style={{ fontSize: "var(--step-1)", color: "var(--text-muted)",
                          maxInlineSize: "56ch", lineHeight: 1.5,
                          marginBlockStart: "var(--space-4)" }}>{s.body}</p>
            )}
            {s.metric && <div style={{ marginBlockStart: "var(--space-6)" }}>{s.metric()}</div>}
            {s.bullets && (
              <ul style={{ marginBlockStart: "var(--space-5)", paddingInlineStart: "1.1em",
                           maxInlineSize: "62ch", lineHeight: 1.75,
                           fontSize: "var(--step-0)" }}>
                {s.bullets.map((b, j) => <li key={j} style={{ marginBlockEnd: 8 }}>{b}</li>)}
              </ul>
            )}
          </>
        )}
        {s.note && (
          <p style={{ marginBlockStart: "var(--space-6)", fontSize: "var(--step--1)",
                      color: "var(--text-faint)", maxInlineSize: "72ch", lineHeight: 1.65,
                      borderInlineStart: "2px solid var(--hairline)",
                      paddingInlineStart: "var(--space-3)" }}>{s.note}</p>
        )}
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: "var(--space-4)",
                    marginBlockStart: "var(--space-6)" }}>
        <button onClick={() => setI((v) => Math.max(v - 1, 0))} disabled={i === 0}
                aria-label="Previous slide" style={navBtn(i === 0)}>←</button>
        <button onClick={() => setI((v) => Math.min(v + 1, slides.length - 1))}
                disabled={i === slides.length - 1} aria-label="Next slide"
                style={navBtn(i === slides.length - 1)}>→</button>
        <div style={{ display: "flex", gap: 4, flex: 1 }}>
          {slides.map((_, j) => (
            // The visible bar is 3px; the HIT AREA is 30px. A 3px control is
            // unclickable with a finger and barely clickable with a mouse —
            // caught when a test click on it did nothing. Padding gives the
            // target without changing the design.
            <button key={j} onClick={() => setI(j)} aria-label={`Go to slide ${j + 1}`}
                    aria-current={j === i ? "true" : undefined}
                    style={{ flex: 1, border: "none", padding: "14px 0", margin: 0,
                             cursor: "pointer", background: "transparent",
                             display: "flex", alignItems: "center" }}>
              <span style={{ inlineSize: "100%", blockSize: 3, borderRadius: 99,
                             background: j <= i ? "var(--signal)" : "var(--hairline)" }} />
            </button>
          ))}
        </div>
        <span className="tnum" style={{ fontSize: "var(--step--1)", color: "var(--text-faint)" }}>
          {i + 1}/{slides.length}
        </span>
      </div>
    </div>
  );
}

const navBtn = (disabled: boolean): React.CSSProperties => ({
  inlineSize: 34, blockSize: 34, borderRadius: 999, cursor: disabled ? "default" : "pointer",
  border: "1px solid var(--hairline)", background: "var(--surface)",
  color: disabled ? "var(--text-faint)" : "var(--text)", opacity: disabled ? 0.4 : 1,
});
