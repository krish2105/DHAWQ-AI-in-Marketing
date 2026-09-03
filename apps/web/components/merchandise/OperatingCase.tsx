"use client";

/* The cost side of the argument, built so the two kinds of number cannot be
   mistaken for each other.

   MEASURED is rendered in the signal colour on a solid rule and says what was
   counted. ASSUMED is muted, on a DASHED rule, carries its range, and is
   labelled "declared" where there is no evidence. A reader who only skims must
   still come away knowing which is which — if the difference were a footnote,
   the assumptions would read as findings, which is the entire failure mode
   this section is trying to avoid. */

import { useEffect, useState } from "react";
import { apiGet } from "@/lib/api";

type Assumption = {
  name: string; low: number; high: number; unit: string;
  source: string; note: string; mid: number;
};

type Case = {
  measured: {
    escalation_rate: number; ungoverned_breach_rate: number;
    silent_breach_rate: number; slates_audited: number; cohorts: number;
    mean_violations_per_ungoverned_slate: number;
    ungoverned_by_rule: Record<string, number>;
    by_model: Record<string, {
      n: number; escalated: number; escalation_rate: number;
      cohort_pool_tail_share: number;
    }>;
  };
  assumptions: Assumption[];
  per_100_slates: Record<string, number>;
  break_even: {
    manual_minutes_to_break_even_low: number;
    manual_minutes_to_break_even_high: number; reading: string;
  };
  caveats: string[];
  method: string;
};

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

function Label({ kind }: { kind: "measured" | "assumed" }) {
  const measured = kind === "measured";
  return (
    <span style={{
      fontFamily: "var(--font-mono)", fontSize: 9, letterSpacing: "0.14em",
      textTransform: "uppercase", padding: "2px 6px", borderRadius: 3,
      color: measured ? "var(--signal)" : "var(--text-faint)",
      border: `1px ${measured ? "solid" : "dashed"} ${
        measured ? "var(--signal)" : "var(--hairline)"}`,
      flexShrink: 0,
    }}>
      {measured ? "measured" : "assumed"}
    </span>
  );
}

function Stat({ value, label, tone = "text" }:
              { value: string; label: string; tone?: "text" | "signal" | "reject" }) {
  return (
    <div>
      <div style={{
        fontFamily: "var(--font-mono)", fontSize: "var(--step-2)",
        fontVariantNumeric: "tabular-nums", letterSpacing: "-0.02em",
        color: tone === "signal" ? "var(--signal)"
             : tone === "reject" ? "var(--reject)" : "var(--text)",
      }}>{value}</div>
      <div style={{ fontSize: "var(--step--1)", color: "var(--text-muted)",
                    lineHeight: 1.45, marginBlockStart: 2, maxInlineSize: "30ch" }}>
        {label}
      </div>
    </div>
  );
}

export default function OperatingCase() {
  const [c, setC] = useState<Case | null>(null);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    let live = true;
    apiGet<Case>("/api/merchandise/operating-case")
      .then((r) => { if (live && r.ok) setC(r.data); });
    return () => { live = false; };
  }, []);

  if (!c) return null;
  const m = c.measured;
  const p = c.per_100_slates;
  const models = Object.entries(m.by_model)
    .sort((a, b) => a[1].escalation_rate - b[1].escalation_rate);

  return (
    <section style={{ marginBlock: "var(--space-7)" }}>
      <h2 style={{
        fontSize: "var(--step--1)", textTransform: "uppercase",
        letterSpacing: "0.12em", color: "var(--text-muted)", fontWeight: 600,
      }}>
        What governance is worth, when the revenue case is not proven
      </h2>
      <p style={{ fontSize: "var(--step--1)", color: "var(--text-muted)",
                  maxInlineSize: "76ch", lineHeight: 1.65,
                  marginBlockEnd: "var(--space-5)" }}>
        The lift above has a confidence interval that includes zero, and no
        offline estimator can close that — only a live A/B test could. This is
        the half that does not need one: {m.slates_audited} slates across{" "}
        {m.cohorts} cohorts, each built twice and audited against the policy.
      </p>

      <div style={{ display: "flex", alignItems: "center", gap: 10,
                    marginBlockEnd: "var(--space-4)", flexWrap: "wrap" }}>
        <Label kind="measured" />
        <span style={{ fontSize: "var(--step--1)", color: "var(--text-muted)" }}>
          counted by <code style={{ fontFamily: "var(--font-mono)", fontSize: 11 }}>
          09_operating_case.py</code>, reproducible by re-running it
        </span>
      </div>

      <div style={{
        display: "grid", gap: "var(--space-5)",
        gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))",
        padding: "var(--space-5)", borderRadius: "var(--radius-md)",
        background: "var(--surface)",
        borderInlineStart: "3px solid var(--signal)",
      }}>
        <Stat value={pct(m.ungoverned_breach_rate)} tone="reject"
              label={`of revenue-ranked slates breach the policy — mean ${m.mean_violations_per_ungoverned_slate} rules each`} />
        <Stat value={pct(m.escalation_rate)}
              label="of governed slates declare a breach and reach a human" />
        <Stat value={pct(m.silent_breach_rate)} tone="signal"
              label="ship non-compliant without saying so — the one failure the escalation path exists to prevent" />
      </div>

      <p style={{ fontSize: "var(--step--1)", color: "var(--text-faint)",
                  marginBlockStart: "var(--space-3)", maxInlineSize: "76ch",
                  lineHeight: 1.6 }}>
        Rules breached ungoverned:{" "}
        {Object.entries(m.ungoverned_by_rule)
          .map(([r, n]) => `${r} (${n})`).join(" · ")}
      </p>

      <h3 style={{ fontSize: "var(--step-0)", marginBlockStart: "var(--space-6)",
                   marginBlockEnd: 4 }}>
        Which recommender you ship decides the sign-off load
      </h3>
      <p style={{ fontSize: "var(--step--1)", color: "var(--text-muted)",
                  maxInlineSize: "76ch", lineHeight: 1.6,
                  marginBlockEnd: "var(--space-4)" }}>
        A 40&times; spread, and the cause is not catalogue coverage. Coverage is
        measured across <em>all</em> users; what decides whether one page can
        meet the long-tail quota unaided is the tail share of{" "}
        <em>that cohort&rsquo;s</em> candidate list. The cascade hybrid wins on
        the frontier plot and loses badly here.
      </p>

      <div style={{ overflowX: "auto" }}>
        <table style={{ borderCollapse: "collapse", inlineSize: "100%",
                        minInlineSize: 400, fontSize: "var(--step--1)" }}>
          <thead>
            <tr style={{ color: "var(--text-faint)" }}>
              <th style={{ textAlign: "start", paddingBlock: 6, fontWeight: 500 }}>model</th>
              <th style={{ textAlign: "end", paddingBlock: 6, fontWeight: 500 }}>escalates</th>
              <th style={{ textAlign: "end", paddingBlock: 6, fontWeight: 500 }}>cohort pool tail</th>
            </tr>
          </thead>
          <tbody style={{ fontFamily: "var(--font-mono)",
                          fontVariantNumeric: "tabular-nums" }}>
            {models.map(([name, v]) => (
              <tr key={name} style={{ borderBlockStart: "1px solid var(--hairline)" }}>
                <td style={{ paddingBlock: 8 }}>{name}</td>
                <td style={{ textAlign: "end",
                             color: v.escalation_rate > 0.5
                               ? "var(--reject)" : "var(--signal)" }}>
                  {pct(v.escalation_rate)}
                </td>
                <td style={{ textAlign: "end", color: "var(--text-muted)" }}>
                  {pct(v.cohort_pool_tail_share)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: 10,
                    marginBlockStart: "var(--space-6)",
                    marginBlockEnd: "var(--space-4)", flexWrap: "wrap" }}>
        <Label kind="assumed" />
        <span style={{ fontSize: "var(--step--1)", color: "var(--text-muted)",
                       maxInlineSize: "60ch" }}>
          nobody has measured these — replace them with your own team&rsquo;s
          numbers before quoting anything below
        </span>
      </div>

      <div style={{
        padding: "var(--space-5)", borderRadius: "var(--radius-md)",
        border: "1px dashed var(--hairline)",
      }}>
        <div style={{ display: "grid", gap: "var(--space-5)",
                      gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))" }}>
          <Stat value={`${p.manual_hours_low}–${p.manual_hours_high} h`}
                label="to build and check 100 compliant slates by hand" />
          <Stat value={`${p.system_hours_low}–${p.system_hours_high} h`}
                label="human time for the same 100 through DHAWQ, all of it reviewing escalations" />
          <Stat value={`${p.hours_saved_low}–${p.hours_saved_high} h`}
                label="the range, not an estimate — one figure here would read as a finding" />
        </div>

        <button
          onClick={() => setOpen((v) => !v)}
          style={{
            marginBlockStart: "var(--space-4)", background: "transparent",
            border: "none", padding: "10px 0", cursor: "pointer",
            color: "var(--text-muted)", font: "inherit",
            fontSize: "var(--step--1)", textDecoration: "underline",
            textUnderlineOffset: 3,
          }}
        >
          {open ? "Hide" : "Show"} the four assumptions and why each is a range
        </button>

        {open && (
          <dl style={{ margin: 0, marginBlockStart: "var(--space-2)" }}>
            {c.assumptions.map((a) => (
              <div key={a.name} style={{
                paddingBlock: "var(--space-3)",
                borderBlockStart: "1px solid var(--hairline)",
              }}>
                <dt style={{ display: "flex", gap: 10, alignItems: "baseline",
                             flexWrap: "wrap" }}>
                  <code style={{ fontFamily: "var(--font-mono)",
                                 fontSize: "var(--step--1)" }}>{a.name}</code>
                  <span style={{ fontFamily: "var(--font-mono)",
                                 fontSize: "var(--step--1)" }}>
                    {a.low}–{a.high} {a.unit}
                  </span>
                  <span style={{ fontFamily: "var(--font-mono)", fontSize: 10,
                                 color: "var(--reject)", letterSpacing: "0.08em",
                                 textTransform: "uppercase" }}>
                    {a.source}
                  </span>
                </dt>
                <dd style={{ margin: 0, marginBlockStart: 4,
                             fontSize: "var(--step--1)", lineHeight: 1.6,
                             color: "var(--text-muted)", maxInlineSize: "76ch" }}>
                  {a.note}
                </dd>
              </div>
            ))}
          </dl>
        )}
      </div>

      <div style={{
        marginBlockStart: "var(--space-5)", padding: "var(--space-5)",
        borderRadius: "var(--radius-md)", background: "var(--surface)",
        borderInlineStart: "3px solid var(--signal)", maxInlineSize: "80ch",
      }}>
        <div style={{
          fontFamily: "var(--font-mono)", fontSize: "var(--step-1)",
          color: "var(--signal)", letterSpacing: "-0.02em",
        }}>
          {c.break_even.manual_minutes_to_break_even_low}–
          {c.break_even.manual_minutes_to_break_even_high} min
        </div>
        <div style={{ fontSize: "var(--step--1)", lineHeight: 1.65,
                      marginBlockStart: 6 }}>
          {c.break_even.reading}
        </div>
      </div>

      <ul style={{ marginBlockStart: "var(--space-4)", paddingInlineStart: 18,
                   color: "var(--text-faint)", fontSize: "var(--step--1)",
                   lineHeight: 1.65, maxInlineSize: "80ch" }}>
        {c.caveats.map((x, i) => <li key={i} style={{ marginBlock: 4 }}>{x}</li>)}
      </ul>
    </section>
  );
}
