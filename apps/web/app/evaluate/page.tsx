"use client";

import { useCallback, useEffect, useState } from "react";
import { apiGet, type ApiError } from "@/lib/api";
import { ApiNotice } from "@/components/ui/ApiNotice";

/*
 * The evaluation view (§12.6).
 *
 * "The tension IS the finding" (§9). Accuracy and coverage trade off, so they
 * are shown TOGETHER — you cannot read the NDCG column without the coverage
 * column beside it. A model that wins ranking while collapsing coverage to 4%
 * of the catalogue is a merchandising problem, and this table is where that
 * becomes visible.
 */

const SECTION: React.CSSProperties = {
  fontSize: "var(--step--1)", textTransform: "uppercase", letterSpacing: "0.12em",
  color: "var(--text-muted)", marginBlockEnd: "var(--space-3)", fontWeight: 600,
};

function Bar({ value, max, tone }: { value: number; max: number; tone: string }) {
  return (
    <div className="bar-track">
      <div className="bar-fill"
           style={{ inlineSize: `${Math.min(100, (value / max) * 100)}%`, background: tone }} />
    </div>
  );
}

export default function EvaluatePage() {
  const [data, setData] = useState<any>(null);
  const [err, setErr] = useState<ApiError | null>(null);

  const load = useCallback(() => {
    setErr(null);
    apiGet<any>("/api/evaluate/latest").then((r) =>
      r.ok ? setData(r.data) : setErr(r.error));
  }, []);

  useEffect(load, [load]);

  // Previously this rendered String(e) — the raw "SyntaxError: The string did
  // not match the expected pattern" from JSON.parse choking on an HTML error
  // page. A parser message is not a user-facing state.
  if (err) {
    return (
      <div style={{ padding: "var(--space-6)", maxInlineSize: 1320, marginInline: "auto" }}>
        <h1 style={{ fontSize: "var(--step-3)", margin: 0, letterSpacing: "-0.03em" }}>
          Evaluation
        </h1>
        <div style={{ marginBlockStart: "var(--space-5)" }}>
          <ApiNotice error={err} onRetry={load} />
        </div>
      </div>
    );
  }
  if (!data) return <div className="skeleton" style={{ blockSize: 400, margin: "var(--space-6)" }} />;

  const recs = data.recommenders;
  const agent = data.agent;
  const maxNdcg = recs ? Math.max(...recs.frontier.map((f: any) => f["ndcg@10"])) : 1;

  return (
    <div style={{ padding: "var(--space-6)", maxInlineSize: 1320, marginInline: "auto" }}>
      <h1 style={{ fontSize: "var(--step-3)", margin: 0, letterSpacing: "-0.03em" }}>Evaluation</h1>
      <p style={{ color: "var(--text-muted)", maxInlineSize: "66ch", lineHeight: 1.6 }}>
        Accuracy and its cost, side by side. A model that wins NDCG while
        collapsing catalogue coverage is a merchandising problem — naming that
        trade is the finding, not a footnote.
      </p>

      {agent?.provenance_warning && (
        <div style={{
          marginBlock: "var(--space-5)", padding: "var(--space-4)",
          border: "1px solid var(--reject)", borderRadius: "var(--radius-md)",
          background: "var(--reject-dim)", maxInlineSize: "80ch",
        }}>
          <strong style={{ color: "var(--reject)" }}>Provisional</strong>
          <div style={{ fontSize: "var(--step--1)", marginBlockStart: 4, lineHeight: 1.55 }}>
            {agent.provenance_warning}
          </div>
        </div>
      )}

      {agent && (
        <section style={{ marginBlockStart: "var(--space-6)" }}>
          <h2 style={SECTION}>Hard gates — binary, non-negotiable, CI-blocking</h2>
          <div style={{
            display: "grid", gap: "var(--space-3)",
            gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))",
          }}>
            {Object.entries(agent.gates).map(([k, v]: any) => {
              const pass = agent.gates_pass[k];
              return (
                <div key={k} style={{
                  padding: "var(--space-4)", borderRadius: "var(--radius-md)",
                  background: "var(--surface)",
                  border: `1px solid ${pass ? "var(--hairline)" : "var(--reject)"}`,
                }}>
                  <div className="tnum" style={{
                    fontSize: "var(--step-2)", color: pass ? "var(--signal)" : "var(--reject)",
                    lineHeight: 1,
                  }}>{Number(v).toFixed(3)}</div>
                  <div style={{ fontSize: "var(--step--1)", color: "var(--text-muted)", marginBlockStart: 6 }}>
                    {k.replace(/_/g, " ")}
                  </div>
                  <div className="mono" style={{
                    fontSize: "var(--step--1)", marginBlockStart: 4,
                    color: pass ? "var(--text-faint)" : "var(--reject)",
                  }}>
                    target {agent.gate_targets[k].toFixed(2)} · {pass ? "PASS" : "FAIL"}
                  </div>
                </div>
              );
            })}
          </div>

          <h2 style={{ ...SECTION, marginBlockStart: "var(--space-6)" }}>
            Golden set by stratum · {agent.golden_set.n} briefs
          </h2>
          <div style={{ display: "grid", gap: 6, maxInlineSize: 560 }}>
            {Object.entries(agent.by_stratum).map(([s, v]: any) => (
              <div key={s} className="row-bar">
                <span className="row-label" style={{ color: "var(--text-muted)" }}>
                  {s.replace(/_/g, " ")}
                </span>
                <Bar value={v.passed} max={v.n} tone="var(--signal)" />
                <span className="tnum" style={{ fontSize: "var(--step--1)", color: "var(--text-faint)", textAlign: "end" }}>
                  {v.passed}/{v.n}
                </span>
              </div>
            ))}
          </div>

          <h2 style={{ ...SECTION, marginBlockStart: "var(--space-6)" }}>
            Injection detection — split, because the aggregate hides the gap
          </h2>
          <div style={{ display: "flex", gap: "var(--space-6)", flexWrap: "wrap" }}>
            {[
              ["designed payloads", agent.injection.recall_on_designed_payloads, "var(--signal)"],
              ["novel payloads", agent.injection.recall_on_novel_payloads, "var(--reject)"],
            ].map(([label, v, tone]: any) => (
              <div key={label}>
                <div className="tnum" style={{ fontSize: "var(--step-2)", color: tone }}>
                  {Number(v).toFixed(2)}
                </div>
                <div style={{ fontSize: "var(--step--1)", color: "var(--text-muted)" }}>{label}</div>
              </div>
            ))}
          </div>
          <p style={{ fontSize: "var(--step--1)", color: "var(--text-faint)", maxInlineSize: "70ch", marginBlockStart: "var(--space-3)", lineHeight: 1.6 }}>
            {agent.injection.honesty_note}
          </p>
        </section>
      )}

      {recs && (
        <section style={{ marginBlockStart: "var(--space-8)" }}>
          <h2 style={SECTION}>The accuracy–coverage frontier</h2>
          <div className="scroll-x">
            <table style={{ inlineSize: "100%", borderCollapse: "collapse", minInlineSize: 660 }}>
              <thead>
                <tr style={{ borderBlockEnd: "1px solid var(--hairline)" }}>
                  {["model", "NDCG@10", "coverage", "Gini", "long-tail", "pop. lift"].map((h) => (
                    <th key={h} style={{
                      textAlign: h === "model" ? "start" : "end", padding: "8px 12px",
                      fontSize: "var(--step--1)", color: "var(--text-muted)", fontWeight: 500,
                    }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {recs.frontier.map((f: any) => (
                  <tr key={f.model} style={{ borderBlockEnd: "1px solid var(--hairline)" }}>
                    <td style={{ padding: "10px 12px" }}>{f.model.replace(/_/g, " ")}</td>
                    <td className="tnum" style={{ textAlign: "end", padding: "10px 12px", color: f["ndcg@10"] === maxNdcg ? "var(--signal)" : "var(--text)" }}>
                      {f["ndcg@10"].toFixed(4)}
                    </td>
                    <td className="tnum" style={{ textAlign: "end", padding: "10px 12px" }}>{(f.coverage * 100).toFixed(1)}%</td>
                    <td className="tnum" style={{ textAlign: "end", padding: "10px 12px", color: "var(--text-muted)" }}>{f.gini.toFixed(3)}</td>
                    <td className="tnum" style={{ textAlign: "end", padding: "10px 12px", color: "var(--tail)" }}>{(f.long_tail_exposure * 100).toFixed(1)}%</td>
                    <td className="tnum" style={{ textAlign: "end", padding: "10px 12px", color: "var(--text-muted)" }}>
                      {recs.results[f.model].bias.popularity_lift.toFixed(1)}×
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p style={{ fontSize: "var(--step--1)", color: "var(--text-faint)", maxInlineSize: "76ch", marginBlockStart: "var(--space-4)", lineHeight: 1.65 }}>
            Popularity reaches competitive NDCG on <strong>0.2%</strong> of the
            catalogue at a Gini of 0.999 — it is a bestseller re-ranker.
            hybrid_cascade gives roughly 78% of collaborative&rsquo;s NDCG at
            nearly double the coverage. Which trade is right is a business
            decision, and it is presented as one.
          </p>
          <p style={{ fontSize: "var(--step--1)", color: "var(--text-faint)", maxInlineSize: "76ch", marginBlockStart: "var(--space-3)", lineHeight: 1.65 }}>
            {recs.protocol.known_limitation}
          </p>
        </section>
      )}
    </div>
  );
}
