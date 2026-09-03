"use client";

import { useEffect, useState } from "react";

/* Segments (§12.6) — RFM cohorts and the projected CLV distribution.
   AGGREGATES ONLY. There is no endpoint that returns an individual customer,
   because POL-SEG-02 and the §13.2 matrix both forbid it. */

const SECTION: React.CSSProperties = {
  fontSize: "var(--step--1)", textTransform: "uppercase", letterSpacing: "0.12em",
  color: "var(--text-muted)", marginBlockEnd: "var(--space-3)", fontWeight: 600,
};

export default function SegmentsPage() {
  const [rfm, setRfm] = useState<any>(null);
  const [clv, setClv] = useState<any>(null);

  useEffect(() => {
    fetch("/api/segments/rfm").then((r) => r.json()).then(setRfm).catch(() => {});
    fetch("/api/segments/clv").then((r) => r.json()).then(setClv).catch(() => {});
  }, []);

  const maxSeg = rfm ? Math.max(...rfm.segments.map((s: any) => s.customers)) : 1;
  const maxBin = clv ? Math.max(...clv.histogram.map((h: any) => h.n)) : 1;
  const corr = clv?.assumption_check?.frequency_monetary_correlation ?? 0;

  return (
    <div style={{ padding: "var(--space-6)", maxInlineSize: 1120, marginInline: "auto" }}>
      <h1 style={{ fontSize: "var(--step-3)", margin: 0, letterSpacing: "-0.03em" }}>
        Segments
      </h1>
      <p style={{ color: "var(--text-muted)", maxInlineSize: "64ch", lineHeight: 1.6 }}>
        RFM cohorts and projected customer lifetime value. Aggregates only —
        no endpoint here returns an individual customer.
      </p>

      <section style={{ marginBlockStart: "var(--space-6)" }}>
        <h2 style={SECTION}>RFM segments</h2>
        {!rfm ? (
          <div className="skeleton" style={{ blockSize: 200, borderRadius: 8 }} />
        ) : (
          <div style={{ display: "grid", gap: 8, maxInlineSize: 720 }}>
            {rfm.segments.map((s: any) => (
              <div key={s.segment} style={{
                display: "grid", gridTemplateColumns: "160px 1fr 90px 90px",
                gap: "var(--space-3)", alignItems: "center",
              }}>
                <span style={{ fontSize: "var(--step--1)" }}>
                  {s.segment.replace(/_/g, " ")}
                </span>
                <div style={{ background: "var(--surface)", blockSize: 6, borderRadius: 99, overflow: "hidden" }}>
                  <div style={{
                    inlineSize: `${(s.customers / maxSeg) * 100}%`, blockSize: "100%",
                    background: "var(--signal)", borderRadius: 99,
                    transition: "inline-size var(--dur-slow) var(--ease-out)",
                  }} />
                </div>
                <span className="tnum" style={{ fontSize: "var(--step--1)", textAlign: "end" }}>
                  {s.customers.toLocaleString()}
                </span>
                <span className="tnum" style={{ fontSize: "var(--step--1)", textAlign: "end", color: "var(--text-faint)" }}
                      title="mean shopping occasions">
                  {s.mean_frequency}×
                </span>
              </div>
            ))}
          </div>
        )}
      </section>

      <section style={{ marginBlockStart: "var(--space-7)" }}>
        <h2 style={SECTION}>Projected CLV distribution</h2>
        {!clv ? (
          <div className="skeleton" style={{ blockSize: 180, borderRadius: 8 }} />
        ) : (
          <>
            <div style={{ display: "flex", gap: "var(--space-6)", flexWrap: "wrap", marginBlockEnd: "var(--space-5)" }}>
              {[
                ["customers", clv.n_customers.toLocaleString()],
                ["mean projected CLV", clv.projected_clv.mean.toFixed(4)],
                ["median", clv.projected_clv.median.toFixed(4)],
                ["mean P(alive)", clv.probability_alive.mean.toFixed(3)],
              ].map(([k, v]: any) => (
                <div key={k}>
                  <div className="tnum" style={{ fontSize: "var(--step-2)", color: "var(--signal)", lineHeight: 1 }}>{v}</div>
                  <div style={{ fontSize: "var(--step--1)", color: "var(--text-muted)", marginBlockStart: 4 }}>{k}</div>
                </div>
              ))}
            </div>

            <div style={{ display: "flex", alignItems: "flex-end", gap: 2, blockSize: 130 }}>
              {clv.histogram.map((h: any, i: number) => (
                <div key={i} title={`${h.n} customers`} style={{
                  flex: 1, blockSize: `${Math.max(2, (h.n / maxBin) * 100)}%`,
                  background: "var(--signal)", opacity: 0.28 + 0.72 * (h.n / maxBin),
                  borderRadius: "2px 2px 0 0",
                }} />
              ))}
            </div>
            <div style={{ fontSize: "var(--step--1)", color: "var(--text-faint)", marginBlockStart: 6 }}>
              projected CLV, clipped at the 99th percentile so the tail does not flatten the chart
            </div>

            <div style={{
              marginBlockStart: "var(--space-5)", padding: "var(--space-4)",
              borderRadius: "var(--radius-md)", background: "var(--surface)",
              maxInlineSize: "76ch",
            }}>
              <div style={{ fontSize: "var(--step--1)", fontWeight: 600 }}>
                Assumption check ·{" "}
                <span className="tnum" style={{ color: Math.abs(corr) > 0.3 ? "var(--reject)" : "var(--signal)" }}>
                  r = {corr.toFixed(3)}
                </span>
              </div>
              <div style={{ fontSize: "var(--step--1)", color: "var(--text-muted)", marginBlockStart: 6, lineHeight: 1.6 }}>
                {clv.assumption_check.note}{" "}
                {Math.abs(corr) > 0.3
                  ? "This correlation is high enough to undermine the model — treat the values as indicative only."
                  : "Low enough for the independence assumption to hold."}
              </div>
              <div style={{ fontSize: "var(--step--1)", color: "var(--text-faint)", marginBlockStart: 8 }}>
                {clv.language}
              </div>
            </div>
          </>
        )}
      </section>
    </div>
  );
}
