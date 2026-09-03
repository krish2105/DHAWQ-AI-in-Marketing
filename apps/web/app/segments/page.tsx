"use client";

import { useCallback, useEffect, useState } from "react";
import { apiGet, type ApiError } from "@/lib/api";
import { ApiNotice } from "@/components/ui/ApiNotice";
import { RequireScope } from "@/components/ui/SignIn";
import Link from "next/link";
import { ClvHoldout } from "@/components/evaluate/ClvHoldout";

/* Segments (§12.6) — RFM cohorts and the projected CLV distribution.
   AGGREGATES ONLY. There is no endpoint that returns an individual customer,
   because POL-SEG-02 and the §13.2 matrix both forbid it. */

const SECTION: React.CSSProperties = {
  fontSize: "var(--step--1)", textTransform: "uppercase", letterSpacing: "0.12em",
  color: "var(--text-muted)", marginBlockEnd: "var(--space-3)", fontWeight: 600,
};

export default function SegmentsPage() {
  return (
    <RequireScope scope="segments:read:agg" reason="Segment aggregates and projected CLV are scope-protected.">
      <SegmentsView />
    </RequireScope>
  );
}

function SegmentsView() {
  const [rfm, setRfm] = useState<any>(null);
  const [clv, setClv] = useState<any>(null);
  const [holdout, setHoldout] = useState<any>(null);
  const [err, setErr] = useState<ApiError | null>(null);

  const load = useCallback(() => {
    setErr(null);
    apiGet<any>("/api/segments/rfm").then((r) =>
      r.ok ? setRfm(r.data) : setErr(r.error));
    apiGet<any>("/api/segments/clv").then((r) =>
      r.ok ? setClv(r.data) : setErr(r.error));
    apiGet<any>("/api/segments/clv/holdout").then((r) =>
      r.ok ? setHoldout(r.data) : null);   // optional: absent until pipeline 07 runs
  }, []);

  useEffect(load, [load]);

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

      {err && (
        <div style={{ marginBlockStart: "var(--space-5)" }}>
          <ApiNotice error={err} onRetry={load} />
        </div>
      )}

      <section style={{ marginBlockStart: "var(--space-6)" }}>
        <h2 style={SECTION}>RFM segments</h2>
        {!rfm ? (
          err ? null : <div className="skeleton" style={{ blockSize: 200, borderRadius: 8 }} />
        ) : (
          <div style={{ display: "grid", gap: 8, maxInlineSize: 720 }}>
            {rfm.segments.map((s: any) => (
              <Link key={s.segment} href={`/merchandise?segment=${s.segment}`}
                    className="row-bar seg-row"
                    title={`Simulate a slate for ${s.segment.replace(/_/g, " ")}`}>
                <span className="row-label">{s.segment.replace(/_/g, " ")}</span>
                <div className="bar-track">
                  <div className="bar-fill"
                       style={{ inlineSize: `${(s.customers / maxSeg) * 100}%` }} />
                </div>
                <span className="tnum" style={{ fontSize: "var(--step--1)", textAlign: "end" }}>
                  {s.customers.toLocaleString()}
                </span>
                <span className="tnum row-extra"
                      style={{ fontSize: "var(--step--1)", textAlign: "end", color: "var(--text-faint)" }}
                      title="mean shopping occasions">
                  {s.mean_frequency}×
                </span>
              </Link>
            ))}
          </div>
        )}
      </section>

      <section style={{ marginBlockStart: "var(--space-7)" }}>
        <h2 style={SECTION}>Projected CLV distribution</h2>
        {!clv ? (
          err ? null : <div className="skeleton" style={{ blockSize: 180, borderRadius: 8 }} />
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

            {holdout && (
              <div style={{ marginBlockStart: "var(--space-6)" }}>
                <h2 style={SECTION}>
                  Holdout validation — fit on the first period, predict the second
                </h2>
                <p style={{ fontSize: "var(--step--1)", color: "var(--text-muted)",
                            maxInlineSize: "74ch", marginBlockEnd: "var(--space-4)", lineHeight: 1.6 }}>
                  Until this existed the CLV figures were <strong>unvalidated</strong> —
                  the model fitted, produced plausible numbers, and nothing had
                  checked whether they corresponded to anything. Calibrated on{" "}
                  {holdout.protocol.calibration.weeks} weeks, tested against the{" "}
                  {holdout.protocol.holdout.days} days that followed.
                </p>
                <ClvHoldout buckets={holdout.buckets} accuracy={holdout.accuracy} />
                <p style={{ fontSize: "var(--step--1)", color: "var(--text-faint)",
                            maxInlineSize: "74ch", marginBlockStart: "var(--space-3)", lineHeight: 1.6 }}>
                  {holdout.limitation}
                </p>
              </div>
            )}

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
