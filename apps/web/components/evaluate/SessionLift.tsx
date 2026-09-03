"use client";

import { useState } from "react";

/**
 * THE MARKETING HEADLINE, and it is not the one this project went looking for.
 *
 * Personalisation does not lift projected revenue at this horizon — the
 * confidence intervals say so. What it buys is REACH: hundreds of times more of
 * the catalogue receives exposure for a revenue difference the data cannot
 * distinguish from zero. For a retailer holding inventory that is the trade,
 * and dead stock is a cost a bestseller page never addresses.
 *
 * The CI bar is the point of the chart. A bar crossing zero means "we cannot
 * tell these apart", which is a different claim from "these are the same" and a
 * very different one from "this is worse".
 */

type Arm = {
  model: string;
  projected_lift_pct: number;
  ci95: [number, number];
  significant: boolean;
  coverage: number;
  baseline_coverage: number;
  reach_multiple: number;
  hit_rate: number;
  baseline_hit_rate: number;
};

const PAD = { t: 16, r: 20, b: 34, l: 128 };

export function SessionLift({ results, width = 620 }:
  { results: Record<string, Arm>; width?: number }) {
  const [active, setActive] = useState<string | null>(null);
  const arms = Object.values(results);
  const height = PAD.t + PAD.b + arms.length * 46;
  const iw = width - PAD.l - PAD.r;

  const lo = Math.min(...arms.map((a) => a.ci95[0]), 0) * 1.1;
  const hi = Math.max(...arms.map((a) => a.ci95[1]), 0) * 1.1;
  const x = (v: number) => PAD.l + ((v - lo) / (hi - lo)) * iw;

  return (
    <div>
      <svg viewBox={`0 0 ${width} ${height}`}
           style={{ inlineSize: "100%", blockSize: "auto", maxInlineSize: width, overflow: "visible" }}
           role="img" aria-label="Projected revenue lift per session with 95% confidence intervals">
        {/* zero line — crossing it is the whole story */}
        <line x1={x(0)} x2={x(0)} y1={PAD.t - 6} y2={height - PAD.b + 4}
              stroke="var(--text-faint)" strokeDasharray="3 3" />
        <text x={x(0)} y={height - PAD.b + 18} textAnchor="middle" fontSize={10}
              fill="var(--text-faint)">no difference</text>

        {arms.map((a, i) => {
          const y = PAD.t + i * 46 + 14;
          const crosses = a.ci95[0] <= 0 && a.ci95[1] >= 0;
          const tone = crosses ? "var(--tail)" : "var(--reject)";
          return (
            <g key={a.model} onMouseEnter={() => setActive(a.model)}
               onMouseLeave={() => setActive(null)} style={{ cursor: "pointer" }}
               opacity={active && active !== a.model ? 0.4 : 1}>
              <text x={PAD.l - 10} y={y + 4} textAnchor="end" fontSize={11}
                    fill="var(--text-muted)">{a.model.replace(/_/g, " ")}</text>
              <line x1={x(a.ci95[0])} x2={x(a.ci95[1])} y1={y} y2={y}
                    stroke={tone} strokeWidth={3} strokeLinecap="round" opacity={0.55} />
              {[a.ci95[0], a.ci95[1]].map((v, j) => (
                <line key={j} x1={x(v)} x2={x(v)} y1={y - 5} y2={y + 5}
                      stroke={tone} strokeWidth={1.5} />
              ))}
              <circle cx={x(a.projected_lift_pct)} cy={y} r={5} fill={tone}
                      stroke="var(--ground)" strokeWidth={2} />
              <text x={x(a.ci95[1]) + 8} y={y + 4} fontSize={10}
                    className="tnum" fill="var(--text-faint)">
                {a.reach_multiple}× reach
              </text>
            </g>
          );
        })}
      </svg>

      <div style={{ marginBlockStart: "var(--space-3)", fontSize: "var(--step--1)",
                    color: "var(--text-faint)", lineHeight: 1.6, minBlockSize: 48 }}>
        {active && results[active] ? (
          <>
            <strong style={{ color: "var(--text)" }}>{active.replace(/_/g, " ")}</strong>{" "}
            <span className="tnum">
              {results[active].projected_lift_pct.toFixed(1)}% lift, 95% CI [
              {results[active].ci95[0].toFixed(1)}, {results[active].ci95[1].toFixed(1)}]
            </span>{" "}
            — {results[active].significant
              ? "the interval excludes zero, so the difference is real"
              : "the interval INCLUDES zero: the data cannot distinguish this from no difference"}.
            Reach {(results[active].coverage * 100).toFixed(1)}% of the catalogue
            against {(results[active].baseline_coverage * 100).toFixed(2)}% for one
            bestseller page.
          </>
        ) : (
          "Bars are 95% bootstrap intervals over sessions. A bar crossing the dashed line means the data cannot tell that arm apart from showing everyone the bestsellers — which is a different claim from saying it is worse."
        )}
      </div>
    </div>
  );
}


/** What one shared page is worth as the cohort widens. */
export function GranularityCurve({ curve, width = 480, height = 220 }:
  { curve: { cohort_size: number; pct_of_personalised: number }[]; width?: number; height?: number }) {
  const P = { t: 14, r: 16, b: 38, l: 44 };
  const iw = width - P.l - P.r, ih = height - P.t - P.b;
  const xs = curve.map((c) => Math.log10(Math.max(c.cohort_size, 1)));
  const maxX = Math.max(...xs, 1);
  const x = (v: number) => P.l + (Math.log10(Math.max(v, 1)) / maxX) * iw;
  const y = (v: number) => P.t + ih - (v / 100) * ih;

  return (
    <svg viewBox={`0 0 ${width} ${height}`}
         style={{ inlineSize: "100%", blockSize: "auto", maxInlineSize: width, overflow: "visible" }}
         role="img" aria-label="Value of one shared page as cohort size grows">
      {[0, 50, 100].map((t) => (
        <g key={t}>
          <line x1={P.l} x2={P.l + iw} y1={y(t)} y2={y(t)} stroke="var(--hairline)" />
          <text x={P.l - 6} y={y(t) + 3} textAnchor="end" fontSize={10}
                fill="var(--text-faint)" className="tnum">{t}%</text>
        </g>
      ))}
      <polyline points={curve.map((c) => `${x(c.cohort_size)},${y(c.pct_of_personalised)}`).join(" ")}
                fill="none" stroke="var(--signal)" strokeWidth={2} />
      {curve.map((c) => (
        <g key={c.cohort_size}>
          <title>{`cohort of ${c.cohort_size}: ${c.pct_of_personalised}% of a fully personalised page`}</title>
          <circle cx={x(c.cohort_size)} cy={y(c.pct_of_personalised)} r={4}
                  fill="var(--signal)" stroke="var(--ground)" strokeWidth={2} />
          <text x={x(c.cohort_size)} y={height - P.b + 15} textAnchor="middle"
                fontSize={10} fill="var(--text-faint)" className="tnum">{c.cohort_size}</text>
        </g>
      ))}
      <text x={P.l + iw / 2} y={height - 4} textAnchor="middle" fontSize={10}
            fill="var(--text-muted)">customers sharing one page (log scale)</text>
    </svg>
  );
}
