"use client";

import { useState } from "react";

/**
 * THE ACCURACY–COVERAGE FRONTIER (§9, §12.6).
 *
 * "The tension is the finding. Accuracy and coverage trade off. Plot the
 * frontier." A table lets a reader take the NDCG column and leave the coverage
 * column behind, which is exactly the misreading the whole section exists to
 * prevent. On a scatter you cannot look at one axis without the other.
 *
 * Inline SVG rather than a chart library: it reads theme tokens directly, so
 * it rebinds on theme change like everything else, and it costs no bundle.
 */

export type FrontierPoint = {
  model: string;
  "ndcg@10": number;
  coverage: number;
  gini: number;
  long_tail_exposure: number;
  popularity_lift?: number;
};

const PAD = { t: 18, r: 18, b: 44, l: 52 };

export function Frontier({ points, width = 560, height = 340 }:
  { points: FrontierPoint[]; width?: number; height?: number }) {
  const [active, setActive] = useState<string | null>(null);

  const iw = width - PAD.l - PAD.r;
  const ih = height - PAD.t - PAD.b;
  const maxN = Math.max(...points.map((p) => p["ndcg@10"])) * 1.15;
  const maxC = 1;

  const x = (c: number) => PAD.l + (c / maxC) * iw;
  const y = (n: number) => PAD.t + ih - (n / maxN) * ih;

  // Pareto front: a model is on it if nothing beats it on BOTH axes. That is
  // the set a merchandiser actually chooses between; everything else is
  // dominated and should not be in the conversation.
  const onFront = (p: FrontierPoint) =>
    !points.some((q) => q !== p && q["ndcg@10"] >= p["ndcg@10"] && q.coverage >= p.coverage);

  const front = points.filter(onFront).sort((a, b) => a.coverage - b.coverage);
  const shown = points.find((p) => p.model === active);

  return (
    <div>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        style={{ inlineSize: "100%", blockSize: "auto", maxInlineSize: width, overflow: "visible" }}
        role="img"
        aria-label="Accuracy against catalogue coverage for each recommender arm"
      >
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <g key={t}>
            <line x1={x(t)} x2={x(t)} y1={PAD.t} y2={PAD.t + ih}
                  stroke="var(--hairline)" strokeWidth={1} />
            <text x={x(t)} y={PAD.t + ih + 18} textAnchor="middle"
                  fill="var(--text-faint)" fontSize={11} className="tnum">
              {(t * 100).toFixed(0)}%
            </text>
          </g>
        ))}
        {[0, 0.5, 1].map((t) => (
          <g key={t}>
            <line x1={PAD.l} x2={PAD.l + iw} y1={y(maxN * t)} y2={y(maxN * t)}
                  stroke="var(--hairline)" strokeWidth={1} />
            <text x={PAD.l - 8} y={y(maxN * t) + 4} textAnchor="end"
                  fill="var(--text-faint)" fontSize={11} className="tnum">
              {(maxN * t).toFixed(3)}
            </text>
          </g>
        ))}

        {/* the frontier itself */}
        <polyline
          points={front.map((p) => `${x(p.coverage)},${y(p["ndcg@10"])}`).join(" ")}
          fill="none" stroke="var(--signal)" strokeWidth={1.5}
          strokeDasharray="4 4" opacity={0.55}
        />

        {points.map((p) => {
          const isActive = active === p.model;
          const dominated = !onFront(p);
          return (
            <g key={p.model}
               onMouseEnter={() => setActive(p.model)}
               onMouseLeave={() => setActive(null)}
               onFocus={() => setActive(p.model)}
               onBlur={() => setActive(null)}
               tabIndex={0}
               style={{ cursor: "pointer", outline: "none" }}>
              <circle cx={x(p.coverage)} cy={y(p["ndcg@10"])} r={isActive ? 9 : 6}
                      fill={dominated ? "var(--tail)" : "var(--signal)"}
                      opacity={dominated ? 0.55 : 1}
                      stroke="var(--ground)" strokeWidth={2}
                      style={{ transition: "r var(--dur-fast) var(--ease-out)" }} />
              <text x={x(p.coverage)} y={y(p["ndcg@10"]) - 14} textAnchor="middle"
                    fontSize={10} fill={isActive ? "var(--text)" : "var(--text-muted)"}>
                {p.model.replace(/_/g, " ")}
              </text>
            </g>
          );
        })}

        <text x={PAD.l + iw / 2} y={height - 6} textAnchor="middle"
              fill="var(--text-muted)" fontSize={11}>
          catalogue coverage →
        </text>
        <text x={14} y={PAD.t + ih / 2} textAnchor="middle" fontSize={11}
              fill="var(--text-muted)" transform={`rotate(-90 14 ${PAD.t + ih / 2})`}>
          NDCG@10 →
        </text>
      </svg>

      <div style={{
        marginBlockStart: "var(--space-3)", minBlockSize: 62,
        padding: "var(--space-3)", borderRadius: "var(--radius-md)",
        background: shown ? "var(--surface)" : "transparent",
        border: `1px solid ${shown ? "var(--hairline)" : "transparent"}`,
      }}>
        {shown ? (
          <>
            <div style={{ fontWeight: 600, fontSize: "var(--step--1)" }}>
              {shown.model.replace(/_/g, " ")}
              {!onFront(shown) && (
                <span style={{ color: "var(--tail)", fontWeight: 400 }}>
                  {" "}· dominated — another arm beats it on both axes
                </span>
              )}
            </div>
            <div className="tnum" style={{
              fontSize: "var(--step--1)", color: "var(--text-muted)",
              marginBlockStart: 4, display: "flex", gap: "var(--space-4)", flexWrap: "wrap",
            }}>
              <span>NDCG@10 {shown["ndcg@10"].toFixed(4)}</span>
              <span>coverage {(shown.coverage * 100).toFixed(1)}%</span>
              <span>Gini {shown.gini.toFixed(3)}</span>
              <span>long-tail {(shown.long_tail_exposure * 100).toFixed(1)}%</span>
              {shown.popularity_lift != null && <span>pop. lift {shown.popularity_lift.toFixed(1)}×</span>}
            </div>
          </>
        ) : (
          <div style={{ fontSize: "var(--step--1)", color: "var(--text-faint)" }}>
            Hover or focus a point. The dashed line is the Pareto front — the arms
            nothing else beats on both axes. Everything off it is dominated.
          </div>
        )}
      </div>
    </div>
  );
}
