"use client";

import { useState } from "react";

/**
 * The cold-start curve (§9): "Stratify every metric by user history depth.
 * Report the curve. Personalisation that only works for heavy buyers is a known
 * and important limitation."
 *
 * Interactive because the SHAPE is the point: popularity slopes DOWN with more
 * history while the personalised arms slope up from zero, and you only see that
 * by comparing lines, not by reading a table of sixteen numbers.
 */

const BUCKETS = ["0", "1-2", "3-9", "10+"];
const PAD = { t: 18, r: 96, b: 40, l: 52 };

export function ColdStart({ results, width = 560, height = 300 }:
  { results: Record<string, any>; width?: number; height?: number }) {
  const [active, setActive] = useState<string | null>(null);

  const models = Object.keys(results);
  const series = models.map((m) => ({
    model: m,
    values: BUCKETS.map((b) => results[m].by_history_depth?.[b]?.["ndcg@10"] ?? 0),
  }));
  const max = Math.max(...series.flatMap((s) => s.values)) * 1.15 || 1;

  const iw = width - PAD.l - PAD.r;
  const ih = height - PAD.t - PAD.b;
  const x = (i: number) => PAD.l + (i / (BUCKETS.length - 1)) * iw;
  const y = (v: number) => PAD.t + ih - (v / max) * ih;

  return (
    <svg viewBox={`0 0 ${width} ${height}`}
         style={{ inlineSize: "100%", blockSize: "auto", maxInlineSize: width, overflow: "visible" }}
         role="img" aria-label="NDCG@10 by training history depth for each arm">
      {BUCKETS.map((b, i) => (
        <g key={b}>
          <line x1={x(i)} x2={x(i)} y1={PAD.t} y2={PAD.t + ih}
                stroke="var(--hairline)" strokeWidth={1} />
          <text x={x(i)} y={PAD.t + ih + 18} textAnchor="middle"
                fill="var(--text-faint)" fontSize={11}>{b}</text>
        </g>
      ))}
      <text x={PAD.l + iw / 2} y={height - 4} textAnchor="middle"
            fill="var(--text-muted)" fontSize={11}>
        purchases in the training window
      </text>

      {series.map((s, si) => {
        const dim = active !== null && active !== s.model;
        return (
          <g key={s.model} onMouseEnter={() => setActive(s.model)}
             onMouseLeave={() => setActive(null)} style={{ cursor: "pointer" }}>
            <polyline
              points={s.values.map((v, i) => `${x(i)},${y(v)}`).join(" ")}
              fill="none"
              stroke={si === 0 ? "var(--tail)" : "var(--signal)"}
              strokeWidth={active === s.model ? 2.5 : 1.5}
              opacity={dim ? 0.18 : si === 0 ? 0.9 : 0.35 + si * 0.16}
              style={{ transition: "opacity var(--dur-fast), stroke-width var(--dur-fast)" }}
            />
            {s.values.map((v, i) => (
              <circle key={i} cx={x(i)} cy={y(v)} r={active === s.model ? 4 : 2.5}
                      fill={si === 0 ? "var(--tail)" : "var(--signal)"}
                      opacity={dim ? 0.18 : 1} />
            ))}
            <text x={x(BUCKETS.length - 1) + 8} y={y(s.values[3]) + 4}
                  fontSize={10} fill={dim ? "var(--text-faint)" : "var(--text-muted)"}>
              {s.model.replace(/_/g, " ")}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
