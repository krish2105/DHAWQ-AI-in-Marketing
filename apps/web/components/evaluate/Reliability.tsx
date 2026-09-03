"use client";

import { useState } from "react";

/**
 * The reliability curve (§10.3, "the senior signal").
 *
 * "Accuracy tells you how often the system is right. Calibration tells you
 * whether its confidence means anything. A system that is 70% accurate and
 * knows it is more useful than one that is 85% accurate and always says 99%."
 *
 * The diagonal is perfect calibration. A point ABOVE it is overconfident — it
 * claimed more than it delivered — and that is the direction §10.3 says to act
 * on by suppressing confidence rather than inflating the accuracy claim.
 */

type Bin = { bin: string; n: number; mean_confidence: number; observed_accuracy: number; gap: number };
const PAD = { t: 16, r: 16, b: 40, l: 46 };

export function Reliability({ bins, brier, ece, width = 360, height = 320 }:
  { bins: Bin[]; brier: number; ece: number; width?: number; height?: number }) {
  const [active, setActive] = useState<number | null>(null);
  const iw = width - PAD.l - PAD.r;
  const ih = height - PAD.t - PAD.b;
  const x = (v: number) => PAD.l + v * iw;
  const y = (v: number) => PAD.t + ih - v * ih;
  const maxN = Math.max(...bins.map((b) => b.n), 1);

  return (
    <div>
      <svg viewBox={`0 0 ${width} ${height}`}
           style={{ inlineSize: "100%", blockSize: "auto", maxInlineSize: width, overflow: "visible" }}
           role="img" aria-label="Stated confidence against observed accuracy">
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <g key={t}>
            <line x1={x(t)} x2={x(t)} y1={PAD.t} y2={PAD.t + ih} stroke="var(--hairline)" />
            <line x1={PAD.l} x2={PAD.l + iw} y1={y(t)} y2={y(t)} stroke="var(--hairline)" />
            <text x={x(t)} y={PAD.t + ih + 16} textAnchor="middle" fontSize={10}
                  fill="var(--text-faint)" className="tnum">{t.toFixed(2)}</text>
            <text x={PAD.l - 7} y={y(t) + 3} textAnchor="end" fontSize={10}
                  fill="var(--text-faint)" className="tnum">{t.toFixed(2)}</text>
          </g>
        ))}

        {/* perfect calibration */}
        <line x1={x(0)} y1={y(0)} x2={x(1)} y2={y(1)}
              stroke="var(--text-faint)" strokeDasharray="4 4" strokeWidth={1} />
        <text x={x(0.62)} y={y(0.68)} fontSize={9} fill="var(--text-faint)"
              transform={`rotate(-45 ${x(0.62)} ${y(0.68)})`}>perfectly calibrated</text>

        <polyline points={bins.map((b) => `${x(b.mean_confidence)},${y(b.observed_accuracy)}`).join(" ")}
                  fill="none" stroke="var(--signal)" strokeWidth={1.5} />

        {bins.map((b, i) => (
          <g key={b.bin} onMouseEnter={() => setActive(i)} onMouseLeave={() => setActive(null)}
             tabIndex={0} onFocus={() => setActive(i)} onBlur={() => setActive(null)}
             style={{ cursor: "pointer", outline: "none" }}>
            <circle cx={x(b.mean_confidence)} cy={y(b.observed_accuracy)}
                    r={5 + 7 * (b.n / maxN)}
                    fill={b.gap > 0.05 ? "var(--reject)" : "var(--signal)"}
                    opacity={active === null || active === i ? 0.9 : 0.35}
                    stroke="var(--ground)" strokeWidth={2} />
          </g>
        ))}

        <text x={PAD.l + iw / 2} y={height - 4} textAnchor="middle" fontSize={10}
              fill="var(--text-muted)">stated confidence →</text>
        <text x={11} y={PAD.t + ih / 2} textAnchor="middle" fontSize={10}
              fill="var(--text-muted)"
              transform={`rotate(-90 11 ${PAD.t + ih / 2})`}>observed accuracy →</text>
      </svg>

      <div style={{ marginBlockStart: "var(--space-3)", fontSize: "var(--step--1)" }}>
        <div className="tnum" style={{ display: "flex", gap: "var(--space-4)", flexWrap: "wrap" }}>
          <span>Brier <strong style={{ color: brier <= 0.25 ? "var(--signal)" : "var(--reject)" }}>
            {brier.toFixed(3)}</strong></span>
          <span style={{ color: "var(--text-muted)" }}>ECE {ece.toFixed(3)}</span>
        </div>
        <div style={{ color: "var(--text-faint)", marginBlockStart: 6, lineHeight: 1.55 }}>
          {active !== null
            ? `${bins[active].n} runs stated ${bins[active].mean_confidence.toFixed(3)} and were right ${(bins[active].observed_accuracy * 100).toFixed(0)}% of the time — ${bins[active].gap > 0 ? "overconfident" : "under-confident"} by ${Math.abs(bins[active].gap).toFixed(3)}.`
            : "Point size is how many runs landed in that bin. Above the diagonal is overconfident; below is under-confident, which is the safe direction."}
        </div>
      </div>
    </div>
  );
}
