"use client";

/**
 * §11's holdout validation: "fit on the first period, predict the second, plot
 * predicted vs actual."
 *
 * Paired bars per calibration-frequency bucket rather than a scatter, because
 * the question is not "is there a correlation" — it is "for customers who
 * looked like THIS, did the model get them right?" Systematic under- or
 * over-prediction shows up immediately as one bar consistently taller.
 */

type Bucket = { calibration_frequency: string; n: number; mean_predicted: number; mean_actual: number };

export function ClvHoldout({ buckets, accuracy, width = 460, height = 250 }:
  { buckets: Bucket[]; accuracy: any; width?: number; height?: number }) {
  const PAD = { t: 16, r: 12, b: 44, l: 40 };
  const iw = width - PAD.l - PAD.r;
  const ih = height - PAD.t - PAD.b;
  const max = Math.max(...buckets.flatMap((b) => [b.mean_predicted, b.mean_actual])) * 1.15 || 1;
  const bw = iw / buckets.length;

  return (
    <div>
      <svg viewBox={`0 0 ${width} ${height}`}
           style={{ inlineSize: "100%", blockSize: "auto", maxInlineSize: width, overflow: "visible" }}
           role="img" aria-label="Predicted against actual holdout purchases by calibration frequency">
        {[0, 0.5, 1].map((t) => (
          <g key={t}>
            <line x1={PAD.l} x2={PAD.l + iw} y1={PAD.t + ih - t * ih} y2={PAD.t + ih - t * ih}
                  stroke="var(--hairline)" />
            <text x={PAD.l - 6} y={PAD.t + ih - t * ih + 3} textAnchor="end" fontSize={10}
                  fill="var(--text-faint)" className="tnum">{(max * t).toFixed(1)}</text>
          </g>
        ))}
        {buckets.map((b, i) => {
          const x0 = PAD.l + i * bw;
          const hp = (b.mean_predicted / max) * ih;
          const ha = (b.mean_actual / max) * ih;
          return (
            <g key={b.calibration_frequency}>
              <title>{`${b.n.toLocaleString()} customers · predicted ${b.mean_predicted.toFixed(3)} · actual ${b.mean_actual.toFixed(3)}`}</title>
              <rect x={x0 + bw * 0.16} y={PAD.t + ih - hp} width={bw * 0.3} height={hp}
                    fill="var(--signal)" opacity={0.85} rx={2} />
              <rect x={x0 + bw * 0.52} y={PAD.t + ih - ha} width={bw * 0.3} height={ha}
                    fill="var(--tail)" opacity={0.85} rx={2} />
              <text x={x0 + bw / 2} y={PAD.t + ih + 15} textAnchor="middle" fontSize={10}
                    fill="var(--text-faint)">{b.calibration_frequency}</text>
            </g>
          );
        })}
        <text x={PAD.l + iw / 2} y={height - 6} textAnchor="middle" fontSize={10}
              fill="var(--text-muted)">purchases during calibration</text>
      </svg>

      <div style={{ display: "flex", gap: "var(--space-4)", fontSize: "var(--step--1)",
                    marginBlockStart: 6, flexWrap: "wrap" }}>
        <span style={{ color: "var(--signal)" }}>■ predicted</span>
        <span style={{ color: "var(--tail)" }}>■ actual</span>
        <span className="tnum" style={{ color: "var(--text-muted)" }}>
          MAE {accuracy.mae_purchases.toFixed(3)} vs naive {accuracy.naive_baseline_mae.toFixed(3)}
          {accuracy.beats_naive ? " — beats naive" : " — DOES NOT beat naive"}
        </span>
      </div>
    </div>
  );
}
