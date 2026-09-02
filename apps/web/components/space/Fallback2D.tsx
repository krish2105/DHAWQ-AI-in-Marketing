"use client";

import { useEffect, useRef } from "react";
import type { SpaceData } from "@/lib/space";
import { token } from "@/lib/theme";

/**
 * The 2D fallback — MANDATORY (§12.5).
 *
 * Accessibility first, WebGL failure second, debugging escape hatch third. It
 * renders IDENTICAL data: the same UMAP coordinates projected to XY, the same
 * measured dominant colours, the same canonical index. If the two views ever
 * disagree, the index invariant has broken.
 */
export function Fallback2D({
  data,
  selected,
  onSelect,
}: {
  data: SpaceData;
  selected: number;
  onSelect: (i: number) => void;
}) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const ctx = cv.getContext("2d")!;
    const dpr = Math.min(window.devicePixelRatio, 2);

    const draw = () => {
      const w = cv.clientWidth, h = cv.clientHeight;
      cv.width = w * dpr; cv.height = h * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.fillStyle = token("--scene-bg", "#080807");
      ctx.fillRect(0, 0, w, h);

      const n = data.manifest.variants[data.variant].n_instances;
      const [minX, minY] = data.manifest.extent.min;
      const [maxX, maxY] = data.manifest.extent.max;
      const sx = (w - 40) / (maxX - minX), sy = (h - 40) / (maxY - minY);
      const s = Math.min(sx, sy);

      for (let i = 0; i < n; i++) {
        const x = 20 + (data.positions[i * 3] - minX) * s;
        const y = 20 + (data.positions[i * 3 + 1] - minY) * s;
        ctx.fillStyle = `rgb(${data.colours[i * 3]},${data.colours[i * 3 + 1]},${data.colours[i * 3 + 2]})`;
        ctx.fillRect(x, y, 2.5, 2.5);
      }

      if (selected >= 0) {
        const x = 20 + (data.positions[selected * 3] - minX) * s;
        const y = 20 + (data.positions[selected * 3 + 1] - minY) * s;
        ctx.strokeStyle = token("--signal", "#0CF9E6");
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(x, y, 9, 0, Math.PI * 2); ctx.stroke();
      }
    };

    draw();
    window.addEventListener("resize", draw);
    window.addEventListener("dhawq-theme-change", draw);
    return () => {
      window.removeEventListener("resize", draw);
      window.removeEventListener("dhawq-theme-change", draw);
    };
  }, [data, selected]);

  return (
    <canvas
      ref={ref}
      onClick={(e) => {
        const cv = ref.current!;
        const r = cv.getBoundingClientRect();
        const [minX, minY] = data.manifest.extent.min;
        const [maxX, maxY] = data.manifest.extent.max;
        const s = Math.min((r.width - 40) / (maxX - minX), (r.height - 40) / (maxY - minY));
        const px = e.clientX - r.left, py = e.clientY - r.top;
        const n = data.manifest.variants[data.variant].n_instances;
        let best = -1, bestD = 400;
        for (let i = 0; i < n; i++) {
          const x = 20 + (data.positions[i * 3] - minX) * s;
          const y = 20 + (data.positions[i * 3 + 1] - minY) * s;
          const d = (x - px) ** 2 + (y - py) ** 2;
          if (d < bestD) { bestD = d; best = i; }
        }
        if (best >= 0) onSelect(best);
      }}
      style={{ inlineSize: "100%", blockSize: "100%", display: "block", cursor: "crosshair" }}
      aria-label="Two-dimensional scatter of the catalogue embedding space"
    />
  );
}
