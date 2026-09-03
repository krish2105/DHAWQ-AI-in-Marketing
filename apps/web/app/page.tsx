"use client";

import dynamic from "next/dynamic";
import { useEffect, useMemo, useState } from "react";
import { loadSpace, type SpaceData } from "@/lib/space";
import { Fallback2D } from "@/components/space/Fallback2D";

// R3F is code-split out of the initial bundle (§12.7).
const Scene = dynamic(
  () => import("@/components/space/SceneRaw").then((m) => m.SceneRaw),
  { ssr: false },
);

function webglAvailable(): boolean {
  try {
    const c = document.createElement("canvas");
    return !!c.getContext("webgl2");
  } catch {
    return false;
  }
}

export default function SpacePage() {
  const [data, setData] = useState<SpaceData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState(-1);
  const [hovered, setHovered] = useState(-1);
  const [use3D, setUse3D] = useState(true);
  const [glReady, setGlReady] = useState(false);
  const [autoFellBack, setAutoFellBack] = useState(false);
  const [neighbours, setNeighbours] = useState<{ index: number; weight: number }[]>([]);
  const [why, setWhy] = useState<any>(null);

  useEffect(() => {
    const mobile = window.matchMedia("(max-width: 767px)").matches;
    if (!webglAvailable()) setUse3D(false);
    loadSpace(mobile).then(setData).catch((e) => setError(String(e)));
  }, []);

  /*
   * PROGRESSIVE ENHANCEMENT, ENFORCED (§12.5).
   *
   * A WebGL2 context being AVAILABLE is not the same as the renderer actually
   * starting. This was not hypothetical: under react-three-fiber the canvas
   * mounted, no error was thrown, and the renderer never initialised — a
   * silently black scene that feature detection would have called healthy.
   * The scene now drives three.js directly and starts reliably, but the guard
   * stays: it costs nothing and it caught a real failure once already.
   *
   * So readiness is confirmed by the renderer itself calling back. If it has
   * not within the deadline we switch to the 2D view, which renders identical
   * data. The fallback is an accessibility requirement first and a robustness
   * hedge second; this makes it both.
   */
  useEffect(() => {
    if (!use3D || glReady) return;
    const t = setTimeout(() => {
      if (!glReady) { setUse3D(false); setAutoFellBack(true); }
    }, 2500);
    return () => clearTimeout(t);
  }, [use3D, glReady]);

  // "Why this?" — the three signals arrive SEPARATELY from the API. The
  // overlay renders them; it does not compute them.
  useEffect(() => {
    if (!data || selected < 0) { setNeighbours([]); setWhy(null); return; }
    const id = data.ids[selected];
    fetch(`/api/recs/article/${id}/why?k=8`)
      .then((r) => r.json())
      .then((j) => {
        setWhy(j);
        const idx = new Map(data.ids.map((a, i) => [a, i]));
        setNeighbours(
          (j.neighbours ?? [])
            .map((nb: any) => ({ index: idx.get(nb.article_id) ?? -1, weight: nb.visual }))
            .filter((nb: any) => nb.index >= 0),
        );
      })
      .catch(() => { setNeighbours([]); setWhy(null); });
  }, [data, selected]);

  const active = hovered >= 0 ? hovered : selected;
  const meta = useMemo(
    () => (data && active >= 0 ? data.meta[data.ids[active]] : null),
    [data, active],
  );

  if (error) {
    return (
      <div style={{ padding: "var(--space-8)", color: "var(--reject)" }}>
        <h1>Scene artefacts unavailable</h1>
        <p className="mono" style={{ fontSize: "var(--step--1)" }}>{error}</p>
      </div>
    );
  }

  return (
    <div style={{ position: "relative", blockSize: "calc(100dvh - 53px)" }}>
      {!data ? (
        <div className="skeleton" style={{ inlineSize: "100%", blockSize: "100%" }} />
      ) : use3D ? (
        <Scene
          data={data}
          selected={selected}
          neighbours={neighbours}
          onSelect={setSelected}
          onHover={setHovered}
          onReady={() => setGlReady(true)}
        />
      ) : (
        <Fallback2D data={data} selected={selected} onSelect={setSelected} />
      )}

      {/* ── overlay: title ── */}
      <div
        style={{
          position: "absolute", insetBlockStart: "var(--space-6)",
          insetInlineStart: "var(--space-6)", maxInlineSize: "38ch",
          pointerEvents: "none",
        }}
      >
        <h1 style={{ fontSize: "var(--step-3)", margin: 0, letterSpacing: "-0.03em", lineHeight: 1.05 }}>
          The shape of the catalogue
        </h1>
        <p style={{ color: "var(--text-muted)", marginBlockStart: "var(--space-3)", lineHeight: 1.6 }}>
          {data ? data.manifest.variants[data.variant].n_instances.toLocaleString() : "—"} real
          garments positioned by learned visual similarity. Clusters are real: dresses
          drift from footwear, colour gradients across regions.
        </p>
      </div>

      {/* ── overlay: controls ── */}
      <div
        style={{
          position: "absolute", insetBlockStart: "var(--space-6)",
          insetInlineEnd: "var(--space-6)", display: "flex", gap: "var(--space-2)",
        }}
      >
        {autoFellBack && (
          <span
            title="The 3D renderer did not start; showing identical data in 2D."
            style={{
              padding: "6px 12px", fontSize: "var(--step--1)",
              color: "var(--text-faint)", border: "1px solid var(--hairline)",
              borderRadius: 999, background: "var(--surface)",
            }}
          >
            2D fallback active
          </span>
        )}
        <button
          onClick={() => { setUse3D((v) => !v); setAutoFellBack(false); }}
          style={{
            padding: "6px 12px", fontSize: "var(--step--1)", cursor: "pointer",
            background: "var(--surface)", color: "var(--text-muted)",
            border: "1px solid var(--hairline)", borderRadius: 999,
          }}
        >
          {use3D ? "2D view" : "3D view"}
        </button>
      </div>

      {/* ── overlay: hover / selection card ── */}
      {meta && (
        <div
          style={{
            position: "absolute", insetBlockEnd: "var(--space-6)",
            insetInlineStart: "var(--space-6)", maxInlineSize: "46ch",
            padding: "var(--space-4)", borderRadius: "var(--radius-lg)",
            background: "color-mix(in srgb, var(--surface-raised) 92%, transparent)",
            border: "1px solid var(--hairline)", backdropFilter: "blur(10px)",
          }}
        >
          <div className="editorial">{meta.prod_name ?? data!.ids[active]}</div>
          <div
            className="mono"
            style={{ fontSize: "var(--step--1)", color: "var(--text-faint)", marginBlockStart: 4 }}
          >
            {data!.ids[active]} · {meta.product_type_name} · {meta.colour_group_name}
          </div>

          {why && selected >= 0 && (
            <div style={{ marginBlockStart: "var(--space-4)" }}>
              <div
                style={{
                  fontSize: "var(--step--1)", color: "var(--text-muted)",
                  textTransform: "uppercase", letterSpacing: "0.1em", marginBlockEnd: 6,
                }}
              >
                Why this?
              </div>
              {(why.neighbours ?? []).slice(0, 4).map((nb: any) => (
                <div
                  key={nb.article_id}
                  style={{
                    display: "grid", gridTemplateColumns: "1fr auto auto",
                    gap: "var(--space-3)", fontSize: "var(--step--1)",
                    paddingBlock: 3, alignItems: "center",
                  }}
                >
                  <span className="mono" style={{ color: "var(--text-faint)" }}>
                    {nb.article_id}
                  </span>
                  <span className="tnum" style={{ color: "var(--signal)" }} title="visual similarity">
                    {nb.visual.toFixed(3)}
                  </span>
                  <span className="tnum" style={{ color: "var(--tail)" }} title="taxonomy path score">
                    {nb.collaborative.toFixed(3)}
                  </span>
                </div>
              ))}
              <div style={{ fontSize: "var(--step--1)", color: "var(--text-faint)", marginBlockStart: 6 }}>
                visual · taxonomy-path — separated, not blended
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
