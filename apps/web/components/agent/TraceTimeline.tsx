"use client";

/* §7.9 rendered. PLAN.md §13 cut the OTel collector and kept the span model on
   the argument that "the console is a better demo than a Jaeger UI". This is
   the component that has to make that true.

   The thing worth looking at is NOT the durations — it is the reasoning
   events: plan, action, observation, next decision. A flat log shows that the
   critic ran. This shows what it saw and where it sent the run next. */

import { useEffect, useState } from "react";
import { apiGet } from "@/lib/api";

type Span = {
  span_id: string; name: string; kind: string; duration_ms: number;
  status: string; attributes: Record<string, unknown>;
  events: { name: string; [k: string]: unknown }[];
  children: Span[];
};

const KIND_LABEL: Record<string, string> = {
  run: "RUN", node: "NODE", tool: "TOOL", model: "MODEL",
  decision: "BRANCH", gate: "GATE",
};

function Row({ span, depth, total }: { span: Span; depth: number; total: number }) {
  const [open, setOpen] = useState(depth < 2);
  const reasoning = span.events.filter((e) => e.name === "reasoning");
  const pct = total > 0 ? Math.max(0.6, (span.duration_ms / total) * 100) : 0;

  return (
    <div>
      <button
        onClick={() => setOpen((v) => !v)}
        style={{
          inlineSize: "100%", display: "grid",
          gridTemplateColumns: "minmax(0,1fr) 64px", gap: "var(--space-3)",
          alignItems: "center", background: "transparent", border: "none",
          borderBlockEnd: "1px solid var(--hairline)", padding: "10px 0",
          paddingInlineStart: depth * 16, cursor: "pointer", textAlign: "start",
          color: "var(--text)", font: "inherit",
        }}
      >
        <span style={{ minInlineSize: 0 }}>
          <span style={{ display: "flex", gap: 8, alignItems: "baseline" }}>
            <span style={{
              fontFamily: "var(--font-mono)", fontSize: 10,
              color: span.status === "error" ? "var(--reject)" : "var(--text-faint)",
              letterSpacing: "0.08em", inlineSize: 46, flexShrink: 0,
            }}>
              {KIND_LABEL[span.kind] ?? span.kind}
            </span>
            <span style={{ fontSize: "var(--step-0)", overflow: "hidden",
                           textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {span.name}
            </span>
            {reasoning.length > 0 && (
              <span style={{
                fontFamily: "var(--font-mono)", fontSize: 10,
                color: "var(--signal)", flexShrink: 0,
              }}>why</span>
            )}
          </span>
          {/* Duration as a bar, because relative cost is the only thing the
              number is used for and a bar reads faster than nine decimals. */}
          <span style={{
            display: "block", blockSize: 2, marginBlockStart: 6,
            inlineSize: `${pct}%`, borderRadius: 99,
            background: span.status === "error" ? "var(--reject)" : "var(--signal-dim)",
          }} />
        </span>
        <span style={{
          fontFamily: "var(--font-mono)", fontSize: 11, textAlign: "end",
          color: "var(--text-muted)", fontVariantNumeric: "tabular-nums",
        }}>
          {span.duration_ms < 1 ? "<1" : Math.round(span.duration_ms)}ms
        </span>
      </button>

      {open && reasoning.map((r, i) => (
        <dl key={i} style={{
          margin: 0, marginInlineStart: depth * 16 + 46,
          padding: "10px 0 12px", display: "grid",
          gridTemplateColumns: "76px minmax(0,1fr)", gap: "4px var(--space-3)",
          borderBlockEnd: "1px solid var(--hairline)", fontSize: "var(--step--1)",
        }}>
          {(["plan", "action", "observation", "next_decision"] as const).map((k) => (
            <Fragmentish key={k} label={k.replace("_", " ")} value={String(r[k] ?? "—")} />
          ))}
        </dl>
      ))}

      {open && span.children.map((c) => (
        <Row key={c.span_id} span={c} depth={depth + 1} total={total} />
      ))}
    </div>
  );
}

function Fragmentish({ label, value }: { label: string; value: string }) {
  return (
    <>
      <dt style={{ color: "var(--text-faint)", fontFamily: "var(--font-mono)",
                   fontSize: 10, letterSpacing: "0.06em", textTransform: "uppercase" }}>
        {label}
      </dt>
      <dd style={{ margin: 0, color: "var(--text-muted)", overflowWrap: "anywhere" }}>
        {value}
      </dd>
    </>
  );
}

export default function TraceTimeline({ runId }: { runId: string | null }) {
  const [spans, setSpans] = useState<Span[] | null>(null);
  const [source, setSource] = useState<string>("");

  useEffect(() => {
    if (!runId) { setSpans(null); return; }
    let live = true;
    apiGet<{ spans: Span[]; source: string }>(`/api/agent/runs/${runId}/trace`)
      .then((r) => { if (live && r.ok) { setSpans(r.data.spans); setSource(r.data.source); } });
    return () => { live = false; };
  }, [runId]);

  if (!runId || !spans || spans.length === 0) return null;
  const total = spans[0]?.duration_ms ?? 1;

  return (
    <section style={{ marginBlockStart: "var(--space-6)" }}>
      <h2 style={{ fontSize: "var(--step-1)", margin: "0 0 4px", letterSpacing: "-0.02em" }}>
        Reasoning trace
      </h2>
      <p style={{ margin: "0 0 var(--space-3)", color: "var(--text-muted)",
                  fontSize: "var(--step--1)", maxInlineSize: "62ch" }}>
        Nested spans, OpenTelemetry GenAI attribute conventions, no collector.
        The rows marked <span style={{ color: "var(--signal)" }}>why</span> carry
        the plan, the action, what came back and where the run went next — a flat
        log would show that the critic ran, not what it decided.
        {source === "stored" && " Replayed from storage."}
      </p>
      <div style={{ borderBlockStart: "1px solid var(--hairline)" }}>
        {spans.map((s) => <Row key={s.span_id} span={s} depth={0} total={total} />)}
      </div>
    </section>
  );
}
