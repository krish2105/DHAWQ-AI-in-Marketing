"use client";

import { Suspense, useCallback, useEffect, useState } from "react";
import { apiGet, type ApiError } from "@/lib/api";
import { ApiNotice } from "@/components/ui/ApiNotice";
import { RequireScope } from "@/components/ui/SignIn";
import { useSearchParams } from "next/navigation";

/* Merchandise view — the policy the whole constraint layer enforces, readable
   in full. Corpus C is LOADED, not retrieved (§8.2), so there is nothing to
   search here: the entire document is the artefact. */
const SEGMENTS = ["champions", "loyal", "at_risk", "hibernating", "big_spenders"];

function Delta({ label, value, hint }: { label: string; value: number; hint?: string }) {
  const tone = value > 0 ? "var(--signal)" : value < 0 ? "var(--reject)" : "var(--text-muted)";
  return (
    <div title={hint}>
      <div className="tnum" style={{ fontSize: "var(--step-2)", color: tone, lineHeight: 1 }}>
        {value > 0 ? "+" : ""}{value.toFixed(1)}%
      </div>
      <div style={{ fontSize: "var(--step--1)", color: "var(--text-muted)", marginBlockStart: 4 }}>
        {label}
      </div>
    </div>
  );
}

function Slate({ title, side }: { title: string; side: any }) {
  // Each slot names the article it holds, so the comparison is inspectable
  // rather than three grids of numbered boxes.
  return (
    <div>
      <div style={{ fontSize: "var(--step--1)", color: "var(--text-muted)", marginBlockEnd: 8 }}>
        {title} · <span className="tnum">{(side.long_tail_share * 100).toFixed(0)}%</span> long-tail
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(6, 1fr)", gap: 4 }}>
        {side.slate.map((a: any) => (
          <div key={a.article_id} className="mono slot"
               title={`${a.prod_name ?? a.article_id}\n${a.product_type_name ?? ""} · ${a.colour_group_name ?? ""}\n${a.is_long_tail ? "long tail" : "head"}`}
               style={{
                 aspectRatio: "1", borderRadius: 4, fontSize: 9,
                 display: "grid", placeItems: "center", color: "var(--text-faint)",
                 background: "var(--surface)",
                 border: `1px solid ${a.is_long_tail ? "var(--tail)" : "var(--hairline)"}`,
               }}>
            {a.position}
          </div>
        ))}
      </div>
    </div>
  );
}

/* useSearchParams opts a route out of static prerendering unless it sits behind
   a Suspense boundary — Next builds the shell, then fills the param-dependent
   part on the client. Without it the whole page fails to prerender. */
export default function MerchandisePage() {
  return (
    <RequireScope scope="merch:simulate" reason="The slot simulator is scope-protected.">
      <Suspense fallback={<div className="skeleton" style={{ blockSize: 320, margin: "var(--space-6)", borderRadius: 8 }} />}>
        <MerchandiseView />
      </Suspense>
    </RequireScope>
  );
}

function MerchandiseView() {
  const [policy, setPolicy] = useState<any>(null);
  const [sim, setSim] = useState<any>(null);
  const params = useSearchParams();
  const [segment, setSegment] = useState(params.get("segment") ?? "champions");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<ApiError | null>(null);

  const load = useCallback(() => {
    setErr(null);
    apiGet<any>("/api/merchandise/policy").then((r) =>
      r.ok ? setPolicy(r.data) : setErr(r.error));
  }, []);

  useEffect(load, [load]);

  useEffect(() => {
    setBusy(true);
    apiGet<any>(`/api/merchandise/simulate?k=12&segment=${segment}`)
      .then((r) => {
        if (r.ok) { setSim(r.data); setErr(null); }
        else { setSim(null); setErr(r.error); }
      })
      .finally(() => setBusy(false));
  }, [segment]);

  return (
    <div style={{ padding: "var(--space-6)", maxInlineSize: 900, marginInline: "auto" }}>
      <h1 style={{ fontSize: "var(--step-3)", margin: 0, letterSpacing: "-0.03em" }}>
        Merchandising policy
      </h1>
      <p style={{ color: "var(--text-muted)", lineHeight: 1.6, maxInlineSize: "66ch" }}>
        Corpus C. Loaded whole into the critic&rsquo;s context — not chunked, not
        embedded, not retrieved. A critic that reads the entire policy every time
        cannot miss a rule because a chunk failed to rank.
      </p>

      <section style={{ marginBlock: "var(--space-6)" }}>
        <h2 style={{
          fontSize: "var(--step--1)", textTransform: "uppercase",
          letterSpacing: "0.12em", color: "var(--text-muted)", fontWeight: 600,
        }}>
          Slot simulator — your model against the bestseller page
        </h2>

        <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBlock: "var(--space-3)" }}>
          {SEGMENTS.map((sg) => (
            <button key={sg} onClick={() => setSegment(sg)} style={{
              padding: "4px 10px", fontSize: "var(--step--1)", cursor: "pointer",
              border: "1px solid var(--hairline)", borderRadius: 999,
              background: segment === sg ? "var(--signal-dim)" : "transparent",
              color: segment === sg ? "var(--signal)" : "var(--text-muted)",
            }}>{sg.replace(/_/g, " ")}</button>
          ))}
        </div>

        {err ? (
          <ApiNotice error={err} onRetry={load} />
        ) : busy || !sim?.decomposition ? (
          <div className="skeleton" style={{ blockSize: 210, borderRadius: 8 }} />
        ) : (
          <>
            <div style={{ display: "flex", gap: "var(--space-6)", flexWrap: "wrap", marginBlockEnd: "var(--space-5)" }}>
              <Delta label="personalisation effect"
                     value={sim.decomposition.personalisation_effect.projected_lift_pct}
                     hint="unconstrained model vs the bestseller page — neither carrying a quota" />
              <Delta label="cost of the long-tail quota"
                     value={sim.decomposition.quota_cost.projected_lift_pct}
                     hint="the same model with and without POL-LT-01" />
              <Delta label="combined (what ships)"
                     value={sim.decomposition.combined.projected_lift_pct} />
              <div>
                <div className="tnum" style={{ fontSize: "var(--step-2)", color: "var(--tail)", lineHeight: 1 }}>
                  +{sim.decomposition.combined.coverage_cost_pp.toFixed(1)}pp
                </div>
                <div style={{ fontSize: "var(--step--1)", color: "var(--text-muted)", marginBlockStart: 4 }}>
                  long-tail exposure bought
                </div>
              </div>
            </div>

            <div style={{ display: "grid", gap: "var(--space-5)", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))" }}>
              <Slate title="Your model (with quota)" side={sim.model} />
              <Slate title="Same model, no quota" side={sim.unconstrained} />
              <Slate title="Bestseller baseline" side={sim.baseline} />
            </div>

            <p style={{ fontSize: "var(--step--1)", color: "var(--text-faint)", maxInlineSize: "80ch", marginBlockStart: "var(--space-4)", lineHeight: 1.65 }}>
              {sim.the_finding}
            </p>
            <p style={{ fontSize: "var(--step--1)", color: "var(--text-faint)", maxInlineSize: "80ch", marginBlockStart: "var(--space-3)", lineHeight: 1.65 }}>
              {sim.known_bias}
            </p>
          </>
        )}
      </section>

      {policy && (
        <>
          <div style={{
            display: "flex", gap: "var(--space-6)", flexWrap: "wrap",
            marginBlock: "var(--space-5)", padding: "var(--space-4)",
            background: "var(--surface)", borderRadius: "var(--radius-md)",
          }}>
            {[
              ["version", policy.version],
              ["rules", policy.manifest.counts.rules],
              ["unsettled thresholds", policy.manifest.counts.unsettled_thresholds],
              ["est. tokens", policy.manifest.size.estimated_tokens.toLocaleString()],
            ].map(([k, v]: any) => (
              <div key={k}>
                <div className="tnum" style={{ fontSize: "var(--step-1)", color: "var(--signal)" }}>{v}</div>
                <div style={{ fontSize: "var(--step--1)", color: "var(--text-muted)" }}>{k}</div>
              </div>
            ))}
          </div>

          <pre style={{
            whiteSpace: "pre-wrap", fontSize: "var(--step--1)", lineHeight: 1.7,
            color: "var(--text-muted)", background: "var(--surface)",
            padding: "var(--space-5)", borderRadius: "var(--radius-md)",
            maxBlockSize: "70vh", overflowY: "auto",
            border: "1px solid var(--hairline)",
          }}>{policy.document}</pre>
        </>
      )}
    </div>
  );
}
