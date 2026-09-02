"use client";

import { useEffect, useState } from "react";

/* Merchandise view — the policy the whole constraint layer enforces, readable
   in full. Corpus C is LOADED, not retrieved (§8.2), so there is nothing to
   search here: the entire document is the artefact. */
export default function MerchandisePage() {
  const [policy, setPolicy] = useState<any>(null);

  useEffect(() => {
    fetch("/api/merchandise/policy").then((r) => r.json()).then(setPolicy).catch(() => {});
  }, []);

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
