"use client";

import type { ApiError } from "@/lib/api";

/** A failed request should read as a state of the page, not as a stack trace. */
export function ApiNotice({ error, onRetry }: { error: ApiError; onRetry?: () => void }) {
  const waking = error.kind === "cold_start";
  return (
    <div
      role="status"
      style={{
        padding: "var(--space-5)", borderRadius: "var(--radius-md)",
        background: waking ? "var(--surface)" : "var(--reject-dim)",
        border: `1px solid ${waking ? "var(--hairline)" : "var(--reject)"}`,
        maxInlineSize: "62ch",
      }}
    >
      <div style={{
        fontWeight: 600, fontSize: "var(--step--1)",
        color: waking ? "var(--text-muted)" : "var(--reject)",
      }}>
        {waking ? "Waking the API" : "API unavailable"}
      </div>
      <p style={{
        fontSize: "var(--step--1)", color: "var(--text-muted)",
        marginBlock: "var(--space-2) 0", lineHeight: 1.6,
      }}>
        {error.message}
      </p>
      {onRetry && (
        <button
          onClick={onRetry}
          style={{
            marginBlockStart: "var(--space-3)", padding: "6px 14px",
            fontSize: "var(--step--1)", cursor: "pointer", borderRadius: 999,
            border: "1px solid var(--hairline)", background: "transparent",
            color: "var(--text)",
          }}
        >
          Retry
        </button>
      )}
    </div>
  );
}
