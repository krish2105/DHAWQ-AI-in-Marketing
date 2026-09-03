"use client";

import { useState } from "react";
import { useAuth } from "@/lib/auth";

/**
 * The sign-in gate, and a demonstration of §13.2 rather than a wall.
 *
 * Three roles, one click each. The point is that a viewer will be REFUSED the
 * simulator and a merchandiser will not — a permission matrix you can watch
 * behave is worth more than the same matrix as a table.
 */

const BLURB: Record<string, string> = {
  viewer: "Catalogue and recommendations. No segments, no simulation.",
  analyst: "Adds evaluation artefacts and cohort aggregates. Still cannot simulate.",
  merchandiser: "Adds the slot simulator and slate approval. The full demo.",
};

export function SignIn({ reason }: { reason?: string }) {
  const { accounts, password, login } = useAuth();
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const go = async (email: string) => {
    setBusy(email); setErr(null);
    const e = await login(email, password);
    setBusy(null);
    if (e) setErr(e);
  };

  return (
    <div style={{ padding: "var(--space-6)", maxInlineSize: 720, marginInline: "auto" }}>
      <h1 style={{ fontSize: "var(--step-3)", margin: 0, letterSpacing: "-0.03em" }}>
        Choose a role
      </h1>
      <p style={{ color: "var(--text-muted)", lineHeight: 1.6, maxInlineSize: "62ch" }}>
        {reason ?? "This route is scope-protected."} DHAWQ enforces the §13.2
        permission matrix at the route boundary, so what you can reach depends on
        who you are. Pick a role and watch it refuse you.
      </p>

      {err && (
        <div style={{ marginBlock: "var(--space-4)", padding: "var(--space-3)",
                      borderRadius: "var(--radius-md)", background: "var(--reject-dim)",
                      border: "1px solid var(--reject)", fontSize: "var(--step--1)" }}>
          {err}
        </div>
      )}

      <div style={{ display: "grid", gap: "var(--space-3)",
                    marginBlockStart: "var(--space-5)" }}>
        {accounts.map((a) => (
          <button key={a.email} onClick={() => go(a.email)} disabled={busy !== null}
            style={{
              textAlign: "start", padding: "var(--space-4)", cursor: "pointer",
              borderRadius: "var(--radius-md)", background: "var(--surface)",
              border: "1px solid var(--hairline)", color: "var(--text)",
              opacity: busy && busy !== a.email ? 0.5 : 1,
              transition: "border-color var(--dur-fast) var(--ease-out)",
            }}>
            <div style={{ fontWeight: 600, textTransform: "capitalize" }}>
              {a.role}{busy === a.email ? " — signing in…" : ""}
            </div>
            <div style={{ fontSize: "var(--step--1)", color: "var(--text-muted)",
                          marginBlockStart: 4 }}>
              {BLURB[a.role] ?? ""}
            </div>
            <div className="mono" style={{ fontSize: "var(--step--1)",
                                           color: "var(--text-faint)", marginBlockStart: 6 }}>
              {a.scopes.length} scopes
            </div>
          </button>
        ))}
      </div>

      <p style={{ fontSize: "var(--step--1)", color: "var(--text-faint)",
                  marginBlockStart: "var(--space-5)", lineHeight: 1.6, maxInlineSize: "66ch" }}>
        Demo accounts stop at merchandiser. None can manage users or read the
        audit log, so a shared password cannot become an admin session. Tokens
        are httpOnly cookies with refresh rotation — nothing here is readable by
        JavaScript.
      </p>
    </div>
  );
}

/** Wrap a page that needs a scope. */
export function RequireScope({ scope, children, reason }:
  { scope: string; children: React.ReactNode; reason?: string }) {
  const { me, loading, can } = useAuth();
  if (loading) return <div className="skeleton" style={{ blockSize: 260, margin: "var(--space-6)", borderRadius: 8 }} />;
  if (!me) return <SignIn reason={reason} />;
  if (!can(scope)) {
    return (
      <div style={{ padding: "var(--space-6)", maxInlineSize: 640, marginInline: "auto" }}>
        <h1 style={{ fontSize: "var(--step-2)", margin: 0 }}>Not permitted</h1>
        <p style={{ color: "var(--text-muted)", lineHeight: 1.6 }}>
          You are signed in as <strong>{me.role}</strong>, which does not hold{" "}
          <code className="mono" style={{ color: "var(--signal)" }}>{scope}</code>.
          This is the §13.2 matrix refusing you, enforced at the route boundary
          and again at the agent&rsquo;s tool boundary.
        </p>
        <SignIn reason="Switch role to continue." />
      </div>
    );
  }
  return <>{children}</>;
}
