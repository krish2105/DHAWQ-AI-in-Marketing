"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { ThemeToggle } from "./ThemeToggle";
import { useAuth } from "@/lib/auth";

const ROUTES = [
  { href: "/", label: "Space" },
  { href: "/agent", label: "Agent" },
  { href: "/merchandise", label: "Merchandise" },
  { href: "/segments", label: "Segments" },
  { href: "/evaluate", label: "Evaluate" },
];

export function Nav() {
  const path = usePathname();
  const { me, logout } = useAuth();
  return (
    <header
      style={{
        position: "sticky", insetBlockStart: 0, zIndex: 50,
        display: "flex", alignItems: "center", gap: "var(--space-4)",
        paddingInline: "var(--space-4)", paddingBlock: "var(--space-3)",
        // The route list overflowed off the right edge on a phone, cutting
        // "Evaluate" in half with no way to reach it. The bar no longer wraps;
        // the LINKS scroll horizontally while the brand and the theme toggle
        // stay pinned, so every route is reachable at 360px.
        maxInlineSize: "100vw", overflow: "hidden",
        borderBlockEnd: "1px solid var(--hairline)",
        background: "color-mix(in srgb, var(--ground) 86%, transparent)",
        backdropFilter: "blur(12px)",
      }}
    >
      <Link
        href="/"
        style={{
          textDecoration: "none", color: "var(--text)", flex: "0 0 auto",
          fontWeight: 600, letterSpacing: "0.14em", fontSize: "var(--step--1)",
          whiteSpace: "nowrap",
        }}
      >
        DHAWQ<span style={{ color: "var(--text-faint)", marginInlineStart: 8 }}>ذوق</span>
      </Link>

      <nav
        aria-label="Primary"
        style={{
          display: "flex", gap: "var(--space-4)", overflowX: "auto",
          scrollbarWidth: "none", WebkitOverflowScrolling: "touch",
          minInlineSize: 0, flex: "1 1 auto",
        }}
      >
        {ROUTES.map((r) => {
          const active = path === r.href;
          return (
            <Link
              key={r.href}
              href={r.href}
              aria-current={active ? "page" : undefined}
              style={{
                textDecoration: "none", fontSize: "var(--step--1)",
                whiteSpace: "nowrap",
                color: active ? "var(--text)" : "var(--text-muted)",
                borderBlockEnd: `1px solid ${active ? "var(--signal)" : "transparent"}`,
                paddingBlockEnd: 2,
                transition: "color var(--dur-fast) var(--ease-out)",
              }}
            >
              {r.label}
            </Link>
          );
        })}
      </nav>

      <div style={{ flex: "0 0 auto", display: "flex", alignItems: "center",
                    gap: "var(--space-3)" }}>
        {me && (
          <button
            onClick={logout}
            title={`Signed in as ${me.role} · ${me.scopes.length} scopes · click to sign out`}
            style={{
              display: "flex", alignItems: "center", gap: 6, cursor: "pointer",
              padding: "4px 10px", borderRadius: 999, fontSize: "var(--step--1)",
              border: "1px solid var(--hairline)", background: "var(--surface)",
              color: "var(--text-muted)", whiteSpace: "nowrap",
            }}
          >
            <span style={{ inlineSize: 6, blockSize: 6, borderRadius: 999,
                           background: "var(--signal)" }} />
            {me.role}
          </button>
        )}
        <ThemeToggle />
      </div>
    </header>
  );
}
