"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { ThemeToggle } from "./ThemeToggle";

const ROUTES = [
  { href: "/", label: "Space" },
  { href: "/agent", label: "Agent" },
  { href: "/merchandise", label: "Merchandise" },
  { href: "/segments", label: "Segments" },
  { href: "/evaluate", label: "Evaluate" },
];

export function Nav() {
  const path = usePathname();
  return (
    <header
      style={{
        position: "sticky", insetBlockStart: 0, zIndex: 50,
        display: "flex", alignItems: "center", gap: "var(--space-5)",
        paddingInline: "var(--space-5)", paddingBlock: "var(--space-3)",
        borderBlockEnd: "1px solid var(--hairline)",
        background: "color-mix(in srgb, var(--ground) 86%, transparent)",
        backdropFilter: "blur(12px)",
      }}
    >
      <Link
        href="/"
        style={{
          textDecoration: "none", color: "var(--text)",
          fontWeight: 600, letterSpacing: "0.14em", fontSize: "var(--step--1)",
        }}
      >
        DHAWQ<span style={{ color: "var(--text-faint)", marginInlineStart: 8 }}>ذوق</span>
      </Link>

      <nav aria-label="Primary" style={{ display: "flex", gap: "var(--space-4)" }}>
        {ROUTES.map((r) => {
          const active = path === r.href;
          return (
            <Link
              key={r.href}
              href={r.href}
              aria-current={active ? "page" : undefined}
              style={{
                textDecoration: "none", fontSize: "var(--step--1)",
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

      <div style={{ marginInlineStart: "auto" }}>
        <ThemeToggle />
      </div>
    </header>
  );
}
