"use client";

import { useTheme, type Theme } from "@/lib/theme";

const MODES: { value: Theme; label: string; glyph: string }[] = [
  { value: "light", label: "Light theme", glyph: "○" },
  { value: "system", label: "Follow system theme", glyph: "◐" },
  { value: "dark", label: "Dark theme", glyph: "●" },
];

export function ThemeToggle() {
  const { theme, setTheme, mounted } = useTheme();

  // Stable placeholder until mounted — no hydration mismatch (§12.4).
  if (!mounted) {
    return <div style={{ width: 96, height: 30 }} aria-hidden />;
  }

  return (
    <div
      role="radiogroup"
      aria-label="Colour theme"
      style={{
        display: "flex", gap: 2, padding: 2,
        border: "1px solid var(--hairline)", borderRadius: 999,
        background: "var(--surface)",
      }}
    >
      {MODES.map((m) => {
        const active = theme === m.value;
        return (
          <button
            key={m.value}
            role="radio"
            aria-checked={active}
            aria-label={m.label}
            title={m.label}
            onClick={() => setTheme(m.value)}
            style={{
              inlineSize: 30, blockSize: 26, cursor: "pointer",
              border: "none", borderRadius: 999,
              background: active ? "var(--signal-dim)" : "transparent",
              color: active ? "var(--signal)" : "var(--text-faint)",
              fontSize: 11, lineHeight: 1,
              transition: "color var(--dur-fast) var(--ease-out), background var(--dur-fast) var(--ease-out)",
            }}
          >
            {m.glyph}
          </button>
        );
      })}
    </div>
  );
}
