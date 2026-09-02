"use client";

import { useCallback, useEffect, useState } from "react";

export type Theme = "dark" | "light" | "system";
const KEY = "dhawq-theme";

/*
 * Three states, system default (§12.4). The toggle renders a stable
 * placeholder until mounted so there is no server/client mismatch, and the
 * R3F scene subscribes to `dhawq-theme-change` to rebind its tokens.
 */
export function useTheme() {
  const [theme, setThemeState] = useState<Theme>("system");
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setThemeState((localStorage.getItem(KEY) as Theme) ?? "system");
    setMounted(true);
  }, []);

  const apply = useCallback((next: Theme) => {
    const dark =
      next === "dark" ||
      (next === "system" &&
        window.matchMedia("(prefers-color-scheme: dark)").matches);
    document.documentElement.classList.toggle("dark", dark);
    document.documentElement.style.colorScheme = dark ? "dark" : "light";
    window.dispatchEvent(new CustomEvent("dhawq-theme-change", { detail: { dark } }));
  }, []);

  const setTheme = useCallback(
    (next: Theme) => {
      localStorage.setItem(KEY, next);
      setThemeState(next);
      apply(next);
    },
    [apply],
  );

  // A user on "system" must follow the OS if it changes mid-session.
  useEffect(() => {
    if (theme !== "system") return;
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = () => apply("system");
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, [theme, apply]);

  return { theme, setTheme, mounted };
}

/** Read a CSS custom property. The 3D scene reads its colours from tokens so
 *  it rebinds on theme change — but product textures are NEVER tinted. */
export function token(name: string, fallback = "#000"): string {
  if (typeof window === "undefined") return fallback;
  return (
    getComputedStyle(document.documentElement).getPropertyValue(name).trim() ||
    fallback
  );
}
