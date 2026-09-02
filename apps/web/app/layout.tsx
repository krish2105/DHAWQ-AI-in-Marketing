import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { Nav } from "@/components/ui/Nav";

const ui = Inter({ subsets: ["latin"], variable: "--font-ui", display: "swap" });
const mono = JetBrains_Mono({ subsets: ["latin"], variable: "--font-mono", display: "swap" });

export const metadata: Metadata = {
  title: "DHAWQ — Visual Recommendation Intelligence",
  description:
    "13,548 real fashion products in learned embedding space, with an agentic merchandising copilot whose refusals are as visible as its output.",
};

/*
 * FOUC is non-negotiable on a gallery (§12.4). This runs BEFORE first paint,
 * reads the stored choice and the OS preference, and stamps the class on <html>
 * so the page never flashes the wrong ground colour.
 *
 * THREE states, not two: dark / light / system, with system the default. A
 * two-state toggle silently overrides the OS preference on first paint, which
 * is a real accessibility failure for light-sensitive users.
 */
const THEME_SCRIPT = `
(function () {
  try {
    var stored = localStorage.getItem("dhawq-theme");
    var mode = stored || "system";
    var dark = mode === "dark" ||
      (mode === "system" && window.matchMedia("(prefers-color-scheme: dark)").matches);
    document.documentElement.classList.toggle("dark", dark);
    document.documentElement.style.colorScheme = dark ? "dark" : "light";
  } catch (e) {
    document.documentElement.classList.add("dark");
  }
})();`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" dir="ltr" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_SCRIPT }} />
      </head>
      <body className={`${ui.variable} ${mono.variable}`}>
        <Nav />
        <main id="main">{children}</main>
      </body>
    </html>
  );
}
