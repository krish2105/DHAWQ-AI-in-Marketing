import type { NextConfig } from "next";

// The deployed API. Overridable by env for local work, but defaulted here so a
// clean clone deploys without extra configuration.
const API =
  process.env.NEXT_PUBLIC_API_URL ??
  (process.env.VERCEL ? "https://dhawq-api.onrender.com" : "http://127.0.0.1:8001");

const config: NextConfig = {
  reactStrictMode: false,
  async rewrites() {
    // Static artefacts and API share an origin in dev so the browser never
    // deals with CORS for the atlas or positions.bin.
    return [{ source: "/api/:path*", destination: `${API}/:path*` }];
  },
};
export default config;
