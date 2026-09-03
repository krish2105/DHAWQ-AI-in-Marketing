/**
 * One fetch helper for every API call.
 *
 * WHY THIS EXISTS: pages were calling `.then(r => r.json())` directly. When the
 * API was OOM-killed, Render returned an HTML error page, JSON.parse threw
 * "SyntaxError: The string did not match the expected pattern", and the whole
 * route died with a raw parser message where the content should be.
 *
 * A frontend that only works while the backend is healthy is not finished. The
 * failure is now typed, so a page can say what went wrong and stay usable.
 */

export type ApiError = {
  kind: "http" | "not_json" | "network" | "cold_start";
  status?: number;
  message: string;
};

export type ApiResult<T> = { ok: true; data: T } | { ok: false; error: ApiError };

export async function apiGet<T>(path: string, timeoutMs = 45_000): Promise<ApiResult<T>> {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), timeoutMs);
  try {
    const res = await fetch(path, { signal: ctl.signal });

    if (!res.ok) {
      // 502/503 from a sleeping or restarting free instance is the common case
      // and deserves its own message, not "server error".
      const cold = res.status === 502 || res.status === 503 || res.status === 504;
      return {
        ok: false,
        error: {
          kind: cold ? "cold_start" : "http",
          status: res.status,
          message: cold
            ? "The API is waking up. Free instances sleep after inactivity — this takes up to a minute on the first request."
            : `The API returned ${res.status}.`,
        },
      };
    }

    // Guard the parse rather than letting it throw. An HTML error page is a
    // 200 in some proxy configurations.
    const text = await res.text();
    try {
      return { ok: true, data: JSON.parse(text) as T };
    } catch {
      return {
        ok: false,
        error: {
          kind: "not_json",
          message: "The API returned something that was not JSON — usually an error page from the host.",
        },
      };
    }
  } catch (e: unknown) {
    const aborted = e instanceof DOMException && e.name === "AbortError";
    return {
      ok: false,
      error: {
        kind: aborted ? "cold_start" : "network",
        message: aborted
          ? "The API did not respond in time. Free instances sleep after inactivity; try again in a moment."
          : "Could not reach the API.",
      },
    };
  } finally {
    clearTimeout(timer);
  }
}

export async function apiPost<T>(path: string, body: unknown): Promise<ApiResult<T>> {
  try {
    const res = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const text = await res.text();
    if (!res.ok) {
      const cold = res.status >= 502;
      return {
        ok: false,
        error: {
          kind: cold ? "cold_start" : "http", status: res.status,
          message: cold
            ? "The API is waking up. Try again in a moment."
            : `The API returned ${res.status}.`,
        },
      };
    }
    try {
      return { ok: true, data: JSON.parse(text) as T };
    } catch {
      return { ok: false, error: { kind: "not_json", message: "The API returned something that was not JSON." } };
    }
  } catch {
    return { ok: false, error: { kind: "network", message: "Could not reach the API." } };
  }
}
