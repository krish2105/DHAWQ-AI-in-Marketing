"use client";

import { createContext, useCallback, useContext, useEffect, useState } from "react";
import { apiGet, apiPost } from "./api";

/**
 * Auth state for the app.
 *
 * The token lives in an httpOnly cookie, so this holds NO credential — it only
 * mirrors who the server says you are. There is deliberately no way to read the
 * token from JavaScript: that is the property that makes an XSS not an account
 * takeover.
 */

export type Me = { user_id: string; email: string; role: string; scopes: string[] };
export type DemoAccount = { email: string; role: string; scopes: string[] };

type Ctx = {
  me: Me | null;
  loading: boolean;
  accounts: DemoAccount[];
  password: string;
  login: (email: string, password: string) => Promise<string | null>;
  logout: () => Promise<void>;
  can: (scope: string) => boolean;
};

const AuthCtx = createContext<Ctx | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [me, setMe] = useState<Me | null>(null);
  const [loading, setLoading] = useState(true);
  const [accounts, setAccounts] = useState<DemoAccount[]>([]);
  const [password, setPassword] = useState("");

  const refresh = useCallback(async () => {
    const r = await apiGet<Me>("/api/auth/me");
    setMe(r.ok ? r.data : null);
    setLoading(false);
  }, []);

  useEffect(() => {
    refresh();
    apiGet<{ password: string; accounts: DemoAccount[] }>("/api/auth/demo-accounts")
      .then((r) => {
        if (r.ok) { setAccounts(r.data.accounts); setPassword(r.data.password); }
      });
  }, [refresh]);

  const login = useCallback(async (email: string, pw: string) => {
    const r = await apiPost<Me>("/api/auth/login", { email, password: pw });
    if (!r.ok) return r.error.message;
    setMe(r.data);
    return null;
  }, []);

  const logout = useCallback(async () => {
    await apiPost("/api/auth/logout", {});
    setMe(null);
  }, []);

  const can = useCallback((scope: string) => !!me?.scopes.includes(scope), [me]);

  return (
    <AuthCtx.Provider value={{ me, loading, accounts, password, login, logout, can }}>
      {children}
    </AuthCtx.Provider>
  );
}

export function useAuth(): Ctx {
  const c = useContext(AuthCtx);
  if (!c) throw new Error("useAuth outside AuthProvider");
  return c;
}
