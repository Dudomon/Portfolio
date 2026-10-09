import { TokenClaimsSchema, type Role, type TokenClaims, type TokenResponse } from '@/api';

export interface Session {
  accessToken: string;
  claims: TokenClaims;
  /** Epoch milliseconds, on the client clock. */
  expiresAt: number;
}

export type SignOutReason = 'user' | 'expired';

export interface SessionState {
  session: Session | null;
  signOutReason: SignOutReason | null;
}

export function decodeClaims(token: string): TokenClaims | null {
  const payload = token.split('.')[1];
  if (!payload) return null;
  try {
    const base64 = payload.replace(/-/g, '+').replace(/_/g, '/');
    const json = new TextDecoder().decode(Uint8Array.from(atob(base64), (c) => c.charCodeAt(0)));
    const result = TokenClaimsSchema.safeParse(JSON.parse(json));
    return result.success ? result.data : null;
  } catch {
    return null;
  }
}

const ROLE_RANK: Record<Role, number> = { viewer: 1, reviewer: 2, admin: 3 };

export function hasRole(role: Role, minimum: Role): boolean {
  return ROLE_RANK[role] >= ROLE_RANK[minimum];
}

/**
 * Holds the access token in memory only: nothing readable by injected script
 * survives a reload, at the cost of signing in again per tab. The contract has
 * no refresh endpoint, so expiry (timer or a 401) ends the session.
 */
export function createSessionStore() {
  let state: SessionState = { session: null, signOutReason: null };
  let expiryTimer: ReturnType<typeof setTimeout> | undefined;
  const listeners = new Set<() => void>();

  function setState(next: SessionState) {
    state = next;
    listeners.forEach((listener) => listener());
  }

  function signOut(reason: SignOutReason) {
    clearTimeout(expiryTimer);
    if (state.session) setState({ session: null, signOutReason: reason });
  }

  return {
    getState: () => state,
    subscribe: (listener: () => void) => {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
    getAccessToken: () => state.session?.accessToken ?? null,

    signIn: (token: TokenResponse): Session => {
      const claims = decodeClaims(token.access_token);
      if (!claims)
        throw new Error('The server returned an access token without the expected claims.');
      // `expires_in` is relative, so it is immune to skew between client and server clocks.
      const expiresAt = Date.now() + token.expires_in * 1000;
      const session: Session = { accessToken: token.access_token, claims, expiresAt };
      clearTimeout(expiryTimer);
      expiryTimer = setTimeout(() => signOut('expired'), token.expires_in * 1000);
      setState({ session, signOutReason: null });
      return session;
    },

    signOut,
  };
}

export type SessionStore = ReturnType<typeof createSessionStore>;
