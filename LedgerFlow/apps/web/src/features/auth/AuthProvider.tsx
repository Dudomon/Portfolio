import { useQueryClient } from '@tanstack/react-query';
import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useSyncExternalStore,
  type ReactNode,
} from 'react';

import type { TokenResponse } from '@/api';

import type { Session, SessionState, SessionStore } from './sessionStore';

interface AuthContextValue {
  state: SessionState;
  signIn: (token: TokenResponse) => Session;
  signOut: () => void;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ store, children }: { store: SessionStore; children: ReactNode }) {
  const queryClient = useQueryClient();
  const state = useSyncExternalStore(store.subscribe, store.getState);

  // Cached server data belongs to the previous user/tenant; never let it leak across sessions.
  const signIn = useCallback(
    (token: TokenResponse) => {
      queryClient.removeQueries();
      return store.signIn(token);
    },
    [queryClient, store],
  );

  const signOut = useCallback(() => {
    store.signOut('user');
    queryClient.removeQueries();
  }, [queryClient, store]);

  const value = useMemo(() => ({ state, signIn, signOut }), [state, signIn, signOut]);
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext);
  if (!context) throw new Error('useAuth must be used inside <AuthProvider>');
  return context;
}

/** The current session. Only call below <RequireAuth>, where one is guaranteed. */
export function useSession(): Session {
  const { session } = useAuth().state;
  if (!session) throw new Error('useSession called outside an authenticated route');
  return session;
}
