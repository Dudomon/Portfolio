import { useState, useCallback, useEffect, createContext, useContext } from "react";
import { useMutation, useQuery, useApolloClient } from "@apollo/client";
import { LOGIN, REGISTER, GET_ME } from "../graphql/operations";
import { setAuthToken, clearAuthToken, getAuthToken } from "../graphql/client";

/**
 * Authentication hook managing user session state.
 *
 * Architecture decision: Auth state lives in React context, not Apollo cache.
 * Apollo cache persists across page refreshes (with apollo-cache-persist),
 * which would keep stale user data after logout. Context clears on refresh,
 * forcing fresh auth check.
 *
 * Token storage uses localStorage over sessionStorage because users expect
 * to stay logged in across browser sessions. Security tradeoff accepted
 * for UX; XSS is the real threat, mitigated by CSP headers and input
 * sanitization.
 */

interface User {
  id: string;
  email: string;
  displayName: string;
  avatarUrl: string | null;
  verifiedSeller: boolean;
}

interface AuthContextValue {
  user: User | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string, displayName: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within AuthProvider");
  }
  return context;
}

interface AuthProviderProps {
  children: React.ReactNode;
}

export function AuthProvider({ children }: AuthProviderProps): JSX.Element {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const client = useApolloClient();

  const [loginMutation] = useMutation(LOGIN);
  const [registerMutation] = useMutation(REGISTER);

  // Check for existing session on mount
  const { refetch: fetchMe } = useQuery(GET_ME, {
    skip: !getAuthToken(),
    onCompleted: (data) => {
      if (data.me) {
        setUser(data.me);
      }
      setIsLoading(false);
    },
    onError: () => {
      // Token invalid or expired
      clearAuthToken();
      setIsLoading(false);
    },
  });

  useEffect(() => {
    if (!getAuthToken()) {
      setIsLoading(false);
    }
  }, []);

  const login = useCallback(
    async (email: string, password: string) => {
      const { data } = await loginMutation({
        variables: { email, password },
      });

      if (data?.login) {
        setAuthToken(data.login.token);
        setUser(data.login.user);
      }
    },
    [loginMutation]
  );

  const register = useCallback(
    async (email: string, password: string, displayName: string) => {
      const { data } = await registerMutation({
        variables: { email, password, displayName },
      });

      if (data?.register) {
        setAuthToken(data.register.token);
        setUser(data.register.user);
      }
    },
    [registerMutation]
  );

  const logout = useCallback(() => {
    clearAuthToken();
    setUser(null);
    // Clear Apollo cache to remove user specific data
    client.clearStore();
  }, [client]);

  const value: AuthContextValue = {
    user,
    isLoading,
    isAuthenticated: !!user,
    login,
    register,
    logout,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}
