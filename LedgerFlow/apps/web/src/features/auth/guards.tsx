import type { ReactNode } from 'react';
import { Navigate, Outlet, useLocation } from 'react-router-dom';

import type { Role } from '@/api';

import { useAuth, useSession } from './AuthProvider';
import { hasRole } from './sessionStore';

export interface LoginRedirectState {
  from?: string;
  reason?: 'expired';
}

/** Route element: renders child routes for signed-in users, otherwise redirects to /login. */
export function RequireAuth() {
  const { session, signOutReason } = useAuth().state;
  const location = useLocation();
  if (!session) {
    const state: LoginRedirectState = {
      from: `${location.pathname}${location.search}`,
      reason: signOutReason === 'expired' ? 'expired' : undefined,
    };
    return <Navigate to="/login" replace state={state} />;
  }
  return <Outlet />;
}

export function useHasRole(minimum: Role): boolean {
  return hasRole(useSession().claims.role, minimum);
}

/**
 * Renders `children` only for users with at least `minRole`. This is a UX
 * affordance; the API enforces RBAC independently.
 */
export function RequireRole({
  minRole,
  children,
  fallback = null,
}: {
  minRole: Role;
  children: ReactNode;
  fallback?: ReactNode;
}) {
  return useHasRole(minRole) ? children : fallback;
}
