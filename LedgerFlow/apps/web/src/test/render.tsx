import { render } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { ReactElement } from 'react';
import { createMemoryRouter, RouterProvider, type RouteObject } from 'react-router-dom';

import { AppProviders } from '@/app/AppProviders';
import { createAppContext } from '@/app/createAppContext';
import { USERS, type MockUser } from '@/mocks/fixtures';
import { createMockToken } from '@/mocks/handlers';

export const TEST_BASE_URL = 'http://localhost/api/v1';

export function userWithRole(role: MockUser['role']): MockUser {
  const user = USERS.find((candidate) => candidate.role === role);
  if (!user) throw new Error(`No mock user with role ${role}`);
  return user;
}

/**
 * Renders routes inside the real provider stack with a signed-in user, so
 * tests exercise the same API client, cache and MSW handlers as the app.
 */
export function renderRoutes(
  routes: RouteObject[],
  { path = '/', as = userWithRole('reviewer') }: { path?: string; as?: MockUser | null } = {},
) {
  const context = createAppContext({ baseUrl: TEST_BASE_URL });
  context.queryClient.setDefaultOptions({ queries: { retry: false, staleTime: Infinity } });
  if (as)
    context.sessionStore.signIn({
      access_token: createMockToken(as),
      token_type: 'bearer',
      expires_in: 3600,
    });
  const router = createMemoryRouter(routes, { initialEntries: [path] });
  const user = userEvent.setup();
  const view = render(
    <AppProviders context={context}>
      <RouterProvider router={router} />
    </AppProviders>,
  );
  return { ...view, user, router, context };
}

export function renderElement(element: ReactElement, options?: Parameters<typeof renderRoutes>[1]) {
  return renderRoutes([{ path: '*', element }], options);
}
