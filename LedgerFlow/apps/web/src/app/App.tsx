import { useState } from 'react';
import { createBrowserRouter, RouterProvider } from 'react-router-dom';

import { AppProviders } from './AppProviders';
import { createAppContext } from './createAppContext';
import { routes } from './routes';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? '/api/v1';

export function App() {
  // Created once per mount (not at module scope) so tests and HMR get fresh state.
  const [context] = useState(() => createAppContext({ baseUrl: API_BASE_URL }));
  const [router] = useState(() => createBrowserRouter(routes));
  return (
    <AppProviders context={context}>
      <RouterProvider router={router} />
    </AppProviders>
  );
}
