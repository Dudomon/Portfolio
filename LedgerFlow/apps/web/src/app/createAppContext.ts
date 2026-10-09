import { QueryClient } from '@tanstack/react-query';

import { ApiError, createApiClient, createLedgerFlowApi } from '@/api';
import { createSessionStore } from '@/features/auth';

const MAX_RETRIES = 2;

export function createQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: {
        staleTime: 30_000,
        // 4xx answers are deterministic; retrying them only delays the error state.
        retry: (failureCount, error) =>
          !(error instanceof ApiError && error.status < 500) && failureCount < MAX_RETRIES,
      },
    },
  });
}

/** Wires the object graph once: session store → HTTP client → typed API → query cache. */
export function createAppContext({ baseUrl }: { baseUrl: string }) {
  const sessionStore = createSessionStore();
  const client = createApiClient({
    baseUrl,
    getAccessToken: sessionStore.getAccessToken,
    onUnauthorized: () => sessionStore.signOut('expired'),
  });
  return {
    sessionStore,
    api: createLedgerFlowApi(client),
    queryClient: createQueryClient(),
  };
}

export type AppContext = ReturnType<typeof createAppContext>;
