import { QueryClientProvider } from '@tanstack/react-query';
import type { ReactNode } from 'react';

import { ApiProvider } from '@/api';
import { AuthProvider } from '@/features/auth';
import { ToastProvider } from '@/ui';

import type { AppContext } from './createAppContext';

export function AppProviders({ context, children }: { context: AppContext; children: ReactNode }) {
  return (
    <QueryClientProvider client={context.queryClient}>
      <ApiProvider api={context.api}>
        <AuthProvider store={context.sessionStore}>
          <ToastProvider>{children}</ToastProvider>
        </AuthProvider>
      </ApiProvider>
    </QueryClientProvider>
  );
}
