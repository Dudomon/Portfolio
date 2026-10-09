import { createContext, useContext, type ReactNode } from 'react';

import type { LedgerFlowApi } from './ledgerflow';

const ApiContext = createContext<LedgerFlowApi | null>(null);

export function ApiProvider({ api, children }: { api: LedgerFlowApi; children: ReactNode }) {
  return <ApiContext.Provider value={api}>{children}</ApiContext.Provider>;
}

export function useApi(): LedgerFlowApi {
  const api = useContext(ApiContext);
  if (!api) throw new Error('useApi must be used inside <ApiProvider>');
  return api;
}
