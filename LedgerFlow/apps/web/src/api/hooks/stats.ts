import { keepPreviousData, useQuery } from '@tanstack/react-query';

import { useApi } from '../ApiProvider';
import type { StatsRange } from '../ledgerflow';
import { statsKeys } from '../queryKeys';

export function useStats(range: StatsRange) {
  const api = useApi();
  return useQuery({
    queryKey: statsKeys.range(range),
    queryFn: ({ signal }) => api.getStats(range, signal),
    placeholderData: keepPreviousData,
  });
}
