import { useQuery } from '@tanstack/react-query';

import { useApi } from '../ApiProvider';
import { sessionKeys } from '../queryKeys';

export function useMe({ enabled = true }: { enabled?: boolean } = {}) {
  const api = useApi();
  return useQuery({
    queryKey: sessionKeys.me,
    queryFn: ({ signal }) => api.getMe(signal),
    staleTime: 5 * 60_000,
    enabled,
  });
}
