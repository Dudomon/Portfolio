import {
  keepPreviousData,
  useInfiniteQuery,
  useMutation,
  useQuery,
  useQueryClient,
} from '@tanstack/react-query';
import { useState } from 'react';

import { useApi } from '../ApiProvider';
import type { DocumentFilters } from '../ledgerflow';
import { documentKeys, statsKeys } from '../queryKeys';
import type { DocumentStatus } from '../schemas';

const IN_FLIGHT_STATUSES: ReadonlySet<DocumentStatus> = new Set([
  'uploaded',
  'queued',
  'processing',
]);
const IN_FLIGHT_POLL_MS = 5_000;

export function useDocuments(filters: DocumentFilters, { limit = 50 }: { limit?: number } = {}) {
  const api = useApi();
  return useInfiniteQuery({
    queryKey: documentKeys.list(filters),
    queryFn: ({ pageParam, signal }) =>
      api.listDocuments({ ...filters, limit, cursor: pageParam }, signal),
    initialPageParam: undefined as string | undefined,
    getNextPageParam: (lastPage) => lastPage.next_cursor ?? undefined,
    placeholderData: keepPreviousData,
    // Poll while anything on screen is still moving through the pipeline.
    refetchInterval: (query) =>
      query.state.data?.pages.some((page) =>
        page.items.some((item) => IN_FLIGHT_STATUSES.has(item.status)),
      )
        ? IN_FLIGHT_POLL_MS
        : false,
  });
}

export function useDocument(id: string) {
  const api = useApi();
  return useQuery({
    queryKey: documentKeys.detail(id),
    queryFn: ({ signal }) => api.getDocument(id, signal),
  });
}

export function useDocumentFile(id: string) {
  const api = useApi();
  return useQuery({
    queryKey: documentKeys.file(id),
    queryFn: ({ signal }) => api.getDocumentFile(id, signal),
    // The stored original is immutable (content-addressed by sha256).
    staleTime: Infinity,
  });
}

export function useAudit(id: string, { enabled = true }: { enabled?: boolean } = {}) {
  const api = useApi();
  return useQuery({
    queryKey: documentKeys.audit(id),
    queryFn: ({ signal }) => api.getAudit(id, signal),
    enabled,
  });
}

export function useUploadDocument() {
  const api = useApi();
  const queryClient = useQueryClient();
  const [progress, setProgress] = useState(0);

  const mutation = useMutation({
    mutationFn: ({ file, signal }: { file: File; signal?: AbortSignal }) => {
      setProgress(0);
      return api.uploadDocument(file, { onProgress: setProgress, signal });
    },
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: documentKeys.lists() });
      void queryClient.invalidateQueries({ queryKey: statsKeys.all });
    },
  });

  return { ...mutation, progress };
}
