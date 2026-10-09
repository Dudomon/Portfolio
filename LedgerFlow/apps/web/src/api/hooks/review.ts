import { useMutation, useQueryClient, type QueryClient } from '@tanstack/react-query';

import { useApi } from '../ApiProvider';
import type { FieldCorrections, Versioned } from '../ledgerflow';
import { documentKeys, statsKeys } from '../queryKeys';
import type { DocumentDetail } from '../schemas';

/**
 * Every review mutation takes the `etag` the reviewer's view was built from,
 * not whatever is in the cache now: a background refetch may have pulled in
 * someone else's edit, and sending that newer tag would silently overwrite it.
 */
interface Precondition {
  etag: string;
}

function invalidateAfterReview(queryClient: QueryClient, id: string) {
  void queryClient.invalidateQueries({ queryKey: documentKeys.audit(id) });
  void queryClient.invalidateQueries({ queryKey: documentKeys.lists() });
  void queryClient.invalidateQueries({ queryKey: statsKeys.all });
}

export function applyCorrections(
  document: DocumentDetail,
  corrections: FieldCorrections,
): DocumentDetail {
  if (!document.extraction) return document;
  const fields = { ...document.extraction.fields };
  for (const [key, value] of Object.entries(corrections)) {
    fields[key] = { value: value ?? null, confidence: 1, source: 'human' };
  }
  return { ...document, extraction: { ...document.extraction, fields } };
}

export function useCorrectFields(id: string) {
  const api = useApi();
  const queryClient = useQueryClient();
  const key = documentKeys.detail(id);

  return useMutation({
    mutationFn: ({ fields, etag }: Precondition & { fields: FieldCorrections }) =>
      api.correctFields(id, fields, etag),
    onMutate: async ({ fields }) => {
      await queryClient.cancelQueries({ queryKey: key });
      const previous = queryClient.getQueryData<Versioned<DocumentDetail>>(key);
      if (previous) {
        queryClient.setQueryData<Versioned<DocumentDetail>>(key, {
          ...previous,
          data: applyCorrections(previous.data, fields),
        });
      }
      return { previous };
    },
    onError: (_error, _variables, context) => {
      if (context?.previous) queryClient.setQueryData(key, context.previous);
    },
    onSuccess: (result) => {
      queryClient.setQueryData(key, result);
      invalidateAfterReview(queryClient, id);
    },
  });
}

export function useApprove(id: string) {
  const api = useApi();
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ etag }: Precondition) => api.approve(id, etag),
    onSuccess: (result) => {
      queryClient.setQueryData(documentKeys.detail(id), result);
      invalidateAfterReview(queryClient, id);
    },
  });
}

export function useReject(id: string) {
  const api = useApi();
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ reason, etag }: Precondition & { reason: string }) =>
      api.reject(id, reason, etag),
    onSuccess: (result) => {
      queryClient.setQueryData(documentKeys.detail(id), result);
      invalidateAfterReview(queryClient, id);
    },
  });
}

export function useRetryDocument(id: string) {
  const api = useApi();
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () => api.retry(id),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: documentKeys.detail(id) });
      void queryClient.invalidateQueries({ queryKey: documentKeys.lists() });
    },
  });
}
