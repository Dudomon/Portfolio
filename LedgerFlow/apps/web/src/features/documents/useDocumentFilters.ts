import { useCallback, useMemo } from 'react';
import { useSearchParams } from 'react-router-dom';

import { DocTypeSchema, DocumentStatusSchema, type DocumentFilters } from '@/api';

export const DEFAULT_SORT = '-created_at';

/** Reads filters from the URL, dropping anything malformed instead of sending it to the API. */
export function parseDocumentFilters(params: URLSearchParams): DocumentFilters {
  const status = DocumentStatusSchema.safeParse(params.get('status'));
  const docType = DocTypeSchema.safeParse(params.get('doc_type'));
  const minConfidence = Number(params.get('min_confidence'));
  const sort = params.get('sort');
  const q = params.get('q')?.trim();

  return {
    status: status.success ? status.data : undefined,
    doc_type: docType.success ? docType.data : undefined,
    q: q === '' ? undefined : q,
    min_confidence:
      params.has('min_confidence') && minConfidence > 0 && minConfidence <= 1
        ? minConfidence
        : undefined,
    sort: sort === 'created_at' ? 'created_at' : DEFAULT_SORT,
  };
}

/** Filters live in the URL so views are shareable and survive reloads and back/forward. */
export function useDocumentFilters() {
  const [params, setParams] = useSearchParams();
  const filters = useMemo(() => parseDocumentFilters(params), [params]);

  const updateFilters = useCallback(
    (patch: Partial<DocumentFilters>) => {
      setParams(
        (current) => {
          const next = new URLSearchParams(current);
          // Object.entries drops `undefined` from the value type, but clearing a
          // filter is expressed exactly that way.
          const entries = Object.entries(patch) as [string, string | number | undefined][];
          for (const [key, value] of entries) {
            if (value === undefined) next.delete(key);
            else next.set(key, String(value));
          }
          // The default sort is implied; keep URLs short and canonical.
          if (next.get('sort') === DEFAULT_SORT) next.delete('sort');
          return next;
        },
        { replace: true },
      );
    },
    [setParams],
  );

  return [filters, updateFilters] as const;
}
