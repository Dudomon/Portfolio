import type { DocumentFilters, StatsRange } from './ledgerflow';

/**
 * Hierarchical keys so invalidation can target any level, e.g.
 * `documentKeys.lists()` refreshes every filtered list at once.
 */
export const documentKeys = {
  all: ['documents'] as const,
  lists: () => [...documentKeys.all, 'list'] as const,
  list: (filters: DocumentFilters) => [...documentKeys.lists(), filters] as const,
  details: () => [...documentKeys.all, 'detail'] as const,
  detail: (id: string) => [...documentKeys.details(), id] as const,
  audit: (id: string) => [...documentKeys.detail(id), 'audit'] as const,
  // Kept outside `detail` so field edits do not re-download the original file.
  file: (id: string) => [...documentKeys.all, 'file', id] as const,
};

export const statsKeys = {
  all: ['stats'] as const,
  range: (range: StatsRange) => [...statsKeys.all, range] as const,
};

export const sessionKeys = {
  me: ['me'] as const,
};
