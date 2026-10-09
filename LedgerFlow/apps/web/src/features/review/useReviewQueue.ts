import { useMemo } from 'react';

import { useDocuments, type DocumentFilters } from '@/api';

/** Oldest first: the review queue is worked FIFO. */
const QUEUE_FILTERS: DocumentFilters = { status: 'needs_review', sort: 'created_at' };

export interface ReviewQueuePosition {
  previousId: string | undefined;
  nextId: string | undefined;
  /** 1-based position, or undefined when the document is not (or no longer) queued. */
  position: number | undefined;
  loadedCount: number;
}

export function useReviewQueue(currentId: string): ReviewQueuePosition {
  const queue = useDocuments(QUEUE_FILTERS);
  const ids = useMemo(
    () => queue.data?.pages.flatMap((page) => page.items.map((item) => item.id)) ?? [],
    [queue.data],
  );

  const index = ids.indexOf(currentId);
  if (index === -1) {
    return { previousId: undefined, nextId: ids[0], position: undefined, loadedCount: ids.length };
  }
  return {
    previousId: ids[index - 1],
    nextId: ids[index + 1],
    position: index + 1,
    loadedCount: ids.length,
  };
}
