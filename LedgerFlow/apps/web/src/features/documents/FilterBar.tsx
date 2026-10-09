import { useCallback, useEffect, useRef, useState } from 'react';

import { DOCUMENT_STATUSES, type DocType, type DocumentFilters } from '@/api';
import { cx } from '@/lib/cx';
import { useDebouncedValue } from '@/lib/useDebouncedValue';
import { Select, TextField, type SelectOption } from '@/ui';

import styles from './FilterBar.module.css';
import { STATUS_META } from './StatusBadge';

const SEARCH_DEBOUNCE_MS = 300;

const DOC_TYPE_OPTIONS: SelectOption[] = [
  { value: 'invoice', label: 'Invoice' },
  { value: 'nfe', label: 'NF-e' },
];

const CONFIDENCE_OPTIONS: SelectOption[] = [
  { value: '0.5', label: '≥ 50%' },
  { value: '0.7', label: '≥ 70%' },
  { value: '0.85', label: '≥ 85%' },
  { value: '0.95', label: '≥ 95%' },
];

interface FilterBarProps {
  filters: DocumentFilters;
  onChange: (patch: Partial<DocumentFilters>) => void;
}

export function FilterBar({ filters, onChange }: FilterBarProps) {
  const commitSearch = useCallback((q: string) => onChange({ q: q || undefined }), [onChange]);
  return (
    <div className={styles.bar}>
      <div className={styles.row}>
        <SearchInput value={filters.q ?? ''} onCommit={commitSearch} />
        <Select
          label="Document type"
          hideLabel
          placeholder="All types"
          options={DOC_TYPE_OPTIONS}
          value={filters.doc_type ?? ''}
          onChange={(event) =>
            onChange({ doc_type: (event.target.value || undefined) as DocType | undefined })
          }
          className={styles.select}
        />
        <Select
          label="Minimum confidence"
          hideLabel
          placeholder="Any confidence"
          options={CONFIDENCE_OPTIONS}
          value={filters.min_confidence === undefined ? '' : String(filters.min_confidence)}
          onChange={(event) =>
            onChange({
              min_confidence: event.target.value ? Number(event.target.value) : undefined,
            })
          }
          className={styles.select}
        />
      </div>
      <div className={styles.chips} role="group" aria-label="Filter by status">
        <StatusChip active={!filters.status} onClick={() => onChange({ status: undefined })}>
          All
        </StatusChip>
        {DOCUMENT_STATUSES.map((status) => (
          <StatusChip
            key={status}
            active={filters.status === status}
            onClick={() => onChange({ status })}
          >
            {STATUS_META[status].label}
          </StatusChip>
        ))}
      </div>
    </div>
  );
}

function StatusChip({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: string;
}) {
  return (
    <button
      type="button"
      aria-pressed={active}
      className={cx(styles.chip, active && styles.chipActive)}
      onClick={onClick}
    >
      {children}
    </button>
  );
}

/**
 * Typing updates local state immediately; the URL (and therefore the query)
 * only follows once the user pauses, so each keystroke is not a request.
 */
function SearchInput({ value, onCommit }: { value: string; onCommit: (q: string) => void }) {
  const [draft, setDraft] = useState(value);
  const debounced = useDebouncedValue(draft, SEARCH_DEBOUNCE_MS);
  const committed = useRef(value);

  useEffect(() => {
    const next = debounced.trim();
    if (next === committed.current) return;
    committed.current = next;
    onCommit(next);
  }, [debounced, onCommit]);

  // The URL changed underneath us (back/forward, a link): adopt it.
  useEffect(() => {
    if (value === committed.current) return;
    committed.current = value;
    setDraft(value);
  }, [value]);

  return (
    <TextField
      label="Search documents"
      hideLabel
      type="search"
      placeholder="Search supplier, number or file name"
      value={draft}
      onChange={(event) => setDraft(event.target.value)}
      className={styles.search}
    />
  );
}
