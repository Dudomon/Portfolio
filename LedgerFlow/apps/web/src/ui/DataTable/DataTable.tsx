import { useState, type KeyboardEvent, type ReactNode } from 'react';

import { cx } from '@/lib/cx';

import { Button } from '../Button/Button';
import { Callout } from '../Callout/Callout';
import { ChevronDownIcon, ChevronUpIcon, SortIcon } from '../icons';
import { Skeleton } from '../Skeleton/Skeleton';
import styles from './DataTable.module.css';

export type SortDirection = 'asc' | 'desc';

export interface SortState<K extends string = string> {
  columnId: K;
  direction: SortDirection;
}

export interface Column<T, K extends string = string> {
  id: K;
  header: string;
  cell: (row: T) => ReactNode;
  sortable?: boolean;
  align?: 'start' | 'end';
  width?: string;
  /** Hidden below the tablet breakpoint to keep the table readable on phones. */
  hideOnMobile?: boolean;
}

export interface DataTablePagination {
  hasNextPage: boolean;
  isFetchingNextPage: boolean;
  onLoadMore: () => void;
}

export interface DataTableProps<T, K extends string = string> {
  /** Accessible name of the table. */
  caption: string;
  columns: readonly Column<T, K>[];
  rows: readonly T[];
  getRowId: (row: T) => string;
  status: 'pending' | 'error' | 'success';
  errorMessage?: string;
  onRetry?: () => void;
  emptyState?: ReactNode;
  /** Sorting is controlled: the table only reports intent, the owner fetches/sorts. */
  sort?: SortState<K>;
  onSortChange?: (sort: SortState<K>) => void;
  /**
   * Makes rows keyboard-navigable (Arrow keys, Home/End) and activatable
   * (Enter/Space or click) using a roving tabindex: one tab stop per table.
   */
  onRowActivate?: (row: T) => void;
  pagination?: DataTablePagination;
  skeletonRows?: number;
  className?: string;
}

/** Clicks on links/buttons inside a row should do their own thing, not activate the row. */
function isInsideControl(target: EventTarget): boolean {
  return target instanceof Element && target.closest('a, button, input, select, textarea') !== null;
}

const ARIA_SORT: Record<SortDirection, 'ascending' | 'descending'> = {
  asc: 'ascending',
  desc: 'descending',
};

export function DataTable<T, K extends string = string>({
  caption,
  columns,
  rows,
  getRowId,
  status,
  errorMessage = 'The data could not be loaded.',
  onRetry,
  emptyState = 'No results.',
  sort,
  onSortChange,
  onRowActivate,
  pagination,
  skeletonRows = 5,
  className,
}: DataTableProps<T, K>) {
  const [activeIndex, setActiveIndex] = useState(0);
  const interactive = onRowActivate !== undefined;
  const focusableIndex = Math.min(activeIndex, Math.max(rows.length - 1, 0));

  function toggleSort(columnId: K) {
    if (!onSortChange) return;
    const direction: SortDirection =
      sort?.columnId === columnId && sort.direction === 'asc' ? 'desc' : 'asc';
    onSortChange({ columnId, direction });
  }

  function onRowKeyDown(event: KeyboardEvent<HTMLTableRowElement>, index: number, row: T) {
    // Keys pressed inside a nested control belong to that control.
    if (event.target !== event.currentTarget) return;
    let target: number | null = null;
    switch (event.key) {
      case 'ArrowDown':
        target = Math.min(index + 1, rows.length - 1);
        break;
      case 'ArrowUp':
        target = Math.max(index - 1, 0);
        break;
      case 'Home':
        target = 0;
        break;
      case 'End':
        target = rows.length - 1;
        break;
      case 'Enter':
      case ' ':
        event.preventDefault();
        onRowActivate?.(row);
        return;
      default:
        return;
    }
    event.preventDefault();
    setActiveIndex(target);
    const tbody = event.currentTarget.parentElement;
    tbody?.querySelectorAll<HTMLTableRowElement>('tr[data-row]')[target]?.focus();
  }

  function renderBody() {
    if (status === 'pending') {
      return Array.from({ length: skeletonRows }, (_, index) => (
        <tr key={`skeleton-${index}`}>
          {columns.map((column) => (
            <td key={column.id} className={cx(column.hideOnMobile && styles.hideOnMobile)}>
              <Skeleton width={index % 2 ? '70%' : '90%'} />
            </td>
          ))}
        </tr>
      ));
    }
    if (status === 'error') {
      return (
        <tr>
          <td colSpan={columns.length} className={styles.stateCell}>
            <Callout
              tone="danger"
              title={errorMessage}
              actions={
                onRetry && (
                  <Button size="sm" onClick={onRetry}>
                    Retry
                  </Button>
                )
              }
            />
          </td>
        </tr>
      );
    }
    if (rows.length === 0) {
      return (
        <tr>
          <td colSpan={columns.length} className={cx(styles.stateCell, styles.empty)}>
            {emptyState}
          </td>
        </tr>
      );
    }
    return rows.map((row, index) => (
      <tr
        key={getRowId(row)}
        data-row=""
        className={cx(interactive && styles.interactiveRow)}
        tabIndex={interactive ? (index === focusableIndex ? 0 : -1) : undefined}
        onClick={
          interactive
            ? (event) => {
                if (!isInsideControl(event.target)) onRowActivate(row);
              }
            : undefined
        }
        onKeyDown={interactive ? (event) => onRowKeyDown(event, index, row) : undefined}
        onFocus={interactive ? () => setActiveIndex(index) : undefined}
      >
        {columns.map((column) => (
          <td
            key={column.id}
            className={cx(
              column.align === 'end' && styles.alignEnd,
              column.hideOnMobile && styles.hideOnMobile,
            )}
          >
            {column.cell(row)}
          </td>
        ))}
      </tr>
    ));
  }

  return (
    <div className={cx(styles.container, className)}>
      <div className={styles.scroller}>
        <table className={styles.table} aria-busy={status === 'pending' || undefined}>
          <caption className="visually-hidden">{caption}</caption>
          <thead>
            <tr>
              {columns.map((column) => {
                const sorted = sort?.columnId === column.id ? sort.direction : undefined;
                const canSort = column.sortable && onSortChange;
                return (
                  <th
                    key={column.id}
                    scope="col"
                    style={column.width ? { width: column.width } : undefined}
                    aria-sort={canSort ? (sorted ? ARIA_SORT[sorted] : 'none') : undefined}
                    className={cx(
                      column.align === 'end' && styles.alignEnd,
                      column.hideOnMobile && styles.hideOnMobile,
                    )}
                  >
                    {canSort ? (
                      <button
                        type="button"
                        className={styles.sortButton}
                        onClick={() => toggleSort(column.id)}
                      >
                        {column.header}
                        {sorted === 'asc' && <ChevronUpIcon />}
                        {sorted === 'desc' && <ChevronDownIcon />}
                        {!sorted && <SortIcon className={styles.sortIdle} />}
                      </button>
                    ) : (
                      column.header
                    )}
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>{renderBody()}</tbody>
        </table>
      </div>
      {pagination && status === 'success' && pagination.hasNextPage && (
        <div className={styles.footer}>
          <Button
            onClick={pagination.onLoadMore}
            loading={pagination.isFetchingNextPage}
            variant="secondary"
          >
            Load more
          </Button>
        </div>
      )}
    </div>
  );
}
