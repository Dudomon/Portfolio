import { useMemo, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';

import { describeError, useDocuments, type DocumentSummary } from '@/api';
import { formatDateTime, formatMoney } from '@/lib/format';
import { Button, ConfidenceMeter, DataTable, UploadIcon, type Column, type SortState } from '@/ui';

import { RequireRole } from '../auth';
import styles from './DocumentsPage.module.css';
import { StatusBadge } from './StatusBadge';
import { FilterBar } from './FilterBar';
import { UploadDialog } from './UploadDialog';
import { useDocumentFilters } from './useDocumentFilters';

type ColumnId = 'file' | 'supplier' | 'total' | 'confidence' | 'status' | 'created_at';

const DOC_TYPE_LABEL = { invoice: 'Invoice', nfe: 'NF-e' } as const;

const columns: Column<DocumentSummary, ColumnId>[] = [
  {
    id: 'file',
    header: 'Document',
    cell: (doc) => (
      <div className={styles.fileCell}>
        {/* Out of the tab order: keyboard users move between rows instead. Kept
            as a real link so middle-click / open-in-new-tab still work. */}
        <Link to={`/documents/${doc.id}`} tabIndex={-1} className={styles.fileName}>
          {doc.filename}
        </Link>
        <span className={styles.muted}>{doc.doc_type ? DOC_TYPE_LABEL[doc.doc_type] : '—'}</span>
      </div>
    ),
  },
  {
    id: 'supplier',
    header: 'Supplier',
    cell: (doc) => doc.supplier_name ?? <span className={styles.muted}>—</span>,
    hideOnMobile: true,
  },
  {
    id: 'total',
    header: 'Total',
    align: 'end',
    cell: (doc) => formatMoney(doc.total_amount, doc.currency),
  },
  {
    id: 'confidence',
    header: 'Confidence',
    cell: (doc) => <ConfidenceMeter value={doc.overall_confidence} />,
    hideOnMobile: true,
  },
  { id: 'status', header: 'Status', cell: (doc) => <StatusBadge status={doc.status} /> },
  {
    id: 'created_at',
    header: 'Uploaded',
    sortable: true,
    hideOnMobile: true,
    cell: (doc) => (
      <time dateTime={doc.created_at} className={styles.muted}>
        {formatDateTime(doc.created_at)}
      </time>
    ),
  },
];

export function DocumentsPage() {
  const navigate = useNavigate();
  const [filters, updateFilters] = useDocumentFilters();
  const [uploadOpen, setUploadOpen] = useState(false);
  const query = useDocuments(filters);

  const rows = useMemo(() => query.data?.pages.flatMap((page) => page.items) ?? [], [query.data]);
  const sort: SortState<ColumnId> = {
    columnId: 'created_at',
    direction: filters.sort === 'created_at' ? 'asc' : 'desc',
  };

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <div>
          <h1 className={styles.title}>Documents</h1>
          <p className={styles.muted}>Uploaded invoices and their extraction status.</p>
        </div>
        <RequireRole minRole="reviewer">
          <Button variant="primary" startIcon={<UploadIcon />} onClick={() => setUploadOpen(true)}>
            Upload
          </Button>
        </RequireRole>
      </header>

      <FilterBar filters={filters} onChange={updateFilters} />

      <DataTable
        caption="Documents"
        columns={columns}
        rows={rows}
        getRowId={(doc) => doc.id}
        status={query.status}
        errorMessage={query.error ? describeError(query.error) : undefined}
        onRetry={() => void query.refetch()}
        emptyState="No documents match these filters."
        sort={sort}
        onSortChange={(next) =>
          updateFilters({ sort: next.direction === 'asc' ? 'created_at' : '-created_at' })
        }
        onRowActivate={(doc) => navigate(`/documents/${doc.id}`)}
        pagination={{
          hasNextPage: query.hasNextPage,
          isFetchingNextPage: query.isFetchingNextPage,
          onLoadMore: () => void query.fetchNextPage(),
        }}
      />

      <UploadDialog open={uploadOpen} onClose={() => setUploadOpen(false)} />
    </div>
  );
}
