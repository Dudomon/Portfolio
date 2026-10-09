import type { Meta, StoryObj } from '@storybook/react';
import { fn } from '@storybook/test';
import { useMemo, useState } from 'react';

import { Badge } from '../Badge/Badge';
import { DataTable, type Column, type SortState } from './DataTable';

interface Invoice {
  id: string;
  supplier: string;
  total: string;
  status: 'approved' | 'needs_review';
}

const invoices: Invoice[] = [
  { id: '1', supplier: 'Acme Ltda', total: '1530.40', status: 'needs_review' },
  { id: '2', supplier: 'Borealis GmbH', total: '89.90', status: 'approved' },
  { id: '3', supplier: 'Cobalt SA', total: '12045.00', status: 'approved' },
  { id: '4', supplier: 'Delta Supplies', total: '310.75', status: 'needs_review' },
];

type ColumnId = 'supplier' | 'total' | 'status';

const columns: Column<Invoice, ColumnId>[] = [
  { id: 'supplier', header: 'Supplier', cell: (row) => row.supplier, sortable: true },
  { id: 'total', header: 'Total', cell: (row) => row.total, align: 'end' },
  {
    id: 'status',
    header: 'Status',
    cell: (row) => (
      <Badge tone={row.status === 'approved' ? 'success' : 'warning'} dot>
        {row.status === 'approved' ? 'Approved' : 'Needs review'}
      </Badge>
    ),
    hideOnMobile: true,
  },
];

const meta = {
  title: 'UI/DataTable',
  component: DataTable<Invoice, ColumnId>,
  args: {
    caption: 'Invoices',
    columns,
    rows: invoices,
    getRowId: (row) => row.id,
    status: 'success',
    onRowActivate: fn(),
  },
} satisfies Meta<typeof DataTable<Invoice, ColumnId>>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};

export const Sortable: Story = {
  render: function SortableTable(args) {
    const [sort, setSort] = useState<SortState<ColumnId>>({
      columnId: 'supplier',
      direction: 'asc',
    });
    const rows = useMemo(() => {
      const sorted = [...invoices].sort((a, b) => a.supplier.localeCompare(b.supplier));
      return sort.direction === 'asc' ? sorted : sorted.reverse();
    }, [sort]);
    return <DataTable {...args} rows={rows} sort={sort} onSortChange={setSort} />;
  },
};

export const Loading: Story = { args: { status: 'pending', rows: [] } };
export const Empty: Story = { args: { rows: [], emptyState: 'No documents match these filters.' } };
export const ErrorState: Story = { args: { status: 'error', rows: [], onRetry: fn() } };
export const WithLoadMore: Story = {
  args: { pagination: { hasNextPage: true, isFetchingNextPage: false, onLoadMore: fn() } },
};
