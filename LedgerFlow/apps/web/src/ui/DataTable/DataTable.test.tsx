import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { DataTable, type Column, type DataTableProps } from './DataTable';

interface Row {
  id: string;
  name: string;
  amount: string;
}

const rows: Row[] = [
  { id: 'a', name: 'Acme', amount: '10.00' },
  { id: 'b', name: 'Borealis', amount: '20.00' },
  { id: 'c', name: 'Cobalt', amount: '30.00' },
];

const columns: Column<Row, 'name' | 'amount'>[] = [
  { id: 'name', header: 'Name', cell: (row) => row.name, sortable: true },
  { id: 'amount', header: 'Amount', cell: (row) => row.amount, align: 'end' },
];

function renderTable(props: Partial<DataTableProps<Row, 'name' | 'amount'>> = {}) {
  const user = userEvent.setup();
  render(
    <DataTable
      caption="Suppliers"
      columns={columns}
      rows={rows}
      getRowId={(row) => row.id}
      status="success"
      {...props}
    />,
  );
  return { user };
}

const bodyRows = () => within(screen.getAllByRole('rowgroup')[1]!).getAllByRole('row');

describe('DataTable', () => {
  it('renders an accessible table with typed cells', () => {
    renderTable();
    const table = screen.getByRole('table', { name: 'Suppliers' });
    expect(
      within(table)
        .getAllByRole('columnheader')
        .map((th) => th.textContent),
    ).toEqual(['Name', 'Amount']);
    expect(bodyRows()).toHaveLength(3);
    expect(screen.getByRole('cell', { name: 'Borealis' })).toBeInTheDocument();
  });

  it('reports sort intent and reflects it with aria-sort', async () => {
    const onSortChange = vi.fn();
    const { user } = renderTable({ onSortChange, sort: { columnId: 'name', direction: 'asc' } });

    const header = screen.getByRole('columnheader', { name: /name/i });
    expect(header).toHaveAttribute('aria-sort', 'ascending');
    expect(screen.getByRole('columnheader', { name: 'Amount' })).not.toHaveAttribute('aria-sort');

    await user.click(within(header).getByRole('button'));
    expect(onSortChange).toHaveBeenCalledWith({ columnId: 'name', direction: 'desc' });
  });

  it('starts sorting a new column ascending', async () => {
    const onSortChange = vi.fn();
    const { user } = renderTable({ onSortChange });
    expect(screen.getByRole('columnheader', { name: /name/i })).toHaveAttribute(
      'aria-sort',
      'none',
    );
    await user.click(screen.getByRole('button', { name: /name/i }));
    expect(onSortChange).toHaveBeenCalledWith({ columnId: 'name', direction: 'asc' });
  });

  it('supports keyboard navigation with a single tab stop', async () => {
    const onRowActivate = vi.fn();
    const { user } = renderTable({ onRowActivate });
    const [first, second, third] = bodyRows();

    await user.tab();
    expect(first).toHaveFocus();
    expect(bodyRows().filter((row) => row.tabIndex === 0)).toHaveLength(1);

    await user.keyboard('{ArrowDown}');
    expect(second).toHaveFocus();
    await user.keyboard('{End}');
    expect(third).toHaveFocus();
    await user.keyboard('{ArrowDown}');
    expect(third).toHaveFocus();
    await user.keyboard('{Home}');
    expect(first).toHaveFocus();

    await user.keyboard('{ArrowDown}{Enter}');
    expect(onRowActivate).toHaveBeenCalledWith(rows[1]);
  });

  it('activates rows on click', async () => {
    const onRowActivate = vi.fn();
    const { user } = renderTable({ onRowActivate });
    await user.click(screen.getByText('Cobalt'));
    expect(onRowActivate).toHaveBeenCalledWith(rows[2]);
  });

  it('shows a busy skeleton while loading', () => {
    renderTable({ status: 'pending', rows: [], skeletonRows: 4 });
    expect(screen.getByRole('table')).toHaveAttribute('aria-busy', 'true');
    expect(bodyRows()).toHaveLength(4);
  });

  it('shows the empty state', () => {
    renderTable({ rows: [], emptyState: 'No documents match these filters.' });
    expect(screen.getByText('No documents match these filters.')).toBeInTheDocument();
  });

  it('shows the error with a retry action', async () => {
    const onRetry = vi.fn();
    const { user } = renderTable({
      status: 'error',
      rows: [],
      errorMessage: 'Server unavailable',
      onRetry,
    });
    expect(screen.getByRole('alert')).toHaveTextContent('Server unavailable');
    await user.click(screen.getByRole('button', { name: 'Retry' }));
    expect(onRetry).toHaveBeenCalledOnce();
  });

  it('offers "Load more" only while more pages exist', async () => {
    const onLoadMore = vi.fn();
    const { user } = renderTable({
      pagination: { hasNextPage: true, isFetchingNextPage: false, onLoadMore },
    });
    await user.click(screen.getByRole('button', { name: 'Load more' }));
    expect(onLoadMore).toHaveBeenCalledOnce();
  });

  it('hides "Load more" on the last page', () => {
    renderTable({
      pagination: { hasNextPage: false, isFetchingNextPage: false, onLoadMore: vi.fn() },
    });
    expect(screen.queryByRole('button', { name: 'Load more' })).not.toBeInTheDocument();
  });
});
