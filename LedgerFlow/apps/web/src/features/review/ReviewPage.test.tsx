import { screen, waitFor, within } from '@testing-library/react';

import { db } from '@/mocks/db';
import { mockUuid } from '@/mocks/fixtures';
import { renderRoutes, userWithRole } from '@/test/render';

import { ReviewPage } from './ReviewPage';

// The first fixture document is in `needs_review`.
const DOCUMENT_ID = mockUuid(1, 1);

function renderReview(role: 'reviewer' | 'viewer' = 'reviewer') {
  return renderRoutes(
    [
      { path: '/documents/:documentId', element: <ReviewPage /> },
      { path: '/documents', element: <p>Document list</p> },
    ],
    { path: `/documents/${DOCUMENT_ID}`, as: userWithRole(role) },
  );
}

async function openReview(role?: 'reviewer' | 'viewer') {
  const view = renderReview(role);
  const filename = db.find(DOCUMENT_ID)!.detail.filename;
  await screen.findByRole('heading', { level: 1, name: filename });
  return view;
}

const stored = () => db.find(DOCUMENT_ID)!.detail;

describe('Review workbench', () => {
  beforeEach(() => {
    // Previewed as an image: the test DOM cannot render a PDF frame.
    db.find(DOCUMENT_ID)!.detail.content_type = 'image/png';
  });

  it('shows extracted fields with confidence and source', async () => {
    await openReview();
    const supplier = screen.getByRole('textbox', { name: 'Supplier name' });

    expect(supplier).toHaveValue(stored().extraction!.fields.supplier_name!.value as string);
    expect(screen.getByRole('meter', { name: 'Supplier name confidence' })).toBeInTheDocument();
    expect(screen.getAllByText(/below 85% confidence/)).not.toHaveLength(0);
  });

  it('saves a corrected field as a human-sourced value', async () => {
    const { user } = await openReview();
    const supplier = screen.getByRole('textbox', { name: 'Supplier name' });

    await user.clear(supplier);
    await user.type(supplier, 'Acme Componentes Ltda');
    expect(screen.getByRole('button', { name: 'Approve' })).toBeDisabled();
    await user.click(screen.getByRole('button', { name: 'Save corrections' }));

    expect(await screen.findByText('Corrections saved')).toBeInTheDocument();
    expect(stored().version).toBe(2);
    expect(stored().extraction!.fields.supplier_name).toEqual({
      value: 'Acme Componentes Ltda',
      confidence: 1,
      source: 'human',
    });
    // Only the edited field is sent; others keep their machine source.
    expect(stored().extraction!.fields.invoice_number!.source).not.toBe('human');
    await waitFor(() => expect(screen.getByRole('button', { name: 'Approve' })).toBeEnabled());
  });

  it('blocks invalid input with field-level errors', async () => {
    const { user } = await openReview();
    const total = screen.getByRole('textbox', { name: 'Total amount' });

    await user.clear(total);
    await user.type(total, '1.530,40');
    await user.click(screen.getByRole('button', { name: 'Save corrections' }));

    expect(await screen.findByText(/Use digits with a dot for decimals/)).toBeInTheDocument();
    expect(total).toHaveAttribute('aria-invalid', 'true');
    expect(stored().version).toBe(1);
  });

  it('approves the document', async () => {
    const { user } = await openReview();
    await user.click(screen.getByRole('button', { name: 'Approve' }));

    await waitFor(() => expect(stored().status).toBe('approved'));
    expect(await screen.findByText('Document approved')).toBeInTheDocument();
  });

  it('approves with the "a" shortcut', async () => {
    const { user } = await openReview();
    await user.keyboard('a');
    await waitFor(() => expect(stored().status).toBe('approved'));
  });

  it('offers a reload when someone else changed the document (412)', async () => {
    const { user } = await openReview();
    // A concurrent edit lands on the server after this page loaded its copy.
    db.update(DOCUMENT_ID, (detail) => ({
      ...detail,
      extraction: {
        ...detail.extraction!,
        fields: {
          ...detail.extraction!.fields,
          invoice_number: { value: 'INV-CONCURRENT', confidence: 1, source: 'human' },
        },
      },
    }));

    const supplier = screen.getByRole('textbox', { name: 'Supplier name' });
    await user.clear(supplier);
    await user.type(supplier, 'Stale Edit SA');
    await user.click(screen.getByRole('button', { name: 'Save corrections' }));

    const dialog = await screen.findByRole('alertdialog', {
      name: 'This document was changed by someone else',
    });
    // The optimistic update was rolled back and nothing was persisted.
    expect(stored().extraction!.fields.supplier_name!.value).not.toBe('Stale Edit SA');

    await user.click(within(dialog).getByRole('button', { name: 'Reload latest' }));

    await waitFor(() => expect(screen.queryByRole('alertdialog')).not.toBeInTheDocument());
    expect(screen.getByRole('textbox', { name: 'Invoice number' })).toHaveValue('INV-CONCURRENT');
    expect(screen.getByRole('textbox', { name: 'Supplier name' })).not.toHaveValue('Stale Edit SA');
  });

  it('is read-only for viewers', async () => {
    await openReview('viewer');
    expect(screen.getByRole('textbox', { name: 'Supplier name' })).toHaveAttribute('readonly');
    expect(screen.queryByRole('button', { name: 'Approve' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Save corrections' })).not.toBeInTheDocument();
  });
});
