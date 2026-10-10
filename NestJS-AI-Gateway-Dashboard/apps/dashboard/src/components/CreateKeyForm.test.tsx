import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { CreateKeyForm, type CreateKeyFormState } from './CreateKeyForm';

async function fillAndSubmit() {
  const user = userEvent.setup();
  await user.type(screen.getByLabelText('Name'), 'checkout');
  await user.type(screen.getByLabelText('Project'), 'storefront');
  await user.type(screen.getByLabelText('Monthly budget (USD)'), '25');
  await user.click(screen.getByRole('button', { name: 'Create key' }));
}

describe('CreateKeyForm', () => {
  it('submits the form fields and shows the new key exactly once', async () => {
    const action = vi.fn(
      (_state: CreateKeyFormState, _data: FormData): Promise<CreateKeyFormState> =>
        Promise.resolve({ secret: 'gw_0123456789ab_secret', keyName: 'checkout', error: null }),
    );
    render(<CreateKeyForm action={action} />);

    await fillAndSubmit();

    expect(await screen.findByLabelText('New API key')).toHaveTextContent('gw_0123456789ab_secret');
    expect(screen.getByText(/will not be shown again/i)).toBeInTheDocument();
    const submitted = action.mock.calls[0]?.[1];
    expect(submitted?.get('name')).toBe('checkout');
    expect(submitted?.get('monthlyBudgetUsd')).toBe('25');
    expect(submitted?.get('rateLimitPerMinute')).toBe('60');
  });

  it('announces validation errors from the gateway', async () => {
    const action = vi.fn((): Promise<CreateKeyFormState> =>
      Promise.resolve({ secret: null, keyName: null, error: 'name: Too small' }),
    );
    render(<CreateKeyForm action={action} />);

    await fillAndSubmit();

    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent('name: Too small'));
    expect(screen.queryByLabelText('New API key')).not.toBeInTheDocument();
  });
});
