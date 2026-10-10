'use client';

import { useActionState } from 'react';
import { CreatedKeyNotice } from './CreatedKeyNotice';
import ui from './ui.module.css';

export interface CreateKeyFormState {
  /** Plaintext key, present only in the response to a successful create. */
  secret: string | null;
  keyName: string | null;
  error: string | null;
}

type CreateKeyAction = (
  state: CreateKeyFormState,
  formData: FormData,
) => Promise<CreateKeyFormState>;

const initialState: CreateKeyFormState = { secret: null, keyName: null, error: null };

/** The action is injected so the form can be rendered and tested without a server. */
export function CreateKeyForm({ action }: { action: CreateKeyAction }) {
  const [state, formAction, pending] = useActionState(action, initialState);

  return (
    <div className={ui.form}>
      <form action={formAction} className={ui.form} aria-label="Create API key">
        <div className={ui.formRow}>
          <label className={ui.field}>
            Name
            <input
              className={ui.input}
              name="name"
              required
              maxLength={100}
              placeholder="checkout-service"
            />
          </label>
          <label className={ui.field}>
            Project
            <input
              className={ui.input}
              name="project"
              required
              maxLength={100}
              placeholder="storefront"
            />
          </label>
          <label className={ui.field}>
            Monthly budget (USD)
            <input
              className={ui.input}
              name="monthlyBudgetUsd"
              type="number"
              min="0.01"
              step="0.01"
              placeholder="No limit"
            />
          </label>
          <label className={ui.field}>
            Requests per minute
            <input
              className={ui.input}
              name="rateLimitPerMinute"
              type="number"
              min="1"
              step="1"
              defaultValue={60}
            />
          </label>
          <button className={ui.button} type="submit" disabled={pending}>
            {pending ? 'Creating…' : 'Create key'}
          </button>
        </div>
        {state.error ? (
          <p role="alert" className={ui.error}>
            {state.error}
          </p>
        ) : null}
      </form>
      {state.secret && state.keyName ? (
        <CreatedKeyNotice secret={state.secret} keyName={state.keyName} />
      ) : null}
    </div>
  );
}
