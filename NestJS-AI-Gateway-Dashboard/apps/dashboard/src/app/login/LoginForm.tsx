'use client';

import { useActionState } from 'react';
import ui from '@/components/ui.module.css';
import { login, type LoginState } from './actions';

const initialState: LoginState = { error: null };

export function LoginForm({ expired }: { expired: boolean }) {
  const [state, action, pending] = useActionState(login, initialState);
  const message = state.error ?? (expired ? 'Your session expired. Sign in again.' : null);

  return (
    <form action={action} className={ui.form}>
      <label className={ui.field}>
        Admin token
        <input
          className={ui.input}
          type="password"
          name="token"
          autoComplete="current-password"
          required
          aria-describedby={message ? 'login-error' : undefined}
          aria-invalid={state.error ? true : undefined}
        />
      </label>
      {message ? (
        <p id="login-error" role="alert" className={ui.error}>
          {message}
        </p>
      ) : null}
      <button className={ui.button} type="submit" disabled={pending}>
        {pending ? 'Checking…' : 'Sign in'}
      </button>
    </form>
  );
}
