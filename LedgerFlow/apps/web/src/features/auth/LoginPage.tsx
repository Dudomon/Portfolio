import { zodResolver } from '@hookform/resolvers/zod';
import { useMutation } from '@tanstack/react-query';
import { useForm } from 'react-hook-form';
import { Navigate, useLocation } from 'react-router-dom';
import { z } from 'zod';

import { ApiError, describeError, useApi } from '@/api';
import { Button, Callout, TextField } from '@/ui';

import { useAuth } from './AuthProvider';
import type { LoginRedirectState } from './guards';
import styles from './LoginPage.module.css';

const LoginSchema = z.object({
  email: z.string().trim().min(1, 'Enter your email.').email('Enter a valid email address.'),
  password: z.string().min(1, 'Enter your password.'),
});
type LoginValues = z.infer<typeof LoginSchema>;

const DEMO_ACCOUNTS = ['reviewer@demo.test', 'admin@demo.test', 'viewer@demo.test'];

function loginErrorMessage(error: unknown): string {
  if (error instanceof ApiError && (error.status === 400 || error.status === 401)) {
    return 'Email or password is incorrect.';
  }
  return describeError(error);
}

export function LoginPage() {
  const { state, signIn } = useAuth();
  const location = useLocation();
  const api = useApi();
  // Signing in is part of the mutation so a malformed token surfaces as a login error too.
  const login = useMutation({
    mutationFn: async (values: LoginValues) => signIn(await api.login(values)),
  });
  const redirect = (location.state ?? {}) as LoginRedirectState;

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<LoginValues>({
    resolver: zodResolver(LoginSchema),
    defaultValues: { email: '', password: '' },
  });

  if (state.session) return <Navigate to={redirect.from ?? '/documents'} replace />;

  // On success the session changes and the early return above redirects.
  const onSubmit = handleSubmit((values) => login.mutate(values));

  return (
    <main className={styles.page}>
      <section className={styles.card} aria-labelledby="login-title">
        <header className={styles.header}>
          <span className={styles.logo} aria-hidden="true">
            L
          </span>
          <h1 id="login-title" className={styles.title}>
            Sign in to LedgerFlow
          </h1>
        </header>

        {redirect.reason === 'expired' && (
          <Callout tone="warning" title="Your session expired">
            Sign in again to continue where you left off.
          </Callout>
        )}
        {login.isError && <Callout tone="danger" title={loginErrorMessage(login.error)} />}

        <form className={styles.form} onSubmit={onSubmit} noValidate>
          <TextField
            label="Email"
            type="email"
            autoComplete="username"
            error={errors.email?.message}
            {...register('email')}
          />
          <TextField
            label="Password"
            type="password"
            autoComplete="current-password"
            error={errors.password?.message}
            {...register('password')}
          />
          <Button type="submit" variant="primary" loading={login.isPending}>
            Sign in
          </Button>
        </form>

        {import.meta.env.VITE_USE_MOCKS === 'true' && (
          <p className={styles.demo}>
            Mock API: sign in as {DEMO_ACCOUNTS.join(', ')} with any password.
          </p>
        )}
      </section>
    </main>
  );
}
