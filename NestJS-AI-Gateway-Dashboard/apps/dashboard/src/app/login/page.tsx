import type { Metadata } from 'next';
import { LoginForm } from './LoginForm';
import styles from './login.module.css';

export const metadata: Metadata = { title: 'Sign in' };

export default async function LoginPage({
  searchParams,
}: {
  searchParams: Promise<{ expired?: string }>;
}) {
  const { expired } = await searchParams;
  return (
    <main className={styles.page}>
      <section className={styles.card} aria-labelledby="login-title">
        <h1 id="login-title" className={styles.title}>
          AI Gateway
        </h1>
        <p className={styles.subtitle}>Sign in with the gateway&apos;s admin token.</p>
        <LoginForm expired={expired === '1'} />
      </section>
    </main>
  );
}
