import type { ReactNode } from 'react';
import styles from './shell.module.css';

export function PageHeader({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <header className={styles.pageHeader}>
      <h1 className={styles.pageTitle}>{title}</h1>
      {children ? <p className={styles.pageDescription}>{children}</p> : null}
    </header>
  );
}
