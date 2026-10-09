import type { ReactNode } from 'react';

import { cx } from '@/lib/cx';

import styles from './Callout.module.css';

export interface CalloutProps {
  tone?: 'info' | 'warning' | 'danger' | 'success';
  title: string;
  children?: ReactNode;
  /** Buttons rendered at the end, e.g. "Retry". */
  actions?: ReactNode;
  className?: string;
}

/** Inline, persistent message. Danger callouts are announced immediately. */
export function Callout({ tone = 'info', title, children, actions, className }: CalloutProps) {
  return (
    <div
      role={tone === 'danger' ? 'alert' : 'status'}
      className={cx(styles.callout, styles[tone], className)}
    >
      <div className={styles.body}>
        <p className={styles.title}>{title}</p>
        {children && <div className={styles.content}>{children}</div>}
      </div>
      {actions && <div className={styles.actions}>{actions}</div>}
    </div>
  );
}
