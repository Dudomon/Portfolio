import { cx } from '@/lib/cx';

import styles from './Spinner.module.css';

export interface SpinnerProps {
  size?: 'sm' | 'md';
  /** When set, the spinner is announced; otherwise it is decorative. */
  label?: string;
}

export function Spinner({ size = 'md', label }: SpinnerProps) {
  return (
    <span
      className={cx(styles.spinner, styles[size])}
      role={label ? 'status' : undefined}
      aria-label={label}
      aria-hidden={label ? undefined : true}
    />
  );
}
