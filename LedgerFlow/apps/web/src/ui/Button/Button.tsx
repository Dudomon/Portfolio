import { forwardRef, type ButtonHTMLAttributes, type ReactNode } from 'react';

import { cx } from '@/lib/cx';

import { Spinner } from '../Spinner/Spinner';
import styles from './Button.module.css';

export type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger';
export type ButtonSize = 'sm' | 'md';

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  /** Shows a spinner and blocks interaction while keeping the label for screen readers. */
  loading?: boolean;
  startIcon?: ReactNode;
  /** Keyboard shortcut hint, rendered as `<kbd>` and announced via aria-keyshortcuts. */
  shortcut?: string;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  {
    variant = 'secondary',
    size = 'md',
    loading = false,
    startIcon,
    shortcut,
    disabled,
    className,
    children,
    type = 'button',
    ...rest
  },
  ref,
) {
  return (
    <button
      ref={ref}
      type={type}
      className={cx(styles.button, styles[variant], styles[size], className)}
      disabled={disabled === true || loading}
      aria-busy={loading || undefined}
      aria-keyshortcuts={shortcut}
      {...rest}
    >
      {loading ? <Spinner size="sm" /> : startIcon}
      <span>{children}</span>
      {shortcut && (
        <kbd className={styles.kbd} aria-hidden="true">
          {shortcut}
        </kbd>
      )}
    </button>
  );
});
