import { useId, type ReactNode } from 'react';

import { cx } from '@/lib/cx';

import styles from './Field.module.css';

export interface FieldChromeProps {
  label: string;
  /** Keeps the label for assistive tech but hides it visually (e.g. inside table cells). */
  hideLabel?: boolean;
  hint?: ReactNode;
  error?: string;
  /** Rendered next to the label, e.g. a confidence indicator. */
  labelAccessory?: ReactNode;
}

export interface FieldIds {
  controlId: string;
  describedBy: string | undefined;
}

interface FieldProps extends FieldChromeProps {
  id?: string;
  className?: string;
  children: (ids: FieldIds) => ReactNode;
}

/** Shared label / hint / error layout for form controls, wiring up aria-describedby. */
export function Field({
  id,
  label,
  hideLabel,
  hint,
  error,
  labelAccessory,
  className,
  children,
}: FieldProps) {
  const generatedId = useId();
  const controlId = id ?? generatedId;
  const hintId = hint ? `${controlId}-hint` : undefined;
  const errorId = error ? `${controlId}-error` : undefined;
  const describedBy = [errorId, hintId].filter(Boolean).join(' ') || undefined;

  return (
    <div className={cx(styles.field, className)}>
      <div className={cx(styles.labelRow, hideLabel && 'visually-hidden')}>
        <label htmlFor={controlId} className={styles.label}>
          {label}
        </label>
        {labelAccessory}
      </div>
      {children({ controlId, describedBy })}
      {error && (
        <p id={errorId} className={styles.error}>
          {error}
        </p>
      )}
      {hint && !error && (
        <p id={hintId} className={styles.hint}>
          {hint}
        </p>
      )}
    </div>
  );
}
