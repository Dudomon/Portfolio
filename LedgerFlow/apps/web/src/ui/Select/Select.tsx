import { forwardRef, type SelectHTMLAttributes } from 'react';

import { cx } from '@/lib/cx';

import { Field, type FieldChromeProps } from '../Field/Field';
import fieldStyles from '../Field/Field.module.css';
import styles from './Select.module.css';

export interface SelectOption {
  value: string;
  label: string;
}

export interface SelectProps
  extends FieldChromeProps, Omit<SelectHTMLAttributes<HTMLSelectElement>, 'size'> {
  options: readonly SelectOption[];
  /** Adds a leading empty option, e.g. "All statuses". */
  placeholder?: string;
  size?: 'sm' | 'md';
}

/**
 * A styled native `<select>`: keeps platform keyboard behaviour, mobile pickers
 * and form semantics for free.
 */
export const Select = forwardRef<HTMLSelectElement, SelectProps>(function Select(
  {
    label,
    hideLabel,
    hint,
    error,
    labelAccessory,
    options,
    placeholder,
    size = 'md',
    className,
    id,
    ...selectProps
  },
  ref,
) {
  return (
    <Field
      id={id}
      label={label}
      hideLabel={hideLabel}
      hint={hint}
      error={error}
      labelAccessory={labelAccessory}
      className={className}
    >
      {({ controlId, describedBy }) => (
        <div className={styles.wrapper}>
          <select
            ref={ref}
            id={controlId}
            aria-invalid={error ? true : undefined}
            aria-describedby={describedBy}
            className={cx(fieldStyles.control, styles.select, size === 'sm' && fieldStyles.sm)}
            {...selectProps}
          >
            {placeholder !== undefined && <option value="">{placeholder}</option>}
            {options.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>
      )}
    </Field>
  );
});
