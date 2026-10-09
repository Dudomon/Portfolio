import { forwardRef, type InputHTMLAttributes } from 'react';

import { cx } from '@/lib/cx';

import { Field, type FieldChromeProps } from '../Field/Field';
import fieldStyles from '../Field/Field.module.css';

export interface TextFieldProps
  extends FieldChromeProps, Omit<InputHTMLAttributes<HTMLInputElement>, 'size'> {
  size?: 'sm' | 'md';
  /** Class for the outer wrapper; `inputClassName` targets the input itself. */
  className?: string;
  inputClassName?: string;
}

export const TextField = forwardRef<HTMLInputElement, TextFieldProps>(function TextField(
  {
    label,
    hideLabel,
    hint,
    error,
    labelAccessory,
    size = 'md',
    className,
    inputClassName,
    id,
    type = 'text',
    ...inputProps
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
        <input
          ref={ref}
          id={controlId}
          type={type}
          aria-invalid={error ? true : undefined}
          aria-describedby={describedBy}
          className={cx(fieldStyles.control, size === 'sm' && fieldStyles.sm, inputClassName)}
          {...inputProps}
        />
      )}
    </Field>
  );
});
