import { forwardRef, type ButtonHTMLAttributes, type ReactNode } from 'react';

import { cx } from '@/lib/cx';

import styles from './IconButton.module.css';

export interface IconButtonProps extends Omit<
  ButtonHTMLAttributes<HTMLButtonElement>,
  'aria-label' | 'children'
> {
  /** Accessible name. Required because the button has no visible text. */
  label: string;
  icon: ReactNode;
  size?: 'sm' | 'md';
  variant?: 'ghost' | 'secondary';
}

export const IconButton = forwardRef<HTMLButtonElement, IconButtonProps>(function IconButton(
  { label, icon, size = 'md', variant = 'ghost', className, type = 'button', title, ...rest },
  ref,
) {
  return (
    <button
      ref={ref}
      type={type}
      aria-label={label}
      title={title ?? label}
      className={cx(styles.iconButton, styles[size], styles[variant], className)}
      {...rest}
    >
      {icon}
    </button>
  );
});
