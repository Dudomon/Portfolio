import type { CSSProperties } from 'react';

import { cx } from '@/lib/cx';

import styles from './Skeleton.module.css';

export interface SkeletonProps {
  width?: CSSProperties['width'];
  height?: CSSProperties['height'];
  shape?: 'text' | 'rect' | 'circle';
  className?: string;
}

/** Decorative placeholder; pair with an aria-busy container or a status message. */
export function Skeleton({ width = '100%', height, shape = 'text', className }: SkeletonProps) {
  return (
    <span
      className={cx(styles.skeleton, styles[shape], className)}
      style={{ width, height }}
      aria-hidden="true"
    />
  );
}

export function SkeletonText({ lines = 3 }: { lines?: number }) {
  return (
    <span className={styles.stack} aria-hidden="true">
      {Array.from({ length: lines }, (_, index) => (
        <Skeleton key={index} width={index === lines - 1 ? '60%' : '100%'} />
      ))}
    </span>
  );
}
