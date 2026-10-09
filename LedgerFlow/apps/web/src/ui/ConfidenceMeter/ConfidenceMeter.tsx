import { cx } from '@/lib/cx';

import styles from './ConfidenceMeter.module.css';

export type ConfidenceLevel = 'high' | 'medium' | 'low' | 'unknown';

export interface ConfidenceMeterProps {
  /** Model confidence in 0..1, or null when not yet extracted. */
  value: number | null;
  /** At or above this, a value counts as trusted (tenant default: 0.85). */
  threshold?: number;
  /** Below this, a value is flagged as low rather than borderline. */
  lowThreshold?: number;
  /** Prefix for the accessible name, e.g. the field it describes. */
  label?: string;
  showValue?: boolean;
  className?: string;
}

export function confidenceLevel(
  value: number | null,
  threshold = 0.85,
  lowThreshold = 0.6,
): ConfidenceLevel {
  if (value === null || Number.isNaN(value)) return 'unknown';
  if (value >= threshold) return 'high';
  if (value >= lowThreshold) return 'medium';
  return 'low';
}

const LEVEL_TEXT: Record<ConfidenceLevel, string> = {
  high: 'high',
  medium: 'needs check',
  low: 'low',
  unknown: 'unknown',
};

export function ConfidenceMeter({
  value,
  threshold = 0.85,
  lowThreshold = 0.6,
  label = 'Confidence',
  showValue = true,
  className,
}: ConfidenceMeterProps) {
  const level = confidenceLevel(value, threshold, lowThreshold);
  const clamped = value === null ? 0 : Math.min(1, Math.max(0, value));
  const percent = Math.round(clamped * 100);
  const valueText = value === null ? 'Not available' : `${percent}% (${LEVEL_TEXT[level]})`;

  return (
    <span className={cx(styles.meter, className)} data-level={level}>
      <span
        role="meter"
        aria-label={label}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={value === null ? undefined : percent}
        aria-valuetext={valueText}
        className={styles.track}
      >
        <span className={styles.fill} style={{ width: `${percent}%` }} />
        <span
          className={styles.threshold}
          style={{ left: `${threshold * 100}%` }}
          aria-hidden="true"
        />
      </span>
      {showValue && (
        <span className={styles.value} aria-hidden="true">
          {value === null ? '—' : `${percent}%`}
        </span>
      )}
    </span>
  );
}
