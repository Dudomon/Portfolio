import { describe, expect, it } from 'vitest';
import { formatDay, formatLatency, formatTimestamp, formatUsd } from './format';

describe('formatters', () => {
  it('keeps sub-cent spend visible instead of rounding to $0.00', () => {
    expect(formatUsd(0.000123)).toBe('$0.000123');
    expect(formatUsd(0)).toBe('$0.00');
    expect(formatUsd(1234.5)).toBe('$1,234.50');
  });

  it('switches latency units at one second', () => {
    expect(formatLatency(840)).toBe('840 ms');
    expect(formatLatency(1530)).toBe('1.53 s');
    expect(formatLatency(null)).toBe('—');
  });

  it('formats dates in UTC regardless of the runtime time zone', () => {
    expect(formatDay('2025-03-07')).toBe('Mar 7');
    expect(formatTimestamp('2025-03-07T23:59:01.123Z')).toBe('2025-03-07 23:59:01 UTC');
  });
});
