import { describe, expect, it } from 'vitest';
import { fullJitterBackoff, parseRetryAfter } from '../src/backoff';

describe('fullJitterBackoff', () => {
  const options = { baseMs: 100, maxMs: 1_000 };

  it('scales the window exponentially', () => {
    const top = () => 0.999_999;
    expect(fullJitterBackoff(0, options, top)).toBe(99);
    expect(fullJitterBackoff(2, options, top)).toBe(399);
  });

  it('caps the window at maxMs', () => {
    expect(fullJitterBackoff(20, options, () => 0.999_999)).toBe(999);
  });

  it('can return zero (full jitter, not equal jitter)', () => {
    expect(fullJitterBackoff(3, options, () => 0)).toBe(0);
  });
});

describe('parseRetryAfter', () => {
  it('parses delta seconds', () => {
    expect(parseRetryAfter('2')).toBe(2_000);
  });

  it('parses HTTP dates relative to now', () => {
    const now = Date.parse('2025-01-01T00:00:00Z');
    expect(parseRetryAfter('Wed, 01 Jan 2025 00:00:05 GMT', now)).toBe(5_000);
  });

  it('returns undefined for missing or garbage values', () => {
    expect(parseRetryAfter(null)).toBeUndefined();
    expect(parseRetryAfter('soon')).toBeUndefined();
  });
});
