import { describe, expect, it } from 'vitest';
import type { CircuitState } from '@aigw/shared';
import { CircuitBreaker } from '../src/routing/circuit-breaker';

function setup(threshold = 3, cooldownMs = 1_000) {
  let now = 0;
  const transitions: CircuitState[] = [];
  const breaker = new CircuitBreaker(
    { failureThreshold: threshold, cooldownMs },
    (state) => transitions.push(state),
    () => now,
  );
  return { breaker, transitions, advance: (ms: number) => (now += ms) };
}

describe('CircuitBreaker', () => {
  it('opens after the consecutive failure threshold', () => {
    const { breaker } = setup(3);
    breaker.onFailure();
    breaker.onFailure();
    expect(breaker.tryAcquire()).toBe(true);
    breaker.onFailure();
    expect(breaker.currentState).toBe('open');
    expect(breaker.tryAcquire()).toBe(false);
  });

  it('resets the failure count on success', () => {
    const { breaker } = setup(2);
    breaker.onFailure();
    breaker.onSuccess();
    breaker.onFailure();
    expect(breaker.currentState).toBe('closed');
  });

  it('admits exactly one probe after the cooldown', () => {
    const { breaker, advance } = setup(1, 1_000);
    breaker.onFailure();
    advance(999);
    expect(breaker.tryAcquire()).toBe(false);
    advance(1);
    expect(breaker.tryAcquire()).toBe(true);
    expect(breaker.currentState).toBe('half_open');
    expect(breaker.tryAcquire()).toBe(false);
  });

  it('closes when the probe succeeds', () => {
    const { breaker, advance, transitions } = setup(1, 10);
    breaker.onFailure();
    advance(10);
    breaker.tryAcquire();
    breaker.onSuccess();
    expect(transitions).toEqual(['open', 'half_open', 'closed']);
    expect(breaker.tryAcquire()).toBe(true);
  });

  it('reopens with a fresh cooldown when the probe fails', () => {
    const { breaker, advance } = setup(1, 10);
    breaker.onFailure();
    advance(10);
    breaker.tryAcquire();
    breaker.onFailure();
    expect(breaker.currentState).toBe('open');
    advance(9);
    expect(breaker.tryAcquire()).toBe(false);
  });

  it('frees the probe slot on a neutral outcome', () => {
    const { breaker, advance } = setup(1, 10);
    breaker.onFailure();
    advance(10);
    expect(breaker.tryAcquire()).toBe(true);
    breaker.release();
    expect(breaker.currentState).toBe('half_open');
    expect(breaker.tryAcquire()).toBe(true);
  });
});
