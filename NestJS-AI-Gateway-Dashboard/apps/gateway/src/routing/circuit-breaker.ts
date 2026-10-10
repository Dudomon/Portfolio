import type { CircuitState } from '@aigw/shared';

export interface CircuitBreakerOptions {
  failureThreshold: number;
  cooldownMs: number;
}

/**
 * Classic three-state breaker.
 *
 * - closed: calls flow; consecutive provider faults are counted.
 * - open: calls are rejected immediately until `cooldownMs` elapses, so a
 *   failing provider costs callers nothing instead of a full timeout.
 * - half_open: exactly one probe is admitted; its outcome closes or reopens.
 *
 * Outcomes that say nothing about provider health (client errors, client
 * aborts) must call `release()` so a half-open probe slot is not leaked.
 */
export class CircuitBreaker {
  private state: CircuitState = 'closed';
  private consecutiveFailures = 0;
  private openedAt = 0;
  private probeInFlight = false;

  constructor(
    private readonly options: CircuitBreakerOptions,
    private readonly onStateChange: (state: CircuitState) => void = () => undefined,
    private readonly now: () => number = Date.now,
  ) {}

  get currentState(): CircuitState {
    return this.state;
  }

  /** Returns true when a call may proceed. */
  tryAcquire(): boolean {
    if (this.state === 'closed') return true;
    if (this.state === 'open') {
      if (this.now() - this.openedAt < this.options.cooldownMs) return false;
      this.transition('half_open');
    }
    if (this.probeInFlight) return false;
    this.probeInFlight = true;
    return true;
  }

  onSuccess(): void {
    this.consecutiveFailures = 0;
    this.probeInFlight = false;
    if (this.state !== 'closed') this.transition('closed');
  }

  onFailure(): void {
    this.probeInFlight = false;
    if (this.state === 'half_open') {
      this.open();
      return;
    }
    this.consecutiveFailures += 1;
    if (this.state === 'closed' && this.consecutiveFailures >= this.options.failureThreshold) {
      this.open();
    }
  }

  /** Ends a call without counting it either way. */
  release(): void {
    this.probeInFlight = false;
  }

  private open(): void {
    this.openedAt = this.now();
    this.consecutiveFailures = 0;
    this.transition('open');
  }

  private transition(state: CircuitState): void {
    this.state = state;
    this.onStateChange(state);
  }
}
