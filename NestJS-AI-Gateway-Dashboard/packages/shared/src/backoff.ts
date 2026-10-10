export interface BackoffOptions {
  baseMs: number;
  maxMs: number;
}

/**
 * "Full jitter" exponential backoff: a uniformly random delay in
 * [0, min(max, base * 2^attempt)]. Randomizing the whole window (rather than
 * adding small jitter) spreads synchronized retries from many clients, which
 * is what prevents retry storms against a recovering provider.
 *
 * @param attempt zero-based retry number
 * @param random injectable for deterministic tests
 */
export function fullJitterBackoff(
  attempt: number,
  { baseMs, maxMs }: BackoffOptions,
  random: () => number = Math.random,
): number {
  const ceiling = Math.min(maxMs, baseMs * 2 ** attempt);
  return Math.floor(random() * ceiling);
}

/**
 * Parses a Retry-After header (delta-seconds or HTTP date) into milliseconds.
 * Returns undefined when absent or unparsable.
 */
export function parseRetryAfter(
  value: string | null | undefined,
  now = Date.now(),
): number | undefined {
  if (!value) return undefined;
  const seconds = Number(value);
  if (Number.isFinite(seconds)) return Math.max(0, seconds * 1000);
  const date = Date.parse(value);
  return Number.isNaN(date) ? undefined : Math.max(0, date - now);
}

/** The signal's abort reason as an Error (reasons may be any value). */
export function abortReason(signal: AbortSignal): Error {
  const reason: unknown = signal.reason;
  return reason instanceof Error ? reason : new Error('Aborted', { cause: reason });
}

/** Resolves after `ms`, or rejects early if the signal aborts. */
export function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(abortReason(signal));
      return;
    }
    const onAbort = (): void => {
      clearTimeout(timer);
      if (signal) reject(abortReason(signal));
    };
    const timer = setTimeout(() => {
      signal?.removeEventListener('abort', onAbort);
      resolve();
    }, ms);
    signal?.addEventListener('abort', onAbort, { once: true });
  });
}
