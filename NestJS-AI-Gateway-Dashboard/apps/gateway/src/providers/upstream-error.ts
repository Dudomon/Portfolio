import { parseRetryAfter, type ProviderId } from '@aigw/shared';

/**
 * Why an upstream attempt failed. The kind, not the raw status, drives routing:
 *
 * - `client`: the request itself is invalid; every provider would reject it,
 *   so the router fails fast without fallback.
 * - `misconfigured`: bad credentials or an unknown model/deployment (401, 403,
 *   404); a config problem with this target that another target may not share.
 * - `rate_limit`, `server`, `timeout`, `network`: transient or capacity
 *   problems; eligible for retry and fallback, and they trip the breaker.
 */
export type UpstreamErrorKind =
  'client' | 'misconfigured' | 'rate_limit' | 'server' | 'timeout' | 'network';

export class UpstreamError extends Error {
  constructor(
    readonly provider: ProviderId,
    readonly kind: UpstreamErrorKind,
    message: string,
    readonly status: number | null = null,
    readonly retryAfterMs: number | undefined = undefined,
    options?: { cause?: unknown },
  ) {
    super(message, options);
    this.name = 'UpstreamError';
  }

  /**
   * True when the failure is attributable to the provider rather than the
   * request. Such errors trigger fallback and count toward the circuit
   * breaker; client errors do neither, since they say nothing about provider
   * health and would fail identically everywhere.
   */
  get isProviderFault(): boolean {
    return this.kind !== 'client';
  }

  /** Same-target retries only make sense for blips; 429s and config errors won't clear in milliseconds. */
  get retryableOnSameTarget(): boolean {
    return this.kind === 'server' || this.kind === 'timeout' || this.kind === 'network';
  }

  static fromStatus(
    provider: ProviderId,
    status: number,
    message: string,
    retryAfterHeader?: string | null,
  ): UpstreamError {
    return new UpstreamError(
      provider,
      kindForStatus(status),
      `${provider} returned ${status}: ${message}`,
      status,
      parseRetryAfter(retryAfterHeader),
    );
  }

  static network(provider: ProviderId, cause: unknown): UpstreamError {
    const detail = cause instanceof Error ? cause.message : String(cause);
    return new UpstreamError(
      provider,
      'network',
      `${provider} network error: ${detail}`,
      null,
      undefined,
      {
        cause,
      },
    );
  }
}

export function kindForStatus(status: number): UpstreamErrorKind {
  if (status === 401 || status === 403 || status === 404) return 'misconfigured';
  if (status === 408) return 'timeout';
  if (status === 429) return 'rate_limit';
  if (status >= 500) return 'server';
  return 'client';
}

/** Extracts a human-readable message from a provider error body of any shape. */
export async function readErrorMessage(response: Response): Promise<string> {
  const text = await response.text().catch(() => '');
  try {
    const body = JSON.parse(text) as { error?: { message?: unknown } | string; message?: unknown };
    if (typeof body.error === 'object' && typeof body.error.message === 'string') {
      return body.error.message;
    }
    if (typeof body.error === 'string') return body.error;
    if (typeof body.message === 'string') return body.message;
  } catch {
    // Not JSON; fall through to the raw text.
  }
  return text.slice(0, 500) || response.statusText;
}
