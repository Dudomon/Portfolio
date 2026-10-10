import type { RouteTarget } from '@aigw/shared';
import { ApiError } from '../common/api-error';
import type { UpstreamError } from '../providers/upstream-error';

/**
 * Routing failures carry how many upstream attempts were made and which
 * target was tried last, so usage records and per-provider error rates stay
 * accurate even for requests that never succeed.
 */
export interface RoutingFailure {
  readonly attempts: number;
  readonly target: RouteTarget | null;
}

export function isRoutingFailure(error: unknown): error is ApiError & RoutingFailure {
  return (
    error instanceof RouteExhaustedError ||
    error instanceof UpstreamRejectedError ||
    error instanceof ClientAbortedError
  );
}

/** Every target in the route failed, was skipped, or had no configured provider. */
export class RouteExhaustedError extends ApiError implements RoutingFailure {
  constructor(
    alias: string,
    readonly attempts: number,
    readonly target: RouteTarget | null,
    readonly lastError: UpstreamError | undefined,
  ) {
    const [status, type, code] = RouteExhaustedError.classify(attempts, lastError);
    super(
      status,
      type,
      attempts === 0
        ? `No provider is currently available for model "${alias}"`
        : `All providers failed for model "${alias}" after ${attempts} attempt(s)`,
      code,
      null,
      status === 503 || status === 429 ? { 'retry-after': '5' } : {},
    );
    this.name = 'RouteExhaustedError';
  }

  private static classify(
    attempts: number,
    lastError: UpstreamError | undefined,
  ): [number, ApiError['type'], string] {
    if (attempts === 0 || !lastError) return [503, 'api_error', 'no_available_provider'];
    switch (lastError.kind) {
      case 'rate_limit':
        return [429, 'rate_limit_error', 'upstream_rate_limited'];
      case 'timeout':
        return [504, 'timeout_error', 'upstream_timeout'];
      default:
        return [502, 'upstream_error', 'upstream_unavailable'];
    }
  }
}

/** A provider rejected the request itself; retrying elsewhere would not help. */
export class UpstreamRejectedError extends ApiError implements RoutingFailure {
  constructor(
    upstream: UpstreamError,
    readonly attempts: number,
    readonly target: RouteTarget,
  ) {
    super(upstream.status ?? 400, 'invalid_request_error', upstream.message, 'upstream_rejected');
    this.name = 'UpstreamRejectedError';
  }
}

/** The client disconnected or canceled; nobody is left to answer. */
export class ClientAbortedError extends ApiError implements RoutingFailure {
  constructor(
    readonly attempts: number,
    readonly target: RouteTarget | null,
  ) {
    // 499 is nginx's "client closed request"; it is only ever logged.
    super(499, 'api_error', 'Client closed request', 'client_closed_request');
    this.name = 'ClientAbortedError';
  }
}
