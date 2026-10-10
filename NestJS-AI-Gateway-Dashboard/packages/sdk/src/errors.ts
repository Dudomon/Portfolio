import type { ErrorBody } from '@aigw/shared';

/** Base class for every error the SDK throws. */
export class GatewayError extends Error {
  override readonly name: string = 'GatewayError';
}

/** The gateway answered with a non-2xx status. */
export class APIError extends GatewayError {
  override readonly name: string = 'APIError';

  constructor(
    readonly status: number,
    message: string,
    readonly type: string | null,
    readonly code: string | null,
    readonly param: string | null,
    readonly requestId: string | null,
  ) {
    super(message);
  }

  static from(status: number, body: unknown, headers: Headers): APIError {
    const requestId = headers.get('x-request-id');
    const parsed = isErrorBody(body) ? body.error : null;
    const message = parsed?.message ?? `Request failed with status ${status}`;
    const args = [
      status,
      message,
      parsed?.type ?? null,
      parsed?.code ?? null,
      parsed?.param ?? null,
      requestId,
    ] as const;

    if (status === 400 || status === 422) return new BadRequestError(...args);
    if (status === 401) return new AuthenticationError(...args);
    if (status === 403) return new PermissionDeniedError(...args);
    if (status === 404) return new NotFoundError(...args);
    if (status === 429) return new RateLimitError(...args, headers.get('retry-after'));
    if (status >= 500) return new InternalServerError(...args);
    return new APIError(...args);
  }
}

export class BadRequestError extends APIError {
  override readonly name = 'BadRequestError';
}
export class AuthenticationError extends APIError {
  override readonly name = 'AuthenticationError';
}
export class PermissionDeniedError extends APIError {
  override readonly name = 'PermissionDeniedError';
}
export class NotFoundError extends APIError {
  override readonly name = 'NotFoundError';
}
export class InternalServerError extends APIError {
  override readonly name = 'InternalServerError';
}

export class RateLimitError extends APIError {
  override readonly name = 'RateLimitError';

  constructor(
    status: number,
    message: string,
    type: string | null,
    code: string | null,
    param: string | null,
    requestId: string | null,
    /** Raw Retry-After header, when the gateway sent one. */
    readonly retryAfter: string | null,
  ) {
    super(status, message, type, code, param, requestId);
  }
}

/** The request never produced a response (DNS, TCP reset, CORS, ...). */
export class APIConnectionError extends GatewayError {
  override readonly name: string = 'APIConnectionError';

  constructor(message: string, options?: { cause?: unknown }) {
    super(message, options);
  }
}

export class APITimeoutError extends APIConnectionError {
  override readonly name = 'APITimeoutError';
}

/** The stream started successfully but the gateway reported an error mid-way. */
export class StreamError extends GatewayError {
  override readonly name = 'StreamError';

  constructor(
    message: string,
    readonly type: string | null,
    readonly code: string | null,
  ) {
    super(message);
  }
}

export function isErrorBody(value: unknown): value is ErrorBody {
  if (typeof value !== 'object' || value === null || !('error' in value)) return false;
  const error = value.error;
  return typeof error === 'object' && error !== null && 'message' in error;
}
