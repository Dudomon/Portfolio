import type { ZodIssue } from 'zod';

import { ProblemDetailsSchema, type ProblemDetails } from './schemas';

/** A non-2xx response, carrying the RFC 9457 problem document. */
export class ApiError extends Error {
  override readonly name: string = 'ApiError';
  readonly status: number;
  readonly problem: ProblemDetails;
  readonly requestId: string | null;

  constructor(problem: ProblemDetails, requestId: string | null) {
    super(problem.detail ?? problem.title);
    this.status = problem.status;
    this.problem = problem;
    this.requestId = problem.request_id ?? requestId;
  }
}

/**
 * 412: the resource changed since it was read (If-Match mismatch). Callers
 * need to offer a reload rather than a generic retry, hence a distinct type.
 */
export class PreconditionFailedError extends ApiError {
  override readonly name = 'PreconditionFailedError';
}

export class UnauthorizedError extends ApiError {
  override readonly name = 'UnauthorizedError';
}

/** The request never produced an HTTP response (offline, DNS, CORS…). */
export class NetworkError extends Error {
  override readonly name = 'NetworkError';
  constructor(options?: ErrorOptions) {
    super('Could not reach the server. Check your connection and try again.', options);
  }
}

/** A 2xx response whose body does not match the contract. */
export class ResponseValidationError extends Error {
  override readonly name = 'ResponseValidationError';
  readonly issues: readonly ZodIssue[];
  readonly requestId: string | null;

  constructor(path: string, issues: readonly ZodIssue[], requestId: string | null) {
    super(`Unexpected response shape from ${path}`);
    this.issues = issues;
    this.requestId = requestId;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/**
 * Builds a typed error from a failed response. Bodies that are not valid
 * problem+json (proxies, gateways, framework defaults) are normalised into
 * one so callers only ever deal with a single shape.
 */
export function toApiError(
  status: number,
  statusText: string,
  body: unknown,
  requestId: string | null,
): ApiError {
  const parsed = ProblemDetailsSchema.safeParse(body);
  const problem: ProblemDetails = parsed.success
    ? { ...parsed.data, status }
    : {
        type: 'about:blank',
        title: statusText || `Request failed with status ${status}`,
        status,
        detail: isRecord(body) && typeof body.detail === 'string' ? body.detail : null,
      };

  switch (status) {
    case 401:
      return new UnauthorizedError(problem, requestId);
    case 412:
      return new PreconditionFailedError(problem, requestId);
    default:
      return new ApiError(problem, requestId);
  }
}

/** A human-readable message for any error thrown by the API layer. */
export function describeError(error: unknown): string {
  if (error instanceof PreconditionFailedError) {
    return 'This document was changed by someone else. Reload to see the latest version.';
  }
  if (error instanceof ApiError || error instanceof NetworkError) {
    return error.message;
  }
  if (error instanceof ResponseValidationError) {
    return 'The server returned data this version of the app does not understand.';
  }
  return 'Something went wrong.';
}
