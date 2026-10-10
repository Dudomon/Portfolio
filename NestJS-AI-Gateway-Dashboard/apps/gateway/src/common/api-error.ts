import type { ErrorType } from '@aigw/shared';

/**
 * An error that maps directly onto an OpenAI-style error response. Thrown by
 * guards, services and the router; rendered by {@link ApiExceptionFilter}.
 */
export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly type: ErrorType,
    message: string,
    readonly code: string | null = null,
    readonly param: string | null = null,
    readonly headers: Readonly<Record<string, string>> = {},
  ) {
    super(message);
    this.name = 'ApiError';
  }

  static invalidRequest(message: string, param: string | null = null): ApiError {
    return new ApiError(400, 'invalid_request_error', message, 'invalid_request', param);
  }

  static unauthorized(message = 'Invalid or missing API key'): ApiError {
    return new ApiError(401, 'authentication_error', message, 'invalid_api_key');
  }

  static notFound(message: string, code = 'not_found'): ApiError {
    return new ApiError(404, 'not_found_error', message, code);
  }
}
