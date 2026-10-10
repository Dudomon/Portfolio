import {
  ArgumentsHost,
  Catch,
  HttpException,
  HttpStatus,
  Logger,
  type ExceptionFilter,
} from '@nestjs/common';
import { errorBody, type ErrorBody, type ErrorType } from '@aigw/shared';
import type { Response } from 'express';
import { ApiError } from './api-error';

/**
 * Renders every failure as an OpenAI-style error object so SDKs written for
 * OpenAI parse gateway errors natively. Unknown errors are logged with their
 * stack and returned as an opaque 500 to avoid leaking internals.
 */
@Catch()
export class ApiExceptionFilter implements ExceptionFilter {
  private readonly logger = new Logger(ApiExceptionFilter.name);

  catch(exception: unknown, host: ArgumentsHost): void {
    const response = host.switchToHttp().getResponse<Response>();
    // Streaming handlers write errors in-band; once headers are out, the
    // status line can no longer change.
    if (response.headersSent) {
      if (!response.writableEnded) response.end();
      return;
    }

    const { status, body, headers } = this.toResponse(exception);
    for (const [name, value] of Object.entries(headers)) response.setHeader(name, value);
    response.status(status).json(body);
  }

  private toResponse(exception: unknown): {
    status: number;
    body: ErrorBody;
    headers: Readonly<Record<string, string>>;
  } {
    if (exception instanceof ApiError) {
      return {
        status: exception.status,
        body: errorBody(exception.type, exception.message, {
          code: exception.code,
          param: exception.param,
        }),
        headers: exception.headers,
      };
    }

    if (exception instanceof HttpException) {
      const status = exception.getStatus();
      return {
        status,
        body: errorBody(typeForStatus(status), exception.message, { code: codeForStatus(status) }),
        headers: {},
      };
    }

    this.logger.error(exception instanceof Error ? exception.stack : String(exception));
    return {
      status: HttpStatus.INTERNAL_SERVER_ERROR,
      body: errorBody('api_error', 'Internal server error', { code: 'internal_error' }),
      headers: {},
    };
  }
}

function typeForStatus(status: number): ErrorType {
  if (status === 401) return 'authentication_error';
  if (status === 403) return 'permission_error';
  if (status === 404) return 'not_found_error';
  if (status === 429) return 'rate_limit_error';
  if (status >= 400 && status < 500) return 'invalid_request_error';
  return 'api_error';
}

function codeForStatus(status: number): string {
  return status === 404 ? 'not_found' : `http_${status}`;
}
