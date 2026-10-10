import type { PipeTransform } from '@nestjs/common';
import type { ZodType } from 'zod';
import { ApiError } from './api-error';

/**
 * Validates a request body or query against a shared zod schema. The schemas
 * live in @aigw/shared so the SDK and dashboard compile against exactly the
 * contract the gateway enforces.
 *
 * Only the first issue is reported, mirroring OpenAI's single `param` field.
 */
export class ZodValidationPipe<T> implements PipeTransform<unknown, T> {
  constructor(private readonly schema: ZodType<T>) {}

  transform(value: unknown): T {
    const result = this.schema.safeParse(value);
    if (result.success) return result.data;

    const issue = result.error.issues[0];
    const param = issue && issue.path.length > 0 ? issue.path.join('.') : null;
    const message = issue ? `${param ? `${param}: ` : ''}${issue.message}` : 'Invalid request';
    throw ApiError.invalidRequest(message, param);
  }
}
