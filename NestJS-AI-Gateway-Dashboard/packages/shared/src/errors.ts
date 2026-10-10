import { z } from 'zod';

/**
 * Error envelope returned by every gateway endpoint. It mirrors OpenAI's shape
 * so existing client libraries surface gateway errors without special casing.
 */
export const errorTypeSchema = z.enum([
  'invalid_request_error',
  'authentication_error',
  'permission_error',
  'not_found_error',
  'rate_limit_error',
  'insufficient_quota',
  'api_error',
  'upstream_error',
  'timeout_error',
]);
export type ErrorType = z.infer<typeof errorTypeSchema>;

export const errorBodySchema = z.object({
  error: z.object({
    message: z.string(),
    type: z.string(),
    param: z.string().nullable(),
    code: z.string().nullable(),
  }),
});
export type ErrorBody = z.infer<typeof errorBodySchema>;

export function errorBody(
  type: ErrorType,
  message: string,
  options: { code?: string | null; param?: string | null } = {},
): ErrorBody {
  return {
    error: { message, type, param: options.param ?? null, code: options.code ?? null },
  };
}
