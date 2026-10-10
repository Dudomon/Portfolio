import type { ProviderId } from '@aigw/shared';
import type { ZodType } from 'zod';
import { UpstreamError, readErrorMessage } from './upstream-error';

/**
 * POSTs JSON to a provider and returns a 2xx response, translating transport
 * and HTTP failures into {@link UpstreamError}. Abort-driven failures are
 * rethrown untouched so the router can tell client aborts from timeouts by
 * inspecting its own signals.
 */
export async function postJson(
  provider: ProviderId,
  url: string,
  headers: Record<string, string>,
  body: unknown,
  signal: AbortSignal,
): Promise<Response> {
  let response: Response;
  try {
    response = await fetch(url, {
      method: 'POST',
      headers: { 'content-type': 'application/json', ...headers },
      body: JSON.stringify(body),
      signal,
    });
  } catch (error) {
    if (signal.aborted) throw error;
    throw UpstreamError.network(provider, error);
  }

  if (!response.ok) {
    throw UpstreamError.fromStatus(
      provider,
      response.status,
      await readErrorMessage(response),
      response.headers.get('retry-after'),
    );
  }
  return response;
}

/**
 * Validates an upstream payload. A provider returning something we cannot
 * understand is treated as a server fault so the router can fall back.
 */
export function parseUpstream<T>(provider: ProviderId, schema: ZodType<T>, value: unknown): T {
  const result = schema.safeParse(value);
  if (result.success) return result.data;
  throw new UpstreamError(
    provider,
    'server',
    `${provider} returned an unexpected payload: ${result.error.issues[0]?.message ?? 'invalid'}`,
  );
}
