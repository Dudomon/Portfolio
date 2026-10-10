import { fullJitterBackoff, parseRetryAfter, sleep } from '@aigw/shared';
import { APIConnectionError, APIError, APITimeoutError } from './errors';

export interface HttpOptions {
  baseUrl: string;
  /** Sent as `Authorization: Bearer <token>`. */
  token: string;
  /** Retries for 408/409/429/5xx and connection failures. Default 2. */
  maxRetries?: number;
  /** Deadline for receiving response headers (and the body, unless streaming). Default 60s. */
  timeoutMs?: number;
  /** Custom fetch (e.g. for tests or proxies). Defaults to the global fetch. */
  fetch?: typeof fetch;
  defaultHeaders?: Record<string, string>;
}

export interface RequestOptions {
  method: 'GET' | 'POST' | 'PATCH' | 'DELETE';
  path: string;
  body?: unknown;
  query?: Record<string, string | number | undefined>;
  signal?: AbortSignal;
  /**
   * Whether this call may be retried. Non-idempotent admin mutations opt out;
   * chat completions opt in, matching the behavior of the OpenAI SDKs.
   */
  retry?: boolean;
  /** Keep the timeout running only until headers arrive, not for the whole body. */
  stream?: boolean;
}

const RETRY_BACKOFF = { baseMs: 500, maxMs: 8_000 };
const MAX_RETRY_AFTER_MS = 60_000;

/** Fetch wrapper shared by the chat and admin clients: auth, timeouts, retries, typed errors. */
export class HttpClient {
  private readonly baseUrl: string;
  private readonly maxRetries: number;
  private readonly timeoutMs: number;
  private readonly fetchImpl: typeof fetch;
  private readonly headers: Record<string, string>;

  constructor(options: HttpOptions) {
    this.baseUrl = options.baseUrl.replace(/\/+$/, '');
    this.maxRetries = options.maxRetries ?? 2;
    this.timeoutMs = options.timeoutMs ?? 60_000;
    // Bind so that browser fetch is not invoked with a foreign `this`.
    this.fetchImpl = options.fetch ?? globalThis.fetch.bind(globalThis);
    this.headers = {
      ...options.defaultHeaders,
      authorization: `Bearer ${options.token}`,
    };
  }

  async json<T>(options: RequestOptions): Promise<T> {
    const response = await this.request(options);
    if (response.status === 204) return undefined as T;
    return (await response.json()) as T;
  }

  /** Returns a 2xx response, or throws a typed error once retries are exhausted. */
  async request(options: RequestOptions): Promise<Response> {
    const retries = options.retry === false ? 0 : this.maxRetries;
    for (let attempt = 0; ; attempt++) {
      try {
        const response = await this.send(options);
        if (response.ok) return response;
        const error = APIError.from(response.status, await safeJson(response), response.headers);
        if (attempt >= retries || !isRetryableStatus(response.status)) throw error;
        await sleep(this.retryDelay(attempt, response.headers.get('retry-after')), options.signal);
      } catch (error) {
        if (error instanceof APIError) throw error;
        if (options.signal?.aborted) throw options.signal.reason;
        if (attempt >= retries) throw error;
        await sleep(this.retryDelay(attempt, null), options.signal);
      }
    }
  }

  private async send(options: RequestOptions): Promise<Response> {
    const url = new URL(this.baseUrl + options.path);
    for (const [key, value] of Object.entries(options.query ?? {})) {
      if (value !== undefined && value !== '') url.searchParams.set(key, String(value));
    }

    // A manual timer (rather than AbortSignal.timeout) so streaming requests can
    // clear it once headers arrive; long generations must not be cut off.
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(new TimeoutMarker()), this.timeoutMs);
    const signal = options.signal
      ? AbortSignal.any([options.signal, controller.signal])
      : controller.signal;

    try {
      const response = await this.fetchImpl(url, {
        method: options.method,
        headers: {
          ...this.headers,
          ...(options.body !== undefined ? { 'content-type': 'application/json' } : {}),
        },
        body: options.body !== undefined ? JSON.stringify(options.body) : null,
        signal,
      });
      return options.stream ? response : await bufferBody(response);
    } catch (error) {
      if (controller.signal.reason instanceof TimeoutMarker) {
        throw new APITimeoutError(`Request timed out after ${this.timeoutMs}ms`, { cause: error });
      }
      if (options.signal?.aborted) throw options.signal.reason;
      throw new APIConnectionError('Connection to the gateway failed', { cause: error });
    } finally {
      clearTimeout(timer);
    }
  }

  private retryDelay(attempt: number, retryAfter: string | null): number {
    const serverHint = parseRetryAfter(retryAfter);
    if (serverHint !== undefined && serverHint <= MAX_RETRY_AFTER_MS) return serverHint;
    return fullJitterBackoff(attempt, RETRY_BACKOFF);
  }
}

class TimeoutMarker extends Error {}

function isRetryableStatus(status: number): boolean {
  return status === 408 || status === 409 || status === 429 || status >= 500;
}

/**
 * Reads the body while the timeout is still armed, so a server that sends
 * headers and then stalls is still caught, and returns an equivalent Response
 * backed by the buffered text.
 */
async function bufferBody(response: Response): Promise<Response> {
  const text = await response.text();
  return new Response(text === '' ? null : text, {
    status: response.status,
    statusText: response.statusText,
    headers: response.headers,
  });
}

async function safeJson(response: Response): Promise<unknown> {
  try {
    return await response.json();
  } catch {
    return undefined;
  }
}
