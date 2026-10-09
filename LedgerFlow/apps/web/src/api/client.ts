import type { z } from 'zod';

import { NetworkError, ResponseValidationError, toApiError } from './errors';

export interface ApiClientConfig {
  /** Absolute URL or path (e.g. `/api/v1`), resolved against the page origin. */
  baseUrl: string;
  getAccessToken: () => string | null;
  /** Called on a 401 from an authenticated request, before the error is thrown. */
  onUnauthorized?: () => void;
}

export type QueryParams = Record<string, string | number | boolean | null | undefined>;

export interface RequestOptions {
  method?: 'GET' | 'POST' | 'PATCH' | 'PUT' | 'DELETE';
  query?: QueryParams;
  json?: unknown;
  /** Sent as `If-Match` for optimistic concurrency. */
  ifMatch?: string;
  signal?: AbortSignal;
  /** Omit the bearer token and skip 401 handling (e.g. the login call itself). */
  anonymous?: boolean;
}

export interface ApiResponse<T> {
  data: T;
  status: number;
  etag: string | null;
  requestId: string | null;
}

export interface UploadOptions {
  /** Upload progress in the range 0..1. */
  onProgress?: (fraction: number) => void;
  signal?: AbortSignal;
}

const JSON_CONTENT_TYPE = /\bjson\b/i;

function newRequestId(): string {
  return crypto.randomUUID();
}

function parseBody(text: string, contentType: string | null): unknown {
  if (text === '' || !contentType || !JSON_CONTENT_TYPE.test(contentType)) return text;
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return text;
  }
}

export function createApiClient(config: ApiClientConfig) {
  function buildUrl(path: string, query?: QueryParams): URL {
    const base = config.baseUrl.replace(/\/+$/, '');
    const url = new URL(`${base}${path}`, window.location.origin);
    for (const [key, value] of Object.entries(query ?? {})) {
      if (value !== undefined && value !== null && value !== '') {
        url.searchParams.set(key, String(value));
      }
    }
    return url;
  }

  function buildHeaders(options: RequestOptions, requestId: string): Record<string, string> {
    const headers: Record<string, string> = {
      Accept: 'application/json, application/problem+json',
      'X-Request-ID': requestId,
    };
    if (options.json !== undefined) headers['Content-Type'] = 'application/json';
    if (options.ifMatch) headers['If-Match'] = options.ifMatch;
    const token = options.anonymous ? null : config.getAccessToken();
    if (token) headers.Authorization = `Bearer ${token}`;
    return headers;
  }

  function failure(
    options: RequestOptions,
    status: number,
    statusText: string,
    body: unknown,
    requestId: string | null,
  ) {
    if (status === 401 && !options.anonymous) config.onUnauthorized?.();
    return toApiError(status, statusText, body, requestId);
  }

  function validate<S extends z.ZodTypeAny>(
    schema: S,
    body: unknown,
    path: string,
    requestId: string | null,
  ): z.output<S> {
    const result = schema.safeParse(body);
    if (!result.success) throw new ResponseValidationError(path, result.error.issues, requestId);
    return result.data as z.output<S>;
  }

  async function send(path: string, options: RequestOptions): Promise<Response> {
    const requestId = newRequestId();
    let response: Response;
    try {
      response = await fetch(buildUrl(path, options.query), {
        method: options.method ?? 'GET',
        headers: buildHeaders(options, requestId),
        body: options.json === undefined ? undefined : JSON.stringify(options.json),
        signal: options.signal,
      });
    } catch (error) {
      // Aborts are expected (TanStack Query cancels stale requests); let them through untouched.
      if (error instanceof DOMException && error.name === 'AbortError') throw error;
      throw new NetworkError({ cause: error });
    }
    if (!response.ok) {
      const contentType = response.headers.get('Content-Type');
      const body = parseBody(await response.text(), contentType);
      throw failure(
        options,
        response.status,
        response.statusText,
        body,
        response.headers.get('X-Request-ID') ?? requestId,
      );
    }
    return response;
  }

  async function request<S extends z.ZodTypeAny>(
    path: string,
    schema: S,
    options: RequestOptions = {},
  ): Promise<ApiResponse<z.output<S>>> {
    const response = await send(path, options);
    const requestId = response.headers.get('X-Request-ID');
    const body = parseBody(await response.text(), response.headers.get('Content-Type'));
    return {
      data: validate(schema, body, path, requestId),
      status: response.status,
      etag: response.headers.get('ETag'),
      requestId,
    };
  }

  /** For endpoints that answer 202/204 without a meaningful body. */
  async function requestVoid(path: string, options: RequestOptions = {}): Promise<void> {
    await send(path, options);
  }

  async function requestBlob(path: string, options: RequestOptions = {}): Promise<Blob> {
    const response = await send(path, options);
    return response.blob();
  }

  /**
   * Multipart upload over XMLHttpRequest: fetch still has no portable way to
   * report request-body progress.
   */
  function upload<S extends z.ZodTypeAny>(
    path: string,
    schema: S,
    body: FormData,
    { onProgress, signal }: UploadOptions = {},
  ): Promise<ApiResponse<z.output<S>>> {
    const options: RequestOptions = { method: 'POST' };
    const requestId = newRequestId();

    return new Promise((resolve, reject) => {
      if (signal?.aborted) {
        reject(new DOMException('Upload aborted', 'AbortError'));
        return;
      }
      const xhr = new XMLHttpRequest();
      xhr.open('POST', buildUrl(path).toString());
      for (const [name, value] of Object.entries(buildHeaders(options, requestId))) {
        xhr.setRequestHeader(name, value);
      }

      xhr.upload.addEventListener('progress', (event) => {
        if (event.lengthComputable) onProgress?.(event.loaded / event.total);
      });
      xhr.addEventListener('load', () => {
        const responseRequestId = xhr.getResponseHeader('X-Request-ID') ?? requestId;
        const parsed = parseBody(xhr.responseText, xhr.getResponseHeader('Content-Type'));
        if (xhr.status < 200 || xhr.status >= 300) {
          reject(failure(options, xhr.status, xhr.statusText, parsed, responseRequestId));
          return;
        }
        try {
          resolve({
            data: validate(schema, parsed, path, responseRequestId),
            status: xhr.status,
            etag: xhr.getResponseHeader('ETag'),
            requestId: responseRequestId,
          });
        } catch (error) {
          reject(error instanceof Error ? error : new Error(String(error)));
        }
      });
      xhr.addEventListener('error', () => reject(new NetworkError()));
      xhr.addEventListener('abort', () => reject(new DOMException('Upload aborted', 'AbortError')));
      signal?.addEventListener('abort', () => xhr.abort(), { once: true });

      xhr.send(body);
    });
  }

  return { request, requestVoid, requestBlob, upload };
}

export type ApiClient = ReturnType<typeof createApiClient>;
