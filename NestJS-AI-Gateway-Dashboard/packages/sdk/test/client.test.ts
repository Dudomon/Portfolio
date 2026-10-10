import { describe, expect, it, vi } from 'vitest';
import {
  APITimeoutError,
  BadRequestError,
  GatewayAdminClient,
  GatewayClient,
  RateLimitError,
  StreamError,
} from '../src';

const encoder = new TextEncoder();

function sseResponse(chunks: string[]): Response {
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      controller.close();
    },
  });
  return new Response(body, { status: 200, headers: { 'content-type': 'text/event-stream' } });
}

function jsonResponse(status: number, body: unknown, headers: Record<string, string> = {}) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json', ...headers },
  });
}

function chunk(content: string) {
  return {
    id: 'chatcmpl-1',
    object: 'chat.completion.chunk',
    created: 1,
    model: 'fast',
    choices: [{ index: 0, delta: { content }, finish_reason: null }],
  };
}

const request = { model: 'fast', messages: [{ role: 'user' as const, content: 'hi' }] };

describe('GatewayClient streaming', () => {
  it('yields chunks parsed from SSE, tolerating arbitrary chunk boundaries', async () => {
    const payload = [
      `data: ${JSON.stringify(chunk('Hel'))}\n\n`,
      `data: ${JSON.stringify(chunk('lo'))}\n\n`,
      'data: [DONE]\n\n',
    ].join('');
    // Split the wire bytes every 7 characters to exercise reassembly.
    const pieces = payload.match(/[\s\S]{1,7}/g) ?? [];
    const fetch = vi.fn().mockResolvedValue(sseResponse(pieces));
    const client = new GatewayClient({ baseUrl: 'http://gw', apiKey: 'gw_test', fetch });

    let text = '';
    for await (const c of client.chat.completions.stream(request)) {
      text += c.choices[0]?.delta.content ?? '';
    }

    expect(text).toBe('Hello');
    const [, init] = fetch.mock.calls[0] as [URL, RequestInit];
    expect(JSON.parse(init.body as string)).toMatchObject({ stream: true, model: 'fast' });
    expect(new Headers(init.headers).get('authorization')).toBe('Bearer gw_test');
  });

  it('surfaces in-band error events as StreamError', async () => {
    const error = {
      error: { message: 'upstream died', type: 'upstream_error', code: 'x', param: null },
    };
    const fetch = vi
      .fn()
      .mockResolvedValue(
        sseResponse([
          `data: ${JSON.stringify(chunk('a'))}\n\n`,
          `data: ${JSON.stringify(error)}\n\n`,
        ]),
      );
    const client = new GatewayClient({ baseUrl: 'http://gw', apiKey: 'k', fetch });

    const received: string[] = [];
    const consume = async () => {
      for await (const c of client.chat.completions.stream(request)) {
        received.push(c.choices[0]?.delta.content ?? '');
      }
    };

    await expect(consume()).rejects.toBeInstanceOf(StreamError);
    expect(received).toEqual(['a']);
  });

  it('treats a stream that ends without [DONE] as truncated', async () => {
    const fetch = vi
      .fn()
      .mockResolvedValue(sseResponse([`data: ${JSON.stringify(chunk('a'))}\n\n`]));
    const client = new GatewayClient({ baseUrl: 'http://gw', apiKey: 'k', fetch });
    const consume = async () => {
      for await (const _ of client.chat.completions.stream(request)) {
        // drain
      }
    };
    await expect(consume()).rejects.toThrow('Stream ended before [DONE]');
  });
});

describe('GatewayClient retries and errors', () => {
  it('retries 429 honoring Retry-After, then succeeds', async () => {
    const completion = {
      id: 'x',
      object: 'chat.completion',
      created: 1,
      model: 'fast',
      choices: [],
    };
    const fetch = vi
      .fn()
      .mockResolvedValueOnce(
        jsonResponse(
          429,
          { error: { message: 'slow down', type: 'rate_limit_error' } },
          { 'retry-after': '0' },
        ),
      )
      .mockResolvedValueOnce(jsonResponse(200, completion));
    const client = new GatewayClient({ baseUrl: 'http://gw', apiKey: 'k', fetch });

    await expect(client.chat.completions.create(request)).resolves.toEqual(completion);
    expect(fetch).toHaveBeenCalledTimes(2);
  });

  it('throws a typed RateLimitError once retries are exhausted', async () => {
    const fetch = vi
      .fn()
      .mockImplementation(() =>
        Promise.resolve(
          jsonResponse(
            429,
            { error: { message: 'slow down', type: 'rate_limit_error', code: 'rate_limited' } },
            { 'retry-after': '0', 'x-request-id': 'req-1' },
          ),
        ),
      );
    const client = new GatewayClient({ baseUrl: 'http://gw', apiKey: 'k', fetch, maxRetries: 1 });

    const error = await client.chat.completions.create(request).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(RateLimitError);
    expect(error).toMatchObject({
      status: 429,
      code: 'rate_limited',
      requestId: 'req-1',
      retryAfter: '0',
    });
    expect(fetch).toHaveBeenCalledTimes(2);
  });

  it('does not retry client errors', async () => {
    const fetch = vi
      .fn()
      .mockResolvedValue(
        jsonResponse(400, { error: { message: 'bad', type: 'invalid_request_error' } }),
      );
    const client = new GatewayClient({ baseUrl: 'http://gw', apiKey: 'k', fetch });

    await expect(client.chat.completions.create(request)).rejects.toBeInstanceOf(BadRequestError);
    expect(fetch).toHaveBeenCalledTimes(1);
  });

  it('raises APITimeoutError when the deadline passes', async () => {
    const fetch = vi.fn((_url: URL | RequestInfo, init?: RequestInit) => {
      return new Promise<Response>((_resolve, reject) => {
        init?.signal?.addEventListener('abort', () => reject(init?.signal?.reason as Error));
      });
    });
    const client = new GatewayClient({
      baseUrl: 'http://gw',
      apiKey: 'k',
      fetch,
      timeoutMs: 10,
      maxRetries: 0,
    });

    await expect(client.models.list()).rejects.toBeInstanceOf(APITimeoutError);
  });

  it('propagates caller aborts without retrying', async () => {
    const controller = new AbortController();
    const fetch = vi.fn((_url: URL | RequestInfo, init?: RequestInit) => {
      return new Promise<Response>((_resolve, reject) => {
        init?.signal?.addEventListener('abort', () => reject(init?.signal?.reason as Error));
      });
    });
    const client = new GatewayClient({ baseUrl: 'http://gw', apiKey: 'k', fetch });

    const pending = client.models.list({ signal: controller.signal });
    controller.abort(new Error('user canceled'));

    await expect(pending).rejects.toThrow('user canceled');
    expect(fetch).toHaveBeenCalledTimes(1);
  });
});

describe('GatewayAdminClient', () => {
  it('serializes filters as query parameters and skips empty values', async () => {
    const fetch = vi.fn().mockResolvedValue(jsonResponse(200, { items: [], nextCursor: null }));
    const admin = new GatewayAdminClient({ baseUrl: 'http://gw/', adminToken: 'adm', fetch });

    await admin.listRequests({ provider: 'anthropic', status: 'error', alias: '', limit: 25 });

    const [url] = fetch.mock.calls[0] as [URL];
    expect(url.toString()).toBe(
      'http://gw/admin/requests?provider=anthropic&status=error&limit=25',
    );
  });

  it('never retries key creation', async () => {
    const fetch = vi
      .fn()
      .mockResolvedValue(jsonResponse(503, { error: { message: 'down', type: 'api_error' } }));
    const admin = new GatewayAdminClient({ baseUrl: 'http://gw', adminToken: 'adm', fetch });

    await expect(admin.createKey({ name: 'ci', project: 'demo' })).rejects.toMatchObject({
      status: 503,
    });
    expect(fetch).toHaveBeenCalledTimes(1);
  });
});
