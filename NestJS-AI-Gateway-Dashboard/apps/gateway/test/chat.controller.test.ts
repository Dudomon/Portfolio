import { createServer, type Server } from 'node:http';
import type { AddressInfo } from 'node:net';
import express, {
  type NextFunction,
  type Request as ExpressRequest,
  type Response as ExpressResponse,
} from 'express';
import { readSseStream, type ChatCompletionChunk } from '@aigw/shared';
import type { Redis } from 'ioredis';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { AuthenticatedKey } from '../src/auth/api-key.service';
import { ResponseCacheService } from '../src/cache/response-cache.service';
import { ApiError } from '../src/common/api-error';
import type { AppEnv } from '../src/config/env';
import type { BudgetService } from '../src/limits/budget.service';
import { ChatController } from '../src/chat/chat.controller';
import { ChatService } from '../src/chat/chat.service';
import type { UsageEntry, UsageRecorder } from '../src/usage/usage-recorder.service';
import { FakeProvider, createRouter, testConfig, type Step } from './fakes';

const key: AuthenticatedKey = {
  id: 'key_1',
  name: 'test',
  prefix: 'abc',
  projectId: 'proj_1',
  rateLimitPerMinute: 60,
  monthlyBudgetUsd: null,
};

let server: Server | undefined;

afterEach(() => {
  server?.close();
  server = undefined;
});

/**
 * Mounts the controller handler on a real HTTP server (without the Nest
 * bootstrap) so SSE framing, headers and disconnect handling are exercised
 * over an actual socket.
 */
async function startGateway(primary: Step[], fallback: Step[] = []) {
  const config = testConfig();
  const { router, metrics } = createRouter(
    [new FakeProvider('openai', primary), new FakeProvider('anthropic', fallback)],
    config,
  );
  const recorded: UsageEntry[] = [];
  const usage = { record: (entry: UsageEntry) => recorded.push(entry) } as unknown as UsageRecorder;
  const budgets = { addSpend: vi.fn(() => Promise.resolve()) } as unknown as BudgetService;
  const cache = new ResponseCacheService(
    {} as Redis,
    { RESPONSE_CACHE_TTL_SECONDS: 0 } as AppEnv,
    metrics,
  );
  const controller = new ChatController(
    new ChatService(router, cache, usage, budgets, metrics, config),
    router,
    config,
  );

  const app = express();
  app.use(express.json());
  app.post(
    '/v1/chat/completions',
    (req: ExpressRequest, res: ExpressResponse, next: NextFunction) => {
      (req as ExpressRequest & { id: string }).id = 'req-e2e';
      controller.createCompletion(req.body as never, key, req, res).catch(next);
    },
  );
  app.use((error: unknown, _req: ExpressRequest, res: ExpressResponse, _next: NextFunction) => {
    const status = error instanceof ApiError ? error.status : 500;
    res.status(status).json({ error: { message: (error as Error).message } });
  });

  server = createServer(app);
  await new Promise<void>((resolve) => server?.listen(0, resolve));
  const { port } = server.address() as AddressInfo;
  return { url: `http://127.0.0.1:${port}/v1/chat/completions`, recorded };
}

async function post(url: string, body: unknown) {
  return fetch(url, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  });
}

async function readEvents(response: Response): Promise<string[]> {
  if (!response.body) throw new Error('no body');
  const body = response.body;
  const events: string[] = [];
  for await (const event of readSseStream(body)) events.push(event.data);
  return events;
}

const messages = [{ role: 'user', content: 'hi' }];

describe('POST /v1/chat/completions', () => {
  it('returns JSON with routing headers for non-streaming requests', async () => {
    const { url, recorded } = await startGateway([500, 500], ['ok']);

    const response = await post(url, { model: 'fast', messages });

    expect(response.status).toBe(200);
    expect(response.headers.get('x-gateway-provider')).toBe('anthropic');
    expect(response.headers.get('x-gateway-attempts')).toBe('3');
    const completion = (await response.json()) as { usage: { total_tokens: number } };
    expect(completion.usage.total_tokens).toBe(15);
    // anthropic/fallback-model: 10 * 3/1M + 5 * 4/1M
    expect(recorded[0]).toMatchObject({ status: 'success', costUsd: 0.00005, attempts: 3 });
  });

  it('streams SSE chunks terminated by [DONE], hiding usage unless requested', async () => {
    const { url, recorded } = await startGateway(['ok']);

    const response = await post(url, { model: 'fast', messages, stream: true });
    const events = await readEvents(response);

    expect(response.headers.get('content-type')).toContain('text/event-stream');
    expect(events.at(-1)).toBe('[DONE]');
    const chunks = events.slice(0, -1).map((data) => JSON.parse(data) as ChatCompletionChunk);
    expect(chunks.some((c) => c.usage)).toBe(false);
    expect(recorded[0]).toMatchObject({ streamed: true, promptTokens: 3, completionTokens: 2 });
  });

  it('forwards the usage chunk when stream_options.include_usage is set', async () => {
    const { url } = await startGateway(['ok']);

    const response = await post(url, {
      model: 'fast',
      messages,
      stream: true,
      stream_options: { include_usage: true },
    });
    const events = await readEvents(response);

    const usageChunk = JSON.parse(events.at(-2) ?? '{}') as ChatCompletionChunk;
    expect(usageChunk.usage?.total_tokens).toBe(5);
  });

  it('reports a mid-stream failure in-band after the 200 has been sent', async () => {
    const { url, recorded } = await startGateway(['midstream']);

    const response = await post(url, { model: 'fast', messages, stream: true });
    const events = await readEvents(response);

    expect(response.status).toBe(200);
    expect(events).toHaveLength(2);
    expect(JSON.parse(events[1] ?? '{}')).toMatchObject({
      error: { type: 'upstream_error', code: 'stream_interrupted' },
    });
    expect(recorded[0]).toMatchObject({ status: 'error', streamed: true });
  });

  it('returns a normal error response when the stream fails before it starts', async () => {
    const { url } = await startGateway([400]);

    const response = await post(url, { model: 'fast', messages, stream: true });

    expect(response.status).toBe(400);
    expect(response.headers.get('content-type')).toContain('application/json');
  });
});
