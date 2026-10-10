import { describe, expect, it } from 'vitest';
import { ApiError } from '../src/common/api-error';
import {
  ClientAbortedError,
  RouteExhaustedError,
  UpstreamRejectedError,
} from '../src/routing/routing.errors';
import { FakeProvider, createRouter, request, testConfig } from './fakes';

const context = () => ({ signal: new AbortController().signal, requestId: 'req-test' });

describe('RouterService', () => {
  it('uses the first target when it succeeds', async () => {
    const primary = new FakeProvider('openai', ['ok']);
    const fallback = new FakeProvider('anthropic', ['ok']);
    const { router } = createRouter([primary, fallback]);

    const result = await router.complete(request, context());

    expect(result.target).toEqual({ provider: 'openai', model: 'primary-model' });
    expect(result.attempts).toBe(1);
    expect(primary.calls[0]?.model).toBe('primary-model');
    expect(fallback.calls).toHaveLength(0);
  });

  it('retries a 5xx on the same target before falling back', async () => {
    const primary = new FakeProvider('openai', [503, 'ok']);
    const { router } = createRouter([primary, new FakeProvider('anthropic', [])]);

    const result = await router.complete(request, context());

    expect(result.target.provider).toBe('openai');
    expect(result.attempts).toBe(2);
  });

  it('falls back after exhausting same-target retries', async () => {
    const primary = new FakeProvider('openai', [500, 502]);
    const fallback = new FakeProvider('anthropic', ['ok']);
    const { router, metrics } = createRouter([primary, fallback]);

    const result = await router.complete(request, context());

    expect(result.target).toEqual({ provider: 'anthropic', model: 'fallback-model' });
    expect(result.attempts).toBe(3);
    expect(fallback.calls[0]?.model).toBe('fallback-model');
    expect((await metrics.fallbacks.get()).values[0]?.value).toBe(1);
  });

  it('moves straight to the next target on 429 without retrying', async () => {
    const primary = new FakeProvider('openai', [429]);
    const fallback = new FakeProvider('anthropic', ['ok']);
    const { router } = createRouter([primary, fallback]);

    const result = await router.complete(request, context());

    expect(primary.calls).toHaveLength(1);
    expect(result.target.provider).toBe('anthropic');
  });

  it('does not fall back on client errors', async () => {
    const fallback = new FakeProvider('anthropic', ['ok']);
    const { router, breakers } = createRouter([new FakeProvider('openai', [400]), fallback]);

    await expect(router.complete(request, context())).rejects.toBeInstanceOf(UpstreamRejectedError);
    expect(fallback.calls).toHaveLength(0);
    expect(breakers.state('openai')).toBe('closed');
  });

  it('treats a slow provider as a timeout and falls back', async () => {
    const { router } = createRouter(
      [new FakeProvider('openai', ['hang']), new FakeProvider('anthropic', ['ok'])],
      testConfig({ attemptTimeoutMs: 20, maxRetriesPerTarget: 0 }),
    );

    const result = await router.complete(request, context());

    expect(result.target.provider).toBe('anthropic');
  });

  it('maps exhaustion by the last error kind', async () => {
    const { router } = createRouter([
      new FakeProvider('openai', [500, 500]),
      new FakeProvider('anthropic', [429]),
    ]);

    const error = await router.complete(request, context()).catch((e: unknown) => e);

    expect(error).toBeInstanceOf(RouteExhaustedError);
    expect(error).toMatchObject({ status: 429, code: 'upstream_rate_limited', attempts: 3 });
    expect((error as RouteExhaustedError).target?.provider).toBe('anthropic');
  });

  it('returns 503 when no target has a configured provider', async () => {
    const { router } = createRouter([]);
    await expect(router.complete(request, context())).rejects.toMatchObject({
      status: 503,
      code: 'no_available_provider',
    });
  });

  it('rejects unknown aliases with model_not_found', () => {
    const { router } = createRouter([]);
    expect(() => router.resolve('nope')).toThrow(ApiError);
  });

  it('stops immediately when the client aborts, without blaming the provider', async () => {
    const controller = new AbortController();
    const fallback = new FakeProvider('anthropic', ['ok']);
    const { router, breakers } = createRouter([new FakeProvider('openai', ['hang']), fallback]);

    const pending = router.complete(request, { signal: controller.signal, requestId: 'r' });
    controller.abort();

    await expect(pending).rejects.toBeInstanceOf(ClientAbortedError);
    expect(fallback.calls).toHaveLength(0);
    expect(breakers.state('openai')).toBe('closed');
  });

  it('skips a provider whose circuit is open', async () => {
    const primary = new FakeProvider('openai', [500, 500, 500, 500]);
    const fallback = new FakeProvider('anthropic', []);
    const { router, breakers } = createRouter(
      [primary, fallback],
      testConfig({
        maxRetriesPerTarget: 0,
        circuitBreaker: { failureThreshold: 2, cooldownMs: 60_000 },
      }),
    );

    await router.complete(request, context());
    await router.complete(request, context());
    expect(breakers.state('openai')).toBe('open');

    const callsBefore = primary.calls.length;
    const result = await router.complete(request, context());
    expect(primary.calls).toHaveLength(callsBefore);
    expect(result.attempts).toBe(1);
    expect(result.target.provider).toBe('anthropic');
  });

  describe('streaming', () => {
    it('falls back when a stream fails before its first chunk', async () => {
      const { router } = createRouter([
        new FakeProvider('openai', [503, 503]),
        new FakeProvider('anthropic', ['ok']),
      ]);

      const routed = await router.stream({ ...request, stream: true }, context());
      const chunks = [];
      for await (const chunk of routed.value.chunks) chunks.push(chunk);

      expect(routed.target.provider).toBe('anthropic');
      expect(chunks[0]?.choices[0]?.delta.content).toBe('from anthropic');
      expect(chunks.at(-1)?.usage?.total_tokens).toBe(5);
    });
  });
});
