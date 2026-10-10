import type { ExecutionContext } from '@nestjs/common';
import type { Redis } from 'ioredis';
import { describe, expect, it, vi } from 'vitest';
import type { AuthenticatedKey } from '../src/auth/api-key.service';
import { ApiError } from '../src/common/api-error';
import { RateLimitGuard } from '../src/limits/limits.guards';
import { RateLimiterService, TOKEN_BUCKET_SCRIPT } from '../src/limits/rate-limiter.service';
import { MetricsService } from '../src/observability/metrics.service';

/** Minimal ioredis double: defineCommand attaches the scripted command implementation. */
function mockRedis(tokenBucket: (...args: unknown[]) => Promise<unknown>) {
  const redis = {
    defineCommand: vi.fn((name: string) => {
      (redis as Record<string, unknown>)[name] = vi.fn(tokenBucket);
    }),
  };
  return redis as unknown as Redis & {
    defineCommand: ReturnType<typeof vi.fn>;
    tokenBucket: ReturnType<typeof vi.fn>;
  };
}

const key: AuthenticatedKey = {
  id: 'key_1',
  name: 'test',
  prefix: 'abc',
  projectId: 'proj_1',
  rateLimitPerMinute: 120,
  monthlyBudgetUsd: null,
};

function httpContext(apiKey: AuthenticatedKey) {
  const headers = new Map<string, string>();
  const context = {
    switchToHttp: () => ({
      getRequest: () => ({ apiKey }),
      getResponse: () => ({ setHeader: (name: string, value: string) => headers.set(name, value) }),
    }),
  } as unknown as ExecutionContext;
  return { context, headers };
}

describe('RateLimiterService', () => {
  it('registers the Lua script once and passes capacity and per-ms refill rate', async () => {
    const redis = mockRedis(() => Promise.resolve([1, 119, 0]));
    const limiter = new RateLimiterService(redis);

    const decision = await limiter.consume('key_1', 120);

    expect(redis.defineCommand).toHaveBeenCalledWith('tokenBucket', {
      numberOfKeys: 1,
      lua: TOKEN_BUCKET_SCRIPT,
    });
    expect(redis.tokenBucket).toHaveBeenCalledWith('ratelimit:key_1', 120, 120 / 60_000, 1);
    expect(decision).toEqual({ allowed: true, limit: 120, remaining: 119, retryAfterMs: 0 });
  });

  it('reports denial with the retry delay computed by the script', async () => {
    const limiter = new RateLimiterService(mockRedis(() => Promise.resolve([0, 0, 437])));
    await expect(limiter.consume('key_1', 120)).resolves.toMatchObject({
      allowed: false,
      retryAfterMs: 437,
    });
  });

  it('fails open when Redis is unavailable', async () => {
    const limiter = new RateLimiterService(
      mockRedis(() => Promise.reject(new Error("Stream isn't writeable"))),
    );
    await expect(limiter.consume('key_1', 60)).resolves.toMatchObject({ allowed: true });
  });

  it('keeps the read-modify-write inside one atomic script using server time', () => {
    expect(TOKEN_BUCKET_SCRIPT).toContain("redis.call('TIME')");
    expect(TOKEN_BUCKET_SCRIPT).toContain("redis.call('HMGET'");
    expect(TOKEN_BUCKET_SCRIPT).toContain("redis.call('PEXPIRE'");
  });
});

describe('RateLimitGuard', () => {
  it('sets rate limit headers on allowed requests', async () => {
    const guard = new RateLimitGuard(
      new RateLimiterService(mockRedis(() => Promise.resolve([1, 41, 0]))),
      new MetricsService(),
    );
    const { context, headers } = httpContext(key);

    await expect(guard.canActivate(context)).resolves.toBe(true);
    expect(headers.get('x-ratelimit-remaining-requests')).toBe('41');
  });

  it('rejects with 429 and a whole-second Retry-After', async () => {
    const guard = new RateLimitGuard(
      new RateLimiterService(mockRedis(() => Promise.resolve([0, 0, 1_200]))),
      new MetricsService(),
    );

    const error = await guard.canActivate(httpContext(key).context).catch((e: unknown) => e);

    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({
      status: 429,
      code: 'rate_limit_exceeded',
      headers: { 'retry-after': '2' },
    });
  });
});
