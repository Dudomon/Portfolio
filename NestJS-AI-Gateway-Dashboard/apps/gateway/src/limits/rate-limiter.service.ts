import { Inject, Injectable, Logger } from '@nestjs/common';
import type { Redis } from 'ioredis';
import { REDIS } from '../infra/redis.module';

/**
 * Token bucket evaluated atomically inside Redis.
 *
 * Doing the read-refill-decrement in one script removes the race two gateway
 * replicas would otherwise have between reading and writing the bucket. The
 * clock comes from Redis `TIME`, so replicas with skewed clocks still agree.
 * The bucket expires once it would be full again, so idle keys cost nothing.
 *
 * Returns {allowed (0|1), remaining tokens (floored), retry-after ms}.
 */
export const TOKEN_BUCKET_SCRIPT = `
local capacity = tonumber(ARGV[1])
local refill_per_ms = tonumber(ARGV[2])
local cost = tonumber(ARGV[3])
local time = redis.call('TIME')
local now = tonumber(time[1]) * 1000 + math.floor(tonumber(time[2]) / 1000)

local state = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens = tonumber(state[1])
local ts = tonumber(state[2])
if tokens == nil or ts == nil then
  tokens = capacity
  ts = now
end

tokens = math.min(capacity, tokens + math.max(0, now - ts) * refill_per_ms)
local allowed = 0
local retry_after = 0
if tokens >= cost then
  tokens = tokens - cost
  allowed = 1
else
  retry_after = math.ceil((cost - tokens) / refill_per_ms)
end

redis.call('HSET', KEYS[1], 'tokens', tostring(tokens), 'ts', tostring(now))
redis.call('PEXPIRE', KEYS[1], math.ceil(capacity / refill_per_ms) + 1000)
return { allowed, math.floor(tokens), retry_after }
`;

export interface RateLimitDecision {
  allowed: boolean;
  limit: number;
  remaining: number;
  retryAfterMs: number;
}

type RedisWithBucket = Redis & {
  tokenBucket(
    key: string,
    capacity: number,
    refillPerMs: number,
    cost: number,
  ): Promise<[number, number, number]>;
};

@Injectable()
export class RateLimiterService {
  private readonly logger = new Logger(RateLimiterService.name);
  private readonly redis: RedisWithBucket;

  constructor(@Inject(REDIS) redis: Redis) {
    // defineCommand uses EVALSHA and transparently reloads the script after
    // a Redis restart (NOSCRIPT), so the script body is not resent per call.
    redis.defineCommand('tokenBucket', { numberOfKeys: 1, lua: TOKEN_BUCKET_SCRIPT });
    this.redis = redis as RedisWithBucket;
  }

  /**
   * Consumes one request token for the key. Fails open: if Redis is
   * unreachable the request is allowed, trading strict enforcement for
   * availability. Budgets still bound spend during such an outage.
   */
  async consume(keyId: string, perMinute: number): Promise<RateLimitDecision> {
    try {
      const [allowed, remaining, retryAfterMs] = await this.redis.tokenBucket(
        `ratelimit:${keyId}`,
        perMinute,
        perMinute / 60_000,
        1,
      );
      return { allowed: allowed === 1, limit: perMinute, remaining, retryAfterMs };
    } catch (error) {
      this.logger.warn(`Rate limiter unavailable, failing open: ${(error as Error).message}`);
      return { allowed: true, limit: perMinute, remaining: perMinute, retryAfterMs: 0 };
    }
  }
}
