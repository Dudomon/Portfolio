import { createHash } from 'node:crypto';
import { Inject, Injectable, Logger } from '@nestjs/common';
import type { ChatCompletion, ChatCompletionRequest } from '@aigw/shared';
import type { Redis } from 'ioredis';
import { ENV } from '../config/config.module';
import type { AppEnv } from '../config/env';
import { REDIS } from '../infra/redis.module';
import { MetricsService } from '../observability/metrics.service';

/**
 * Exact-match response cache for deterministic requests.
 *
 * Only temperature-0, non-streaming requests are cached: anything sampled
 * would make a cached answer observably different from a fresh one. Entries
 * are scoped per project so tenants never share responses, and keyed by a
 * hash of the canonicalized request so key order and the `user` tag do not
 * cause misses. Redis failures degrade to a miss.
 */
@Injectable()
export class ResponseCacheService {
  private readonly logger = new Logger(ResponseCacheService.name);

  constructor(
    @Inject(REDIS) private readonly redis: Redis,
    @Inject(ENV) private readonly env: AppEnv,
    private readonly metrics: MetricsService,
  ) {}

  isCacheable(request: ChatCompletionRequest): boolean {
    return this.env.RESPONSE_CACHE_TTL_SECONDS > 0 && request.temperature === 0 && !request.stream;
  }

  async get(projectId: string, request: ChatCompletionRequest): Promise<ChatCompletion | null> {
    try {
      const hit = await this.redis.get(cacheKey(projectId, request));
      this.metrics.cacheLookups.inc({ result: hit ? 'hit' : 'miss' });
      return hit ? (JSON.parse(hit) as ChatCompletion) : null;
    } catch (error) {
      this.logger.warn(`Cache read failed: ${(error as Error).message}`);
      this.metrics.cacheLookups.inc({ result: 'error' });
      return null;
    }
  }

  async set(
    projectId: string,
    request: ChatCompletionRequest,
    completion: ChatCompletion,
  ): Promise<void> {
    try {
      await this.redis.set(
        cacheKey(projectId, request),
        JSON.stringify(completion),
        'EX',
        this.env.RESPONSE_CACHE_TTL_SECONDS,
      );
    } catch (error) {
      this.logger.warn(`Cache write failed: ${(error as Error).message}`);
    }
  }
}

export function cacheKey(projectId: string, request: ChatCompletionRequest): string {
  const { stream: _stream, stream_options: _options, user: _user, ...semantic } = request;
  const digest = createHash('sha256').update(canonicalJson(semantic)).digest('hex');
  return `cache:v1:${projectId}:${digest}`;
}

/** JSON with recursively sorted object keys, so equivalent requests hash identically. */
export function canonicalJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
  if (value !== null && typeof value === 'object') {
    const entries = Object.entries(value as Record<string, unknown>)
      .filter(([, v]) => v !== undefined)
      .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
      .map(([k, v]) => `${JSON.stringify(k)}:${canonicalJson(v)}`);
    return `{${entries.join(',')}}`;
  }
  return JSON.stringify(value);
}
