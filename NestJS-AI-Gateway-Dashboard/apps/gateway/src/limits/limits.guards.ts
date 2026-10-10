import { Injectable, type CanActivate, type ExecutionContext } from '@nestjs/common';
import type { Request, Response } from 'express';
import { ApiError } from '../common/api-error';
import { MetricsService } from '../observability/metrics.service';
import { BudgetService } from './budget.service';
import { RateLimiterService } from './rate-limiter.service';

function keyOf(context: ExecutionContext) {
  const key = context.switchToHttp().getRequest<Request>().apiKey;
  // Ordering bug, not a client error: these guards must follow ApiKeyGuard.
  if (!key) throw new Error('Limit guards require ApiKeyGuard to run first');
  return key;
}

/** Per-key request rate limit; must run after ApiKeyGuard. */
@Injectable()
export class RateLimitGuard implements CanActivate {
  constructor(
    private readonly limiter: RateLimiterService,
    private readonly metrics: MetricsService,
  ) {}

  async canActivate(context: ExecutionContext): Promise<boolean> {
    const key = keyOf(context);
    const decision = await this.limiter.consume(key.id, key.rateLimitPerMinute);

    const response = context.switchToHttp().getResponse<Response>();
    response.setHeader('x-ratelimit-limit-requests', String(decision.limit));
    response.setHeader('x-ratelimit-remaining-requests', String(decision.remaining));

    if (!decision.allowed) {
      this.metrics.rejections.inc({ reason: 'rate_limited' });
      const retryAfterSeconds = Math.max(1, Math.ceil(decision.retryAfterMs / 1000));
      throw new ApiError(
        429,
        'rate_limit_error',
        `Rate limit of ${decision.limit} requests per minute exceeded`,
        'rate_limit_exceeded',
        null,
        { 'retry-after': String(retryAfterSeconds) },
      );
    }
    return true;
  }
}

/** Rejects keys that have reached their monthly budget; must run after ApiKeyGuard. */
@Injectable()
export class BudgetGuard implements CanActivate {
  constructor(
    private readonly budgets: BudgetService,
    private readonly metrics: MetricsService,
  ) {}

  async canActivate(context: ExecutionContext): Promise<boolean> {
    const key = keyOf(context);
    if (key.monthlyBudgetUsd === null) return true;

    const spent = await this.budgets.monthToDateSpend(key.id);
    if (spent >= key.monthlyBudgetUsd) {
      this.metrics.rejections.inc({ reason: 'budget_exceeded' });
      // Same type/code OpenAI uses for an exhausted quota, so clients that
      // already handle it stop retrying.
      throw new ApiError(
        429,
        'insufficient_quota',
        `Monthly budget of $${key.monthlyBudgetUsd.toFixed(2)} exhausted for this API key`,
        'insufficient_quota',
      );
    }
    return true;
  }
}
