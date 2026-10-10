import { Inject, Injectable, Logger } from '@nestjs/common';
import type { Redis } from 'ioredis';
import { REDIS } from '../infra/redis.module';
import { PrismaService } from '../infra/prisma.service';

/** Adds to the counter only if it exists, so a missing counter is re-seeded from the database. */
const INCREMENT_IF_EXISTS = `
if redis.call('EXISTS', KEYS[1]) == 1 then
  return redis.call('INCRBYFLOAT', KEYS[1], ARGV[1])
end
return false
`;

const COUNTER_TTL_SECONDS = 35 * 24 * 60 * 60;

type RedisWithBudget = Redis & {
  incrementIfExists(key: string, amount: string): Promise<string | null>;
};

/**
 * Month-to-date spend per API key.
 *
 * PostgreSQL is the source of truth; Redis holds a running counter so the
 * per-request check is O(1). The counter is seeded from the database on
 * first use each month (or after Redis loses it) and incremented as soon as
 * a request completes, ahead of the buffered database write.
 *
 * Budgets are soft limits: requests already in flight when the budget is
 * crossed are allowed to finish, so spend can overshoot by at most the cost
 * of the concurrent requests.
 */
@Injectable()
export class BudgetService {
  private readonly logger = new Logger(BudgetService.name);
  private readonly redis: RedisWithBudget;

  constructor(
    @Inject(REDIS) redis: Redis,
    private readonly prisma: PrismaService,
  ) {
    redis.defineCommand('incrementIfExists', { numberOfKeys: 1, lua: INCREMENT_IF_EXISTS });
    this.redis = redis as RedisWithBudget;
  }

  async monthToDateSpend(keyId: string, now = new Date()): Promise<number> {
    const counterKey = this.counterKey(keyId, now);
    try {
      const cached = await this.redis.get(counterKey);
      if (cached !== null) return Number(cached);
    } catch (error) {
      this.logger.warn(`Budget counter unavailable, using database: ${(error as Error).message}`);
      return this.spendFromDatabase(keyId, now);
    }

    const spend = await this.spendFromDatabase(keyId, now);
    // NX: if a concurrent request seeded (and possibly incremented) the
    // counter meanwhile, keep its value rather than overwrite it.
    await this.redis
      .set(counterKey, spend.toString(), 'EX', COUNTER_TTL_SECONDS, 'NX')
      .catch(() => undefined);
    return spend;
  }

  async addSpend(keyId: string, costUsd: number, now = new Date()): Promise<void> {
    if (costUsd <= 0) return;
    await this.redis
      .incrementIfExists(this.counterKey(keyId, now), costUsd.toFixed(8))
      .catch((error: Error) =>
        this.logger.warn(`Failed to update budget counter: ${error.message}`),
      );
  }

  private async spendFromDatabase(keyId: string, now: Date): Promise<number> {
    const result = await this.prisma.usageRecord.aggregate({
      where: { apiKeyId: keyId, createdAt: { gte: monthStart(now) } },
      _sum: { costUsd: true },
    });
    return result._sum.costUsd?.toNumber() ?? 0;
  }

  private counterKey(keyId: string, now: Date): string {
    return `budget:${keyId}:${now.toISOString().slice(0, 7)}`;
  }
}

export function monthStart(now: Date): Date {
  return new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), 1));
}
