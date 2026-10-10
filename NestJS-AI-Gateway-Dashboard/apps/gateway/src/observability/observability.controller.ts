import { Controller, Get, Header, Inject, ServiceUnavailableException } from '@nestjs/common';
import type { Redis } from 'ioredis';
import { REDIS } from '../infra/redis.module';
import { PrismaService } from '../infra/prisma.service';
import { MetricsService } from './metrics.service';

/**
 * Operational endpoints. Not behind auth: they are meant to be reachable
 * only from inside the cluster (scrapers, kubelet), which is enforced at the
 * network layer rather than with a shared secret.
 */
@Controller()
export class ObservabilityController {
  constructor(
    private readonly metrics: MetricsService,
    private readonly prisma: PrismaService,
    @Inject(REDIS) private readonly redis: Redis,
  ) {}

  @Get('metrics')
  @Header('content-type', 'text/plain; version=0.0.4; charset=utf-8')
  scrape(): Promise<string> {
    return this.metrics.registry.metrics();
  }

  /** Liveness: the process is up and the event loop is responsive. */
  @Get('healthz')
  live(): { status: 'ok' } {
    return { status: 'ok' };
  }

  /**
   * Readiness: the database must be reachable (usage writes and key lookups
   * depend on it). Redis is reported but not required, because rate limiting
   * and caching fail open.
   */
  @Get('readyz')
  async ready(): Promise<{ status: 'ok'; redis: 'ok' | 'degraded' }> {
    try {
      await this.prisma.$queryRaw`SELECT 1`;
    } catch {
      throw new ServiceUnavailableException('database unreachable');
    }
    const redis = await this.redis
      .ping()
      .then(() => 'ok' as const)
      .catch(() => 'degraded' as const);
    return { status: 'ok', redis };
  }
}
