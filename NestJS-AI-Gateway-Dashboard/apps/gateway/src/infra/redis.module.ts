import {
  Global,
  Inject,
  Injectable,
  Logger,
  Module,
  type OnApplicationShutdown,
} from '@nestjs/common';
import { Redis } from 'ioredis';
import { ENV } from '../config/config.module';
import type { AppEnv } from '../config/env';

export const REDIS = Symbol('REDIS');

@Injectable()
class RedisLifecycle implements OnApplicationShutdown {
  constructor(@Inject(REDIS) private readonly redis: Redis) {}

  async onApplicationShutdown(): Promise<void> {
    await this.redis.quit().catch(() => this.redis.disconnect());
  }
}

@Global()
@Module({
  providers: [
    {
      provide: REDIS,
      inject: [ENV],
      useFactory: (env: AppEnv): Redis => {
        const logger = new Logger('Redis');
        const redis = new Redis(env.REDIS_URL, {
          // Fail fast instead of queueing while disconnected: rate limiting
          // and caching degrade gracefully, but must never stall requests.
          enableOfflineQueue: false,
          maxRetriesPerRequest: 1,
          connectTimeout: 2_000,
        });
        // ioredis emits an error per reconnect attempt; log each distinct
        // failure once so an outage does not flood the logs.
        let lastError: string | undefined;
        redis.on('error', (error: Error) => {
          if (error.message === lastError) return;
          lastError = error.message;
          logger.warn(`Redis error: ${error.message}`);
        });
        redis.on('ready', () => {
          if (lastError) logger.log('Redis connection restored');
          lastError = undefined;
        });
        return redis;
      },
    },
    RedisLifecycle,
  ],
  exports: [REDIS],
})
export class RedisModule {}
