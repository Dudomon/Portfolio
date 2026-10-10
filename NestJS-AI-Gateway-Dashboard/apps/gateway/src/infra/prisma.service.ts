import { Global, Injectable, Module, type OnApplicationShutdown } from '@nestjs/common';
import { PrismaClient } from '@prisma/client';

/**
 * Prisma connects lazily on the first query, so startup does not hard-fail
 * when the database is briefly unavailable; /readyz reports it instead.
 */
@Injectable()
export class PrismaService extends PrismaClient implements OnApplicationShutdown {
  async onApplicationShutdown(): Promise<void> {
    await this.$disconnect();
  }
}

@Global()
@Module({
  providers: [PrismaService],
  exports: [PrismaService],
})
export class PrismaModule {}
