import { Module } from '@nestjs/common';
import { APP_FILTER } from '@nestjs/core';
import { AdminModule } from './admin/admin.module';
import { ChatModule } from './chat/chat.module';
import { ApiExceptionFilter } from './common/api-exception.filter';
import { ConfigModule } from './config/config.module';
import { PrismaModule } from './infra/prisma.service';
import { RedisModule } from './infra/redis.module';
import { ObservabilityModule } from './observability/observability.module';

@Module({
  imports: [ConfigModule, ObservabilityModule, PrismaModule, RedisModule, ChatModule, AdminModule],
  providers: [{ provide: APP_FILTER, useClass: ApiExceptionFilter }],
})
export class AppModule {}
