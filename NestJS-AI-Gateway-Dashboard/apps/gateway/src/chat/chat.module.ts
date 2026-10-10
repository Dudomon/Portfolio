import { Module } from '@nestjs/common';
import { AuthModule } from '../auth/auth.module';
import { ResponseCacheService } from '../cache/response-cache.service';
import { LimitsModule } from '../limits/limits.module';
import { RoutingModule } from '../routing/routing.module';
import { UsageModule } from '../usage/usage.module';
import { ChatController } from './chat.controller';
import { ChatService } from './chat.service';

@Module({
  imports: [AuthModule, LimitsModule, RoutingModule, UsageModule],
  controllers: [ChatController],
  providers: [ChatService, ResponseCacheService],
})
export class ChatModule {}
