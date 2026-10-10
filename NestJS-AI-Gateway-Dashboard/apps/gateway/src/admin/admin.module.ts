import { Module } from '@nestjs/common';
import { AuthModule } from '../auth/auth.module';
import { RoutingModule } from '../routing/routing.module';
import { AdminController } from './admin.controller';
import { KeyManagementService } from './key-management.service';
import { UsageAnalyticsService } from './usage-analytics.service';

@Module({
  imports: [AuthModule, RoutingModule],
  controllers: [AdminController],
  providers: [UsageAnalyticsService, KeyManagementService],
})
export class AdminModule {}
