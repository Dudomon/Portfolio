import { Module } from '@nestjs/common';
import { BudgetService } from './budget.service';
import { BudgetGuard, RateLimitGuard } from './limits.guards';
import { RateLimiterService } from './rate-limiter.service';

@Module({
  providers: [RateLimiterService, BudgetService, RateLimitGuard, BudgetGuard],
  exports: [RateLimiterService, BudgetService, RateLimitGuard, BudgetGuard],
})
export class LimitsModule {}
