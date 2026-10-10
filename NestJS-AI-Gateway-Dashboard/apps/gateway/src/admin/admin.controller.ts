import {
  Body,
  Controller,
  Delete,
  Get,
  Inject,
  Param,
  Patch,
  Post,
  Query,
  UseGuards,
} from '@nestjs/common';
import {
  aggregateQuerySchema,
  createApiKeyRequestSchema,
  dateRangeQuerySchema,
  requestLogQuerySchema,
  updateApiKeyRequestSchema,
  type AggregateQuery,
  type AggregateRow,
  type ApiKeyView,
  type CreateApiKeyResponse,
  type DateRangeQuery,
  type GatewayConfig,
  type ProviderStats,
  type RequestLogPage,
  type RouteView,
  type TimeseriesPoint,
  type UpdateApiKeyRequest,
  type UsageSummary,
} from '@aigw/shared';
import type { z } from 'zod';
import { AdminGuard } from '../auth/admin.guard';
import { ZodValidationPipe } from '../common/zod-validation.pipe';
import { GATEWAY_CONFIG } from '../config/config.module';
import { CircuitBreakerRegistry } from '../routing/circuit-breaker.registry';
import { RouterService } from '../routing/router.service';
import { KeyManagementService } from './key-management.service';
import { UsageAnalyticsService } from './usage-analytics.service';

/** Admin API backing the dashboard. Every route requires the admin token. */
@Controller('admin')
@UseGuards(AdminGuard)
export class AdminController {
  constructor(
    private readonly analytics: UsageAnalyticsService,
    private readonly keys: KeyManagementService,
    private readonly router: RouterService,
    private readonly breakers: CircuitBreakerRegistry,
    @Inject(GATEWAY_CONFIG) private readonly config: GatewayConfig,
  ) {}

  @Get('usage/summary')
  summary(
    @Query(new ZodValidationPipe(dateRangeQuerySchema)) query: DateRangeQuery,
  ): Promise<UsageSummary> {
    return this.analytics.summary(query);
  }

  @Get('usage/timeseries')
  timeseries(
    @Query(new ZodValidationPipe(dateRangeQuerySchema)) query: DateRangeQuery,
  ): Promise<TimeseriesPoint[]> {
    return this.analytics.timeseries(query);
  }

  @Get('usage/providers')
  providers(
    @Query(new ZodValidationPipe(dateRangeQuerySchema)) query: DateRangeQuery,
  ): Promise<ProviderStats[]> {
    return this.analytics.providers(query);
  }

  @Get('usage/aggregate')
  aggregate(
    @Query(new ZodValidationPipe(aggregateQuerySchema)) query: AggregateQuery,
  ): Promise<AggregateRow[]> {
    const { groupBy, ...range } = query;
    return this.analytics.aggregate(groupBy, range);
  }

  @Get('requests')
  requests(
    @Query(new ZodValidationPipe(requestLogQuerySchema))
    query: z.output<typeof requestLogQuerySchema>,
  ): Promise<RequestLogPage> {
    return this.analytics.requests(query);
  }

  @Get('routes')
  routes(): RouteView[] {
    return Object.entries(this.config.routes).map(([alias, route]) => ({
      alias,
      description: route.description,
      targets: route.targets.map((target) => {
        const price = this.config.prices[target.provider]?.[target.model];
        return {
          provider: target.provider,
          model: target.model,
          enabled: this.router.isEnabled(target.provider),
          circuit: this.breakers.state(target.provider),
          inputPerMillionUsd: price?.inputPerMillionUsd ?? null,
          outputPerMillionUsd: price?.outputPerMillionUsd ?? null,
        };
      }),
    }));
  }

  @Get('keys')
  listKeys(): Promise<ApiKeyView[]> {
    return this.keys.list();
  }

  @Post('keys')
  createKey(
    @Body(new ZodValidationPipe(createApiKeyRequestSchema))
    body: z.output<typeof createApiKeyRequestSchema>,
  ): Promise<CreateApiKeyResponse> {
    return this.keys.create(body);
  }

  @Patch('keys/:id')
  updateKey(
    @Param('id') id: string,
    @Body(new ZodValidationPipe(updateApiKeyRequestSchema)) body: UpdateApiKeyRequest,
  ): Promise<ApiKeyView> {
    return this.keys.update(id, body);
  }

  @Delete('keys/:id')
  revokeKey(@Param('id') id: string): Promise<ApiKeyView> {
    return this.keys.revoke(id);
  }
}
