import { Inject, Injectable } from '@nestjs/common';
import type { CircuitState, GatewayConfig, ProviderId } from '@aigw/shared';
import { GATEWAY_CONFIG } from '../config/config.module';
import { MetricsService } from '../observability/metrics.service';
import { CircuitBreaker } from './circuit-breaker';

/**
 * One breaker per provider, held in process memory.
 *
 * Breakers are intentionally not shared through Redis: each replica judges
 * the providers from its own vantage point (network path, DNS, egress), and
 * keeping them local avoids a Redis round trip on every attempt.
 */
@Injectable()
export class CircuitBreakerRegistry {
  private readonly breakers = new Map<ProviderId, CircuitBreaker>();

  constructor(
    @Inject(GATEWAY_CONFIG) private readonly config: GatewayConfig,
    private readonly metrics: MetricsService,
  ) {}

  get(provider: ProviderId): CircuitBreaker {
    let breaker = this.breakers.get(provider);
    if (!breaker) {
      breaker = new CircuitBreaker(this.config.routing.circuitBreaker, (state) =>
        this.metrics.setCircuitState(provider, state),
      );
      this.breakers.set(provider, breaker);
      this.metrics.setCircuitState(provider, 'closed');
    }
    return breaker;
  }

  state(provider: ProviderId): CircuitState {
    return this.breakers.get(provider)?.currentState ?? 'closed';
  }
}
