import { Injectable } from '@nestjs/common';
import type { CircuitState } from '@aigw/shared';
import { Counter, Gauge, Histogram, Registry, collectDefaultMetrics } from 'prom-client';

const CIRCUIT_STATE_VALUE: Record<CircuitState, number> = { closed: 0, half_open: 1, open: 2 };

/**
 * Prometheus instruments. A dedicated registry (instead of prom-client's
 * global one) keeps tests isolated and makes the exported set explicit.
 *
 * Labels deliberately exclude API key and request IDs: per-tenant breakdowns
 * belong in the usage database, not in a time series with unbounded cardinality.
 */
@Injectable()
export class MetricsService {
  readonly registry = new Registry();

  readonly httpDuration = new Histogram({
    name: 'gateway_http_request_duration_seconds',
    help: 'Gateway HTTP request duration, including full stream duration.',
    labelNames: ['method', 'route', 'status'] as const,
    buckets: [0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120],
    registers: [this.registry],
  });

  readonly upstreamAttempts = new Counter({
    name: 'gateway_upstream_attempts_total',
    help: 'Upstream provider attempts by outcome (success or error kind).',
    labelNames: ['provider', 'model', 'outcome'] as const,
    registers: [this.registry],
  });

  readonly upstreamLatency = new Histogram({
    name: 'gateway_upstream_latency_seconds',
    help: 'Upstream latency to full response (non-streaming) or first chunk (streaming).',
    labelNames: ['provider', 'model', 'streamed'] as const,
    buckets: [0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64],
    registers: [this.registry],
  });

  readonly fallbacks = new Counter({
    name: 'gateway_fallbacks_total',
    help: 'Requests answered by a target other than the first in their route.',
    labelNames: ['alias'] as const,
    registers: [this.registry],
  });

  readonly tokens = new Counter({
    name: 'gateway_tokens_total',
    help: 'Tokens processed, by direction.',
    labelNames: ['provider', 'model', 'direction'] as const,
    registers: [this.registry],
  });

  readonly costUsd = new Counter({
    name: 'gateway_cost_usd_total',
    help: 'Estimated spend in USD from the configured price table.',
    labelNames: ['provider', 'model'] as const,
    registers: [this.registry],
  });

  readonly circuitState = new Gauge({
    name: 'gateway_circuit_state',
    help: 'Circuit breaker state per provider (0 closed, 1 half-open, 2 open).',
    labelNames: ['provider'] as const,
    registers: [this.registry],
  });

  readonly rejections = new Counter({
    name: 'gateway_rejections_total',
    help: 'Requests rejected before reaching a provider.',
    labelNames: ['reason'] as const,
    registers: [this.registry],
  });

  readonly cacheLookups = new Counter({
    name: 'gateway_cache_lookups_total',
    help: 'Response cache lookups by result.',
    labelNames: ['result'] as const,
    registers: [this.registry],
  });

  readonly usageBufferSize = new Gauge({
    name: 'gateway_usage_buffer_size',
    help: 'Usage records waiting to be written to the database.',
    registers: [this.registry],
  });

  readonly usageDropped = new Counter({
    name: 'gateway_usage_records_dropped_total',
    help: 'Usage records discarded because the buffer overflowed during a database outage.',
    registers: [this.registry],
  });

  constructor() {
    collectDefaultMetrics({ register: this.registry });
  }

  setCircuitState(provider: string, state: CircuitState): void {
    this.circuitState.set({ provider }, CIRCUIT_STATE_VALUE[state]);
  }
}
