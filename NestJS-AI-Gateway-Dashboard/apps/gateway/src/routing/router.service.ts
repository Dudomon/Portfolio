import { Inject, Injectable, Logger } from '@nestjs/common';
import { SpanStatusCode, trace } from '@opentelemetry/api';
import {
  fullJitterBackoff,
  sleep,
  type ChatCompletion,
  type ChatCompletionChunk,
  type ChatCompletionRequest,
  type GatewayConfig,
  type ProviderId,
  type Route,
  type RouteTarget,
} from '@aigw/shared';
import { ApiError } from '../common/api-error';
import { GATEWAY_CONFIG } from '../config/config.module';
import { MetricsService } from '../observability/metrics.service';
import { LLM_PROVIDERS, type LlmProvider } from '../providers/llm-provider';
import { UpstreamError } from '../providers/upstream-error';
import { CircuitBreakerRegistry } from './circuit-breaker.registry';
import { ClientAbortedError, RouteExhaustedError, UpstreamRejectedError } from './routing.errors';

export interface RoutingContext {
  /** Aborted when the client disconnects. */
  signal: AbortSignal;
  requestId: string;
}

export interface Routed<T> {
  value: T;
  target: RouteTarget;
  /** Upstream attempts made, including the successful one. */
  attempts: number;
}

export interface RoutedStream {
  chunks: AsyncIterable<ChatCompletionChunk>;
}

type Attempt<T> = (
  provider: LlmProvider,
  request: ChatCompletionRequest,
  signal: AbortSignal,
) => Promise<T>;

const tracer = trace.getTracer('ai-gateway');

/**
 * Resolves a model alias to its target chain and executes the request with
 * retries, fallback and per-provider circuit breaking.
 *
 * Semantics, per target in priority order:
 * 1. Skip targets whose provider is unconfigured or whose circuit is open.
 * 2. Attempt with a per-attempt deadline linked to the client's signal.
 * 3. server/timeout/network faults retry the same target (jittered backoff)
 *    up to `maxRetriesPerTarget`; rate_limit and misconfigured move straight
 *    to the next target because retrying immediately rarely helps.
 * 4. Client errors (400-class) stop routing: every provider would reject them.
 * 5. A client abort stops everything and is never counted against a provider.
 *
 * Streams are committed once the first chunk arrives. Before that, failures
 * fall back like any other; after it, bytes are on the wire and the error is
 * surfaced in-band by the caller.
 */
@Injectable()
export class RouterService {
  private readonly logger = new Logger(RouterService.name);
  private readonly providers: ReadonlyMap<ProviderId, LlmProvider>;

  constructor(
    @Inject(LLM_PROVIDERS) providers: LlmProvider[],
    @Inject(GATEWAY_CONFIG) private readonly config: GatewayConfig,
    private readonly breakers: CircuitBreakerRegistry,
    private readonly metrics: MetricsService,
  ) {
    this.providers = new Map(providers.map((p) => [p.id, p]));
  }

  isEnabled(provider: ProviderId): boolean {
    return this.providers.has(provider);
  }

  resolve(alias: string): Route {
    const route = this.config.routes[alias];
    if (!route) {
      throw ApiError.notFound(`The model "${alias}" does not exist`, 'model_not_found');
    }
    return route;
  }

  complete(
    request: ChatCompletionRequest,
    context: RoutingContext,
  ): Promise<Routed<ChatCompletion>> {
    return this.execute(request, context, false, (provider, upstream, signal) =>
      provider.complete(upstream, { signal, requestId: context.requestId }),
    );
  }

  async stream(
    request: ChatCompletionRequest,
    context: RoutingContext,
  ): Promise<Routed<RoutedStream>> {
    const routed = await this.execute(
      request,
      context,
      true,
      async (provider, upstream, signal) => {
        const iterator = provider
          .stream(upstream, { signal, requestId: context.requestId })
          [Symbol.asyncIterator]();
        try {
          // Pulling the first chunk inside the attempt is what makes pre-commit
          // failures (connect errors, 429s, early stream errors) fall back.
          const first = await iterator.next();
          return { iterator, first };
        } catch (error) {
          await iterator.return?.().catch(() => undefined);
          throw error;
        }
      },
    );

    const { iterator, first } = routed.value;
    async function* chunks(): AsyncGenerator<ChatCompletionChunk, void, undefined> {
      if (first.done) return;
      yield first.value;
      yield* { [Symbol.asyncIterator]: () => iterator };
    }
    return { ...routed, value: { chunks: chunks() } };
  }

  private async execute<T>(
    request: ChatCompletionRequest,
    context: RoutingContext,
    streamed: boolean,
    attempt: Attempt<T>,
  ): Promise<Routed<T>> {
    const route = this.resolve(request.model);
    const { maxRetriesPerTarget, backoff } = this.config.routing;
    let attempts = 0;
    let lastError: UpstreamError | undefined;
    let lastTarget: RouteTarget | null = null;

    for (const [position, target] of route.targets.entries()) {
      const provider = this.providers.get(target.provider);
      if (!provider) continue;
      const breaker = this.breakers.get(target.provider);

      for (let retry = 0; retry <= maxRetriesPerTarget; retry++) {
        if (!breaker.tryAcquire()) {
          this.metrics.upstreamAttempts.inc({ ...labels(target), outcome: 'circuit_open' });
          break;
        }
        attempts += 1;
        lastTarget = target;

        const started = performance.now();
        try {
          const value = await this.attemptWithDeadline(provider, target, request, context, attempt);
          breaker.onSuccess();
          this.metrics.upstreamAttempts.inc({ ...labels(target), outcome: 'success' });
          this.metrics.upstreamLatency.observe(
            { ...labels(target), streamed: String(streamed) },
            (performance.now() - started) / 1000,
          );
          if (position > 0) this.metrics.fallbacks.inc({ alias: request.model });
          return { value, target, attempts };
        } catch (error) {
          if (!(error instanceof UpstreamError)) {
            // A client abort, or an ApiError from an adapter rejecting input
            // it cannot translate: neither reflects provider health.
            breaker.release();
            if (error instanceof ClientAbortedError) throw new ClientAbortedError(attempts, target);
            throw error;
          }
          this.metrics.upstreamAttempts.inc({ ...labels(target), outcome: error.kind });
          if (!error.isProviderFault) {
            breaker.release();
            throw new UpstreamRejectedError(error, attempts, target);
          }

          breaker.onFailure();
          lastError = error;
          this.logger.warn(
            `Attempt failed [${context.requestId}] ${target.provider}/${target.model}: ${error.message}`,
          );
          if (!error.retryableOnSameTarget || retry === maxRetriesPerTarget) break;
          await sleep(fullJitterBackoff(retry, backoff), context.signal).catch(() => {
            throw new ClientAbortedError(attempts, target);
          });
        }
      }
    }

    throw new RouteExhaustedError(request.model, attempts, lastTarget, lastError);
  }

  /**
   * Runs one attempt under a deadline. A manual timer (rather than
   * AbortSignal.timeout) is cleared once the attempt resolves so a committed
   * stream is not cut off later; the client's signal stays linked so a
   * disconnect still cancels the upstream request.
   */
  private async attemptWithDeadline<T>(
    provider: LlmProvider,
    target: RouteTarget,
    request: ChatCompletionRequest,
    context: RoutingContext,
    attempt: Attempt<T>,
  ): Promise<T> {
    const deadline = new AbortController();
    const timer = setTimeout(
      () => deadline.abort(new Error('attempt deadline exceeded')),
      this.config.routing.attemptTimeoutMs,
    );
    const signal = AbortSignal.any([context.signal, deadline.signal]);

    return tracer.startActiveSpan('gateway.upstream_attempt', async (span) => {
      span.setAttributes({
        'gen_ai.system': target.provider,
        'gen_ai.request.model': target.model,
        'gateway.request_id': context.requestId,
      });
      try {
        return await attempt(provider, { ...request, model: target.model }, signal);
      } catch (error) {
        const classified = this.classify(error, provider.id, context.signal, deadline.signal);
        span.recordException(classified);
        span.setStatus({ code: SpanStatusCode.ERROR, message: classified.message });
        throw classified;
      } finally {
        clearTimeout(timer);
        span.end();
      }
    });
  }

  private classify(
    error: unknown,
    provider: ProviderId,
    clientSignal: AbortSignal,
    deadline: AbortSignal,
  ): Error {
    if (clientSignal.aborted) return new ClientAbortedError(0, null);
    if (deadline.aborted) {
      return new UpstreamError(
        provider,
        'timeout',
        `${provider} did not respond within ${this.config.routing.attemptTimeoutMs}ms`,
      );
    }
    if (error instanceof UpstreamError || error instanceof ApiError) return error;
    // Anything else (malformed upstream JSON, an adapter bug) is treated as a
    // provider fault so the request can still fall back, but logged loudly
    // because it may hide a defect rather than an outage.
    this.logger.error(error instanceof Error ? (error.stack ?? error.message) : String(error));
    return new UpstreamError(
      provider,
      'server',
      `${provider} adapter failed unexpectedly`,
      null,
      undefined,
      {
        cause: error,
      },
    );
  }
}

function labels(target: RouteTarget): { provider: string; model: string } {
  return { provider: target.provider, model: target.model };
}
