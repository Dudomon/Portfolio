import { Inject, Injectable } from '@nestjs/common';
import type {
  ChatCompletion,
  ChatCompletionChunk,
  ChatCompletionRequest,
  GatewayConfig,
  RouteTarget,
  Usage,
} from '@aigw/shared';
import type { AuthenticatedKey } from '../auth/api-key.service';
import { ResponseCacheService } from '../cache/response-cache.service';
import { ApiError } from '../common/api-error';
import { GATEWAY_CONFIG } from '../config/config.module';
import { BudgetService } from '../limits/budget.service';
import { MetricsService } from '../observability/metrics.service';
import { UpstreamError } from '../providers/upstream-error';
import { RouterService, type RoutingContext } from '../routing/router.service';
import { ClientAbortedError, isRoutingFailure } from '../routing/routing.errors';
import { calculateCost, priceFor } from '../usage/pricing';
import { UsageRecorder } from '../usage/usage-recorder.service';

export interface CompletionResult {
  completion: ChatCompletion;
  target: RouteTarget | null;
  attempts: number;
  cached: boolean;
}

export interface StreamResult {
  target: RouteTarget;
  attempts: number;
  /** Chunks to forward. Usage is recorded when iteration ends, however it ends. */
  chunks: AsyncGenerator<ChatCompletionChunk, void, undefined>;
}

interface Outcome {
  target: RouteTarget | null;
  attempts: number;
  usage?: Usage | undefined;
  error?: unknown;
  cached?: boolean;
}

/**
 * Orchestrates a completion: cache lookup, routing, and usage/cost
 * accounting. HTTP concerns (SSE framing, headers, disconnects) stay in the
 * controller.
 */
@Injectable()
export class ChatService {
  constructor(
    private readonly router: RouterService,
    private readonly cache: ResponseCacheService,
    private readonly usage: UsageRecorder,
    private readonly budgets: BudgetService,
    private readonly metrics: MetricsService,
    @Inject(GATEWAY_CONFIG) private readonly config: GatewayConfig,
  ) {}

  async complete(
    request: ChatCompletionRequest,
    key: AuthenticatedKey,
    context: RoutingContext,
  ): Promise<CompletionResult> {
    const started = performance.now();
    const record = (outcome: Outcome): void =>
      this.record(request, key, context, started, false, outcome);

    // Resolve first so unknown models fail with 404 before touching the cache.
    this.router.resolve(request.model);

    const cacheable = this.cache.isCacheable(request);
    if (cacheable) {
      const hit = await this.cache.get(key.projectId, request);
      if (hit) {
        record({ target: null, attempts: 0, cached: true });
        return { completion: hit, target: null, attempts: 0, cached: true };
      }
    }

    try {
      const routed = await this.router.complete(request, context);
      record({ target: routed.target, attempts: routed.attempts, usage: routed.value.usage });
      if (cacheable) void this.cache.set(key.projectId, request, routed.value);
      return {
        completion: routed.value,
        target: routed.target,
        attempts: routed.attempts,
        cached: false,
      };
    } catch (error) {
      record(failureOutcome(error));
      throw error;
    }
  }

  async stream(
    request: ChatCompletionRequest,
    key: AuthenticatedKey,
    context: RoutingContext,
  ): Promise<StreamResult> {
    const started = performance.now();
    const routed = await this.router.stream(request, context).catch((error: unknown) => {
      this.record(request, key, context, started, true, failureOutcome(error));
      throw error;
    });

    const { target, attempts } = routed;
    const includeUsage = request.stream_options?.include_usage === true;
    const upstream = routed.value.chunks;
    const record = (outcome: Outcome): void =>
      this.record(request, key, context, started, true, outcome);

    async function* chunks(): AsyncGenerator<ChatCompletionChunk, void, undefined> {
      let usage: Usage | undefined;
      let error: unknown;
      try {
        for await (const chunk of upstream) {
          if (chunk.usage) {
            usage = chunk.usage;
            // Adapters always report usage; forward it only if the client
            // opted in, as OpenAI does.
            if (!includeUsage) continue;
          }
          yield chunk;
        }
      } catch (caught) {
        error = caught;
        throw caught;
      } finally {
        // Runs on completion, on error, and when the consumer stops early,
        // so every stream produces exactly one usage record. A client that
        // leaves before the final usage chunk is recorded as aborted; its
        // partial output is not billed because providers report usage last.
        if (context.signal.aborted && !usage) error = new ClientAbortedError(attempts, target);
        record({ target, attempts, usage, error });
      }
    }

    return { target, attempts, chunks: chunks() };
  }

  private record(
    request: ChatCompletionRequest,
    key: AuthenticatedKey,
    context: RoutingContext,
    started: number,
    streamed: boolean,
    outcome: Outcome,
  ): void {
    const { target, usage, error } = outcome;
    // Cached answers cost nothing upstream; their tokens are not re-billed.
    const billable = outcome.cached ? undefined : usage;
    const cost = target
      ? calculateCost(priceFor(this.config, target.provider, target.model), billable)
      : 0;

    if (target && billable) {
      const labels = { provider: target.provider, model: target.model };
      this.metrics.tokens.inc({ ...labels, direction: 'prompt' }, billable.prompt_tokens);
      this.metrics.tokens.inc({ ...labels, direction: 'completion' }, billable.completion_tokens);
      this.metrics.costUsd.inc(labels, cost);
      void this.budgets.addSpend(key.id, cost);
    }

    this.usage.record({
      requestId: context.requestId,
      apiKeyId: key.id,
      projectId: key.projectId,
      alias: request.model,
      provider: target?.provider ?? null,
      model: target?.model ?? null,
      status: error ? 'error' : 'success',
      errorCode: error ? errorCode(error) : null,
      promptTokens: billable?.prompt_tokens ?? 0,
      completionTokens: billable?.completion_tokens ?? 0,
      costUsd: cost,
      latencyMs: Math.round(performance.now() - started),
      attempts: outcome.attempts,
      cached: outcome.cached ?? false,
      streamed,
      createdAt: new Date(),
    });
  }
}

function failureOutcome(error: unknown): Outcome {
  return isRoutingFailure(error)
    ? { target: error.target, attempts: error.attempts, error }
    : { target: null, attempts: 0, error };
}

function errorCode(error: unknown): string {
  if (error instanceof ApiError) return error.code ?? error.type;
  if (error instanceof UpstreamError) return `upstream_${error.kind}`;
  return 'internal_error';
}
