import {
  abortReason,
  type ChatCompletion,
  type ChatCompletionChunk,
  type ChatCompletionRequest,
  type GatewayConfig,
  type ProviderId,
} from '@aigw/shared';
import type { LlmProvider, ProviderCallContext } from '../src/providers/llm-provider';
import { UpstreamError } from '../src/providers/upstream-error';
import { MetricsService } from '../src/observability/metrics.service';
import { CircuitBreakerRegistry } from '../src/routing/circuit-breaker.registry';
import { RouterService } from '../src/routing/router.service';

/**
 * One scripted behavior per call: 'ok', a status to fail with, 'hang' (until
 * aborted), or 'midstream' (streams one chunk, then fails).
 */
export type Step = 'ok' | 'hang' | 'midstream' | number;

export class FakeProvider implements LlmProvider {
  readonly calls: ChatCompletionRequest[] = [];

  constructor(
    readonly id: ProviderId,
    private readonly script: Step[],
  ) {}

  private next(request: ChatCompletionRequest, context: ProviderCallContext): Promise<void> {
    this.calls.push(request);
    const step = this.script[this.calls.length - 1] ?? 'ok';
    if (step === 'ok' || step === 'midstream') return Promise.resolve();
    if (step === 'hang') {
      return new Promise((_resolve, reject) => {
        context.signal.addEventListener('abort', () => reject(abortReason(context.signal)), {
          once: true,
        });
      });
    }
    return Promise.reject(UpstreamError.fromStatus(this.id, step, 'scripted failure'));
  }

  async complete(
    request: ChatCompletionRequest,
    context: ProviderCallContext,
  ): Promise<ChatCompletion> {
    await this.next(request, context);
    return completion(this.id, request.model);
  }

  async *stream(
    request: ChatCompletionRequest,
    context: ProviderCallContext,
  ): AsyncGenerator<ChatCompletionChunk, void, undefined> {
    await this.next(request, context);
    yield chunk(request.model, { role: 'assistant', content: `from ${this.id}` });
    if (this.script[this.calls.length - 1] === 'midstream') {
      throw UpstreamError.fromStatus(this.id, 500, 'connection reset mid-stream');
    }
    yield {
      ...chunk(request.model, {}),
      choices: [],
      usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 },
    };
  }
}

export function completion(provider: string, model: string): ChatCompletion {
  return {
    id: `cmpl-${provider}`,
    object: 'chat.completion',
    created: 0,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: `from ${provider}` },
        finish_reason: 'stop',
      },
    ],
    usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
  };
}

function chunk(
  model: string,
  delta: ChatCompletionChunk['choices'][number]['delta'],
): ChatCompletionChunk {
  return {
    id: 'chunk',
    object: 'chat.completion.chunk',
    created: 0,
    model,
    choices: [{ index: 0, delta, finish_reason: null }],
  };
}

export function testConfig(overrides: Partial<GatewayConfig['routing']> = {}): GatewayConfig {
  return {
    routes: {
      fast: {
        description: 'test route',
        targets: [
          { provider: 'openai', model: 'primary-model' },
          { provider: 'anthropic', model: 'fallback-model' },
        ],
      },
    },
    prices: {
      openai: { 'primary-model': { inputPerMillionUsd: 1, outputPerMillionUsd: 2 } },
      anthropic: { 'fallback-model': { inputPerMillionUsd: 3, outputPerMillionUsd: 4 } },
    },
    routing: {
      attemptTimeoutMs: 1_000,
      maxRetriesPerTarget: 1,
      backoff: { baseMs: 1, maxMs: 2 },
      circuitBreaker: { failureThreshold: 3, cooldownMs: 60_000 },
      ...overrides,
    },
    defaultMaxTokens: 1_024,
  };
}

export function createRouter(providers: LlmProvider[], config = testConfig()) {
  const metrics = new MetricsService();
  const breakers = new CircuitBreakerRegistry(config, metrics);
  return { router: new RouterService(providers, config, breakers, metrics), breakers, metrics };
}

export const request: ChatCompletionRequest = {
  model: 'fast',
  messages: [{ role: 'user', content: 'hello' }],
};
