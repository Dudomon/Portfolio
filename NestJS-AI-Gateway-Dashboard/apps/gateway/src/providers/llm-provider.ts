import type {
  ChatCompletion,
  ChatCompletionChunk,
  ChatCompletionRequest,
  ProviderId,
} from '@aigw/shared';

export interface ProviderCallContext {
  /** Aborts on client disconnect or attempt timeout. Adapters must honor it. */
  signal: AbortSignal;
  requestId: string;
}

/**
 * Contract every upstream adapter implements. Requests and responses use the
 * OpenAI wire format; `request.model` has already been resolved from the
 * client's alias to the provider's concrete model ID.
 *
 * Adapter obligations, relied on by the router and usage tracking:
 * - `complete` returns a completion whose `usage` is populated.
 * - `stream` yields OpenAI chunks and finishes with exactly one usage-only
 *   chunk (`choices: []`, `usage` set), regardless of what the client asked.
 * - No network I/O happens until the stream is first iterated, so errors
 *   before the first chunk are still eligible for fallback.
 * - Upstream HTTP and network failures are thrown as `UpstreamError`.
 */
export interface LlmProvider {
  readonly id: ProviderId;
  complete(request: ChatCompletionRequest, context: ProviderCallContext): Promise<ChatCompletion>;
  stream(
    request: ChatCompletionRequest,
    context: ProviderCallContext,
  ): AsyncIterable<ChatCompletionChunk>;
}

/** DI token for the list of configured providers. */
export const LLM_PROVIDERS = Symbol('LLM_PROVIDERS');
