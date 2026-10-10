import {
  chatCompletionChunkSchema,
  chatCompletionSchema,
  readSseStream,
  type ChatCompletion,
  type ChatCompletionChunk,
  type ChatCompletionRequest,
  type ProviderId,
} from '@aigw/shared';
import type { LlmProvider, ProviderCallContext } from '../llm-provider';
import { UpstreamError } from '../upstream-error';
import { parseUpstream, postJson } from '../upstream-http';

/**
 * Shared adapter for providers that speak the OpenAI wire format natively
 * (OpenAI and Azure OpenAI). Subclasses only describe addressing and auth.
 */
export abstract class OpenAiCompatibleProvider implements LlmProvider {
  abstract readonly id: ProviderId;

  protected abstract url(model: string): string;
  protected abstract headers(): Record<string, string>;
  /** Builds the upstream body; Azure addresses the model via the URL instead. */
  protected abstract body(request: ChatCompletionRequest): Record<string, unknown>;

  async complete(
    request: ChatCompletionRequest,
    context: ProviderCallContext,
  ): Promise<ChatCompletion> {
    const response = await this.post(request, { ...this.body(request), stream: false }, context);
    const completion = parseUpstream(this.id, chatCompletionSchema, await response.json());
    if (!completion.usage) {
      throw new UpstreamError(this.id, 'server', `${this.id} response is missing usage`);
    }
    return completion;
  }

  async *stream(
    request: ChatCompletionRequest,
    context: ProviderCallContext,
  ): AsyncGenerator<ChatCompletionChunk, void, undefined> {
    const response = await this.post(
      request,
      {
        ...this.body(request),
        stream: true,
        // Always request usage upstream: cost accounting depends on it even
        // when the client did not ask for it.
        stream_options: { include_usage: true },
      },
      context,
    );
    if (!response.body)
      throw new UpstreamError(this.id, 'server', `${this.id} sent no stream body`);

    let sawUsage = false;
    for await (const event of readSseStream(response.body)) {
      if (event.data === '[DONE]') break;
      const chunk = parseUpstream(this.id, chatCompletionChunkSchema, JSON.parse(event.data));
      // Azure emits prompt-filter bookkeeping chunks with no choices and no
      // usage; they carry nothing a client can use.
      if (chunk.choices.length === 0 && !chunk.usage) continue;
      if (chunk.usage) sawUsage = true;
      yield chunk;
    }
    if (!sawUsage) {
      throw new UpstreamError(this.id, 'server', `${this.id} stream ended without usage`);
    }
  }

  private post(
    request: ChatCompletionRequest,
    body: Record<string, unknown>,
    context: ProviderCallContext,
  ): Promise<Response> {
    return postJson(this.id, this.url(request.model), this.headers(), body, context.signal);
  }
}

/** Fields forwarded verbatim; everything else in the request is gateway-only. */
export function openAiBody(request: ChatCompletionRequest): Record<string, unknown> {
  const { stream: _stream, stream_options: _options, ...rest } = request;
  return rest;
}
