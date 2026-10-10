import {
  readSseStream,
  type ChatCompletion,
  type ChatCompletionChunk,
  type ChatCompletionRequest,
} from '@aigw/shared';
import { unixSeconds } from '../../common/clock';
import type { LlmProvider, ProviderCallContext } from '../llm-provider';
import { UpstreamError } from '../upstream-error';
import { postJson } from '../upstream-http';
import {
  AnthropicStreamTranslator,
  fromAnthropicResponse,
  toAnthropicRequest,
} from './anthropic.mapper';
import type { AnthropicResponse, AnthropicStreamEvent } from './anthropic.types';

const ANTHROPIC_VERSION = '2023-06-01';

export class AnthropicProvider implements LlmProvider {
  readonly id = 'anthropic' as const;

  constructor(
    private readonly apiKey: string,
    private readonly baseUrl: string,
    private readonly defaultMaxTokens: number,
  ) {}

  async complete(
    request: ChatCompletionRequest,
    context: ProviderCallContext,
  ): Promise<ChatCompletion> {
    const response = await this.post(toAnthropicRequest(request, this.defaultMaxTokens), context);
    const body = (await response.json()) as AnthropicResponse;
    return fromAnthropicResponse(body, unixSeconds());
  }

  async *stream(
    request: ChatCompletionRequest,
    context: ProviderCallContext,
  ): AsyncGenerator<ChatCompletionChunk, void, undefined> {
    const response = await this.post(
      { ...toAnthropicRequest(request, this.defaultMaxTokens), stream: true },
      context,
    );
    if (!response.body) throw new UpstreamError(this.id, 'server', 'anthropic sent no stream body');

    const translator = new AnthropicStreamTranslator(request.model, unixSeconds());
    for await (const event of readSseStream(response.body)) {
      yield* translator.translate(JSON.parse(event.data) as AnthropicStreamEvent);
    }
    // A connection dropped mid-stream ends the byte stream cleanly; without
    // message_stop the output is truncated and usage is unknown.
    if (!translator.finished) {
      throw new UpstreamError(this.id, 'server', 'anthropic stream ended before message_stop');
    }
  }

  private post(body: unknown, context: ProviderCallContext): Promise<Response> {
    return postJson(
      this.id,
      `${this.baseUrl.replace(/\/+$/, '')}/v1/messages`,
      { 'x-api-key': this.apiKey, 'anthropic-version': ANTHROPIC_VERSION },
      body,
      context.signal,
    );
  }
}
