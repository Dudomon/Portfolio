import {
  readSseStream,
  type ChatCompletion,
  type ChatCompletionChunk,
  type ChatCompletionRequest,
  type ModelList,
} from '@aigw/shared';
import { StreamError, isErrorBody } from './errors';
import { HttpClient, type HttpOptions } from './http';

export interface GatewayClientOptions extends Omit<HttpOptions, 'token'> {
  /** A gateway API key (`gw_...`). */
  apiKey: string;
}

export interface CallOptions {
  signal?: AbortSignal;
}

type NonStreamingRequest = Omit<ChatCompletionRequest, 'stream'> & { stream?: false };
type StreamingRequest = Omit<ChatCompletionRequest, 'stream'>;

/**
 * Client for the OpenAI-compatible surface of the gateway.
 *
 * @example
 * const gw = new GatewayClient({ baseUrl: 'http://localhost:3000', apiKey });
 * for await (const chunk of gw.chat.completions.stream({ model: 'fast', messages })) {
 *   process.stdout.write(chunk.choices[0]?.delta.content ?? '');
 * }
 */
export class GatewayClient {
  private readonly http: HttpClient;

  readonly chat: {
    completions: {
      create: (request: NonStreamingRequest, options?: CallOptions) => Promise<ChatCompletion>;
      stream: (
        request: StreamingRequest,
        options?: CallOptions,
      ) => AsyncGenerator<ChatCompletionChunk, void, undefined>;
    };
  };

  readonly models: {
    list: (options?: CallOptions) => Promise<ModelList>;
  };

  constructor({ apiKey, ...options }: GatewayClientOptions) {
    this.http = new HttpClient({ ...options, token: apiKey });
    this.chat = {
      completions: {
        create: (request, callOptions) => this.createCompletion(request, callOptions),
        stream: (request, callOptions) => this.streamCompletion(request, callOptions),
      },
    };
    this.models = { list: (callOptions) => this.listModels(callOptions) };
  }

  private createCompletion(
    request: NonStreamingRequest,
    options: CallOptions = {},
  ): Promise<ChatCompletion> {
    return this.http.json<ChatCompletion>({
      method: 'POST',
      path: '/v1/chat/completions',
      body: { ...request, stream: false },
      ...(options.signal ? { signal: options.signal } : {}),
    });
  }

  private async *streamCompletion(
    request: StreamingRequest,
    options: CallOptions = {},
  ): AsyncGenerator<ChatCompletionChunk, void, undefined> {
    const response = await this.http.request({
      method: 'POST',
      path: '/v1/chat/completions',
      body: { ...request, stream: true },
      stream: true,
      ...(options.signal ? { signal: options.signal } : {}),
    });
    if (!response.body) throw new StreamError('Response has no body', null, null);

    for await (const event of readSseStream(response.body)) {
      if (event.data === '[DONE]') return;
      const payload: unknown = JSON.parse(event.data);
      // Once headers are sent the gateway cannot change the status code, so
      // failures after the first byte arrive as an in-band error event.
      if (isErrorBody(payload)) {
        throw new StreamError(payload.error.message, payload.error.type, payload.error.code);
      }
      yield payload as ChatCompletionChunk;
    }
    throw new StreamError('Stream ended before [DONE]', null, null);
  }

  private listModels(options: CallOptions = {}): Promise<ModelList> {
    return this.http.json<ModelList>({
      method: 'GET',
      path: '/v1/models',
      ...(options.signal ? { signal: options.signal } : {}),
    });
  }
}
