import { randomUUID } from 'node:crypto';
import {
  BedrockRuntimeClient,
  ConverseCommand,
  ConverseStreamCommand,
} from '@aws-sdk/client-bedrock-runtime';
import type { ChatCompletion, ChatCompletionChunk, ChatCompletionRequest } from '@aigw/shared';
import { unixSeconds } from '../../common/clock';
import type { LlmProvider, ProviderCallContext } from '../llm-provider';
import { UpstreamError, kindForStatus } from '../upstream-error';
import { ConverseStreamTranslator, fromConverseResponse, toConverseInput } from './bedrock.mapper';

/**
 * AWS Bedrock via the model-agnostic Converse API. Credentials resolve through
 * the default AWS provider chain (env, profile, IMDS/IRSA), so no secrets are
 * handled by the gateway itself.
 */
export class BedrockProvider implements LlmProvider {
  readonly id = 'bedrock' as const;
  private readonly client: BedrockRuntimeClient;

  constructor(
    region: string,
    private readonly defaultMaxTokens: number,
  ) {
    // The router owns retries and fallback; SDK-level retries would multiply
    // attempts and hide failures from the circuit breaker.
    this.client = new BedrockRuntimeClient({ region, maxAttempts: 1 });
  }

  async complete(
    request: ChatCompletionRequest,
    context: ProviderCallContext,
  ): Promise<ChatCompletion> {
    const command = new ConverseCommand({
      modelId: request.model,
      ...toConverseInput(request, this.defaultMaxTokens),
    });
    const response = await this.send(
      () => this.client.send(command, { abortSignal: context.signal }),
      context,
    );
    return fromConverseResponse(response, `chatcmpl-${randomUUID()}`, request.model, unixSeconds());
  }

  async *stream(
    request: ChatCompletionRequest,
    context: ProviderCallContext,
  ): AsyncGenerator<ChatCompletionChunk, void, undefined> {
    const command = new ConverseStreamCommand({
      modelId: request.model,
      ...toConverseInput(request, this.defaultMaxTokens),
    });
    const response = await this.send(
      () => this.client.send(command, { abortSignal: context.signal }),
      context,
    );
    if (!response.stream) throw new UpstreamError(this.id, 'server', 'bedrock sent no stream');

    const translator = new ConverseStreamTranslator(
      `chatcmpl-${randomUUID()}`,
      request.model,
      unixSeconds(),
    );
    try {
      for await (const event of response.stream) yield* translator.translate(event);
    } catch (error) {
      throw this.toUpstreamError(error, context);
    }
    if (!translator.finished) {
      throw new UpstreamError(this.id, 'server', 'bedrock stream ended before metadata');
    }
  }

  private async send<T>(call: () => Promise<T>, context: ProviderCallContext): Promise<T> {
    try {
      return await call();
    } catch (error) {
      throw this.toUpstreamError(error, context);
    }
  }

  /** Maps AWS SDK service exceptions onto the router's error taxonomy. */
  private toUpstreamError(error: unknown, context: ProviderCallContext): unknown {
    if (context.signal.aborted || error instanceof UpstreamError) return error;
    const aws = error as {
      name?: string;
      message?: string;
      $metadata?: { httpStatusCode?: number };
    };
    const status = aws.$metadata?.httpStatusCode;
    if (aws.name === 'ThrottlingException') {
      return new UpstreamError(this.id, 'rate_limit', `bedrock throttled: ${aws.message}`, 429);
    }
    if (status !== undefined) {
      return new UpstreamError(
        this.id,
        kindForStatus(status),
        `bedrock returned ${status}: ${aws.message}`,
        status,
      );
    }
    return UpstreamError.network(this.id, error);
  }
}
