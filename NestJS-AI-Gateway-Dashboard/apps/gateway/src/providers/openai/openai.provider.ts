import type { ChatCompletionRequest } from '@aigw/shared';
import { OpenAiCompatibleProvider, openAiBody } from './openai-compatible.provider';

export class OpenAiProvider extends OpenAiCompatibleProvider {
  readonly id = 'openai' as const;

  constructor(
    private readonly apiKey: string,
    private readonly baseUrl: string,
  ) {
    super();
  }

  protected url(): string {
    return `${this.baseUrl.replace(/\/+$/, '')}/chat/completions`;
  }

  protected headers(): Record<string, string> {
    return { authorization: `Bearer ${this.apiKey}` };
  }

  protected body(request: ChatCompletionRequest): Record<string, unknown> {
    return openAiBody(request);
  }
}
