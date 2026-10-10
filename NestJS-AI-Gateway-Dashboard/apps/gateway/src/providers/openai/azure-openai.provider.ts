import type { ChatCompletionRequest } from '@aigw/shared';
import { OpenAiCompatibleProvider, openAiBody } from './openai-compatible.provider';

/**
 * Azure OpenAI uses the OpenAI wire format but addresses models by
 * deployment name in the URL and authenticates with an `api-key` header.
 * Route targets for this provider therefore name deployments, not models.
 */
export class AzureOpenAiProvider extends OpenAiCompatibleProvider {
  readonly id = 'azure-openai' as const;

  constructor(
    private readonly apiKey: string,
    private readonly endpoint: string,
    private readonly apiVersion: string,
  ) {
    super();
  }

  protected url(deployment: string): string {
    const base = this.endpoint.replace(/\/+$/, '');
    const query = new URLSearchParams({ 'api-version': this.apiVersion });
    return `${base}/openai/deployments/${encodeURIComponent(deployment)}/chat/completions?${query.toString()}`;
  }

  protected headers(): Record<string, string> {
    return { 'api-key': this.apiKey };
  }

  protected body(request: ChatCompletionRequest): Record<string, unknown> {
    const { model: _deployment, ...rest } = openAiBody(request);
    return rest;
  }
}
