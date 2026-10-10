import { Logger, Module } from '@nestjs/common';
import type { GatewayConfig } from '@aigw/shared';
import { ENV, GATEWAY_CONFIG } from '../config/config.module';
import type { AppEnv } from '../config/env';
import { AnthropicProvider } from './anthropic/anthropic.provider';
import { BedrockProvider } from './bedrock/bedrock.provider';
import { LLM_PROVIDERS, type LlmProvider } from './llm-provider';
import { AzureOpenAiProvider } from './openai/azure-openai.provider';
import { OpenAiProvider } from './openai/openai.provider';

/**
 * Instantiates an adapter for every provider that has credentials. Route
 * targets pointing at unconfigured providers are skipped by the router, which
 * lets one config file serve environments with different provider access.
 */
export function createProviders(env: AppEnv, config: GatewayConfig): LlmProvider[] {
  const providers: LlmProvider[] = [];
  if (env.OPENAI_API_KEY) {
    providers.push(new OpenAiProvider(env.OPENAI_API_KEY, env.OPENAI_BASE_URL));
  }
  if (env.ANTHROPIC_API_KEY) {
    providers.push(
      new AnthropicProvider(env.ANTHROPIC_API_KEY, env.ANTHROPIC_BASE_URL, config.defaultMaxTokens),
    );
  }
  if (env.AZURE_OPENAI_ENDPOINT && env.AZURE_OPENAI_API_KEY) {
    providers.push(
      new AzureOpenAiProvider(
        env.AZURE_OPENAI_API_KEY,
        env.AZURE_OPENAI_ENDPOINT,
        env.AZURE_OPENAI_API_VERSION,
      ),
    );
  }
  if (env.BEDROCK_REGION) {
    providers.push(new BedrockProvider(env.BEDROCK_REGION, config.defaultMaxTokens));
  }
  return providers;
}

@Module({
  providers: [
    {
      provide: LLM_PROVIDERS,
      inject: [ENV, GATEWAY_CONFIG],
      useFactory: (env: AppEnv, config: GatewayConfig): LlmProvider[] => {
        const providers = createProviders(env, config);
        const logger = new Logger('Providers');
        if (providers.length === 0) {
          logger.warn('No LLM providers configured; every completion request will fail with 503');
        } else {
          logger.log(`Enabled providers: ${providers.map((p) => p.id).join(', ')}`);
        }
        return providers;
      },
    },
  ],
  exports: [LLM_PROVIDERS],
})
export class ProvidersModule {}
