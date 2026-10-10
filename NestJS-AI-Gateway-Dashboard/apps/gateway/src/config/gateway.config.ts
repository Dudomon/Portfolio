import type { z } from 'zod';
import type { gatewayConfigSchema } from '@aigw/shared';

/**
 * Model aliases, fallback chains and prices.
 *
 * Clients only ever see aliases ("fast", "smart", "cheap"), so upstream models
 * can be swapped or reordered here without client changes. Targets whose
 * provider has no credentials are skipped at runtime.
 *
 * Prices are USD per million tokens and must be kept in sync with each
 * provider's published pricing; they drive both cost reporting and budget
 * enforcement. Every route target must have a price (validated at startup).
 */
export const gatewayConfig = {
  routes: {
    fast: {
      description: 'Low-latency small models for interactive use.',
      targets: [
        { provider: 'openai', model: 'gpt-4o-mini' },
        { provider: 'anthropic', model: 'claude-haiku-4-5' },
        { provider: 'bedrock', model: 'us.anthropic.claude-haiku-4-5-20251001-v1:0' },
      ],
    },
    smart: {
      description: 'Highest-quality models for complex reasoning and tool use.',
      targets: [
        { provider: 'anthropic', model: 'claude-sonnet-4-5' },
        { provider: 'openai', model: 'gpt-4o' },
        { provider: 'azure-openai', model: 'gpt-4o' },
      ],
    },
    cheap: {
      description: 'Lowest cost per token for batch and background jobs.',
      targets: [
        { provider: 'azure-openai', model: 'gpt-4o-mini' },
        { provider: 'openai', model: 'gpt-4o-mini' },
      ],
    },
  },
  prices: {
    openai: {
      'gpt-4o-mini': { inputPerMillionUsd: 0.15, outputPerMillionUsd: 0.6 },
      'gpt-4o': { inputPerMillionUsd: 2.5, outputPerMillionUsd: 10 },
    },
    'azure-openai': {
      'gpt-4o-mini': { inputPerMillionUsd: 0.15, outputPerMillionUsd: 0.6 },
      'gpt-4o': { inputPerMillionUsd: 2.5, outputPerMillionUsd: 10 },
    },
    anthropic: {
      'claude-haiku-4-5': { inputPerMillionUsd: 1, outputPerMillionUsd: 5 },
      'claude-sonnet-4-5': { inputPerMillionUsd: 3, outputPerMillionUsd: 15 },
    },
    bedrock: {
      'us.anthropic.claude-haiku-4-5-20251001-v1:0': {
        inputPerMillionUsd: 1,
        outputPerMillionUsd: 5,
      },
    },
  },
  routing: {
    attemptTimeoutMs: 60_000,
    maxRetriesPerTarget: 1,
    backoff: { baseMs: 200, maxMs: 2_000 },
    circuitBreaker: { failureThreshold: 5, cooldownMs: 30_000 },
  },
  defaultMaxTokens: 4_096,
} satisfies z.input<typeof gatewayConfigSchema>;
