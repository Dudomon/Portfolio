import type { GatewayConfig, ModelPrice, ProviderId, Usage } from '@aigw/shared';

/**
 * USD cost of one request. Rounded to 8 decimal places to match the
 * database column; floating-point error at that scale is far below a cent
 * even when summed over millions of requests.
 */
export function calculateCost(price: ModelPrice | undefined, usage: Usage | undefined): number {
  if (!price || !usage) return 0;
  const cost =
    (usage.prompt_tokens * price.inputPerMillionUsd +
      usage.completion_tokens * price.outputPerMillionUsd) /
    1_000_000;
  return Math.round(cost * 1e8) / 1e8;
}

export function priceFor(
  config: GatewayConfig,
  provider: ProviderId,
  model: string,
): ModelPrice | undefined {
  return config.prices[provider]?.[model];
}
