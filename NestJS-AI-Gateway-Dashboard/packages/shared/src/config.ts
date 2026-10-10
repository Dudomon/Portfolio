import { z } from 'zod';

/**
 * Typed gateway configuration: model aliases, fallback chains, prices and
 * routing policy. Concrete provider model IDs live only in config so that
 * swapping a model is a config change, not a code change.
 */

export const providerIdSchema = z.enum(['openai', 'anthropic', 'azure-openai', 'bedrock']);
export type ProviderId = z.infer<typeof providerIdSchema>;

export const routeTargetSchema = z.object({
  provider: providerIdSchema,
  /** Upstream model ID (Azure: deployment name, Bedrock: model or inference profile ID). */
  model: z.string().min(1),
});
export type RouteTarget = z.infer<typeof routeTargetSchema>;

export const routeSchema = z.object({
  description: z.string(),
  /** Ordered by priority; later targets are fallbacks. */
  targets: z.array(routeTargetSchema).min(1),
});
export type Route = z.infer<typeof routeSchema>;

export const modelPriceSchema = z.object({
  inputPerMillionUsd: z.number().nonnegative(),
  outputPerMillionUsd: z.number().nonnegative(),
});
export type ModelPrice = z.infer<typeof modelPriceSchema>;

export const routingPolicySchema = z.object({
  /** Per-attempt deadline; a slow provider is treated like a failed one. */
  attemptTimeoutMs: z.number().int().positive(),
  /** Retries against the same target before moving down the chain. */
  maxRetriesPerTarget: z.number().int().nonnegative(),
  backoff: z.object({
    baseMs: z.number().int().positive(),
    maxMs: z.number().int().positive(),
  }),
  circuitBreaker: z.object({
    /** Consecutive fallback-eligible failures that open the circuit. */
    failureThreshold: z.number().int().positive(),
    /** Time an open circuit rejects calls before allowing a half-open probe. */
    cooldownMs: z.number().int().positive(),
  }),
});
export type RoutingPolicy = z.infer<typeof routingPolicySchema>;

const aliasSchema = z.string().regex(/^[a-z0-9][a-z0-9._-]{0,63}$/, 'Invalid alias');

export const gatewayConfigSchema = z.object({
  routes: z.record(aliasSchema, routeSchema),
  prices: z.partialRecord(providerIdSchema, z.record(z.string(), modelPriceSchema)),
  routing: routingPolicySchema,
  /** Anthropic and Bedrock require max_tokens; used when the client omits it. */
  defaultMaxTokens: z.number().int().positive(),
});
export type GatewayConfig = z.infer<typeof gatewayConfigSchema>;
