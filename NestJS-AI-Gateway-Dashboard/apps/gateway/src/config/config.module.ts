import { Global, Module } from '@nestjs/common';
import { gatewayConfigSchema, type GatewayConfig } from '@aigw/shared';
import { parseEnv } from './env';
import { gatewayConfig } from './gateway.config';

export const ENV = Symbol('ENV');
export const GATEWAY_CONFIG = Symbol('GATEWAY_CONFIG');

/**
 * Cross-field checks the schema cannot express. A target without a price
 * would be recorded as free and silently bypass budget enforcement, so the
 * process refuses to start instead.
 */
export function validateGatewayConfig(raw: unknown): GatewayConfig {
  const config = gatewayConfigSchema.parse(raw);
  const missing: string[] = [];
  for (const [alias, route] of Object.entries(config.routes)) {
    for (const target of route.targets) {
      if (!config.prices[target.provider]?.[target.model]) {
        missing.push(`${alias} -> ${target.provider}/${target.model}`);
      }
    }
  }
  if (missing.length > 0) {
    throw new Error(`Gateway config has route targets without prices:\n  ${missing.join('\n  ')}`);
  }
  if (config.routing.backoff.baseMs > config.routing.backoff.maxMs) {
    throw new Error('routing.backoff.baseMs must not exceed routing.backoff.maxMs');
  }
  return config;
}

@Global()
@Module({
  providers: [
    { provide: ENV, useFactory: () => parseEnv(process.env) },
    { provide: GATEWAY_CONFIG, useFactory: () => validateGatewayConfig(gatewayConfig) },
  ],
  exports: [ENV, GATEWAY_CONFIG],
})
export class ConfigModule {}
