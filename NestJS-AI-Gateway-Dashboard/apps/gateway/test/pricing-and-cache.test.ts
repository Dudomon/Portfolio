import { describe, expect, it } from 'vitest';
import { validateGatewayConfig } from '../src/config/config.module';
import { gatewayConfig } from '../src/config/gateway.config';
import { cacheKey, canonicalJson } from '../src/cache/response-cache.service';
import { calculateCost, priceFor } from '../src/usage/pricing';
import { request, testConfig } from './fakes';

describe('calculateCost', () => {
  const price = { inputPerMillionUsd: 3, outputPerMillionUsd: 15 };

  it('prices input and output tokens separately per million', () => {
    const usage = { prompt_tokens: 1_000, completion_tokens: 500, total_tokens: 1_500 };
    // 1000 * 3/1M + 500 * 15/1M = 0.003 + 0.0075
    expect(calculateCost(price, usage)).toBe(0.0105);
  });

  it('rounds to 8 decimal places', () => {
    const usage = { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 };
    expect(calculateCost({ inputPerMillionUsd: 0.15, outputPerMillionUsd: 0.6 }, usage)).toBe(
      7.5e-7,
    );
  });

  it('is zero without a price or usage', () => {
    expect(
      calculateCost(undefined, { prompt_tokens: 5, completion_tokens: 5, total_tokens: 10 }),
    ).toBe(0);
    expect(calculateCost(price, undefined)).toBe(0);
  });

  it('looks prices up by provider and model', () => {
    expect(priceFor(testConfig(), 'anthropic', 'fallback-model')).toEqual({
      inputPerMillionUsd: 3,
      outputPerMillionUsd: 4,
    });
    expect(priceFor(testConfig(), 'bedrock', 'fallback-model')).toBeUndefined();
  });
});

describe('gateway config validation', () => {
  it('accepts the shipped config', () => {
    expect(() => validateGatewayConfig(gatewayConfig)).not.toThrow();
  });

  it('refuses route targets without a price', () => {
    const config = testConfig();
    config.routes.fast?.targets.push({ provider: 'bedrock', model: 'unpriced' });
    expect(() => validateGatewayConfig(config)).toThrow(/without prices[\s\S]*bedrock\/unpriced/);
  });
});

describe('response cache key', () => {
  it('ignores key order, the user tag and streaming flags', () => {
    const a = cacheKey('p1', { ...request, temperature: 0, user: 'alice' });
    const b = cacheKey('p1', {
      temperature: 0,
      messages: request.messages,
      model: 'fast',
      stream: false,
    });
    expect(a).toBe(b);
  });

  it('is scoped per project and sensitive to content', () => {
    const base = { ...request, temperature: 0 };
    expect(cacheKey('p1', base)).not.toBe(cacheKey('p2', base));
    expect(cacheKey('p1', base)).not.toBe(
      cacheKey('p1', { ...base, messages: [{ role: 'user', content: 'other' }] }),
    );
  });

  it('canonicalizes nested objects', () => {
    expect(canonicalJson({ b: [{ y: 1, x: 2 }], a: null })).toBe('{"a":null,"b":[{"x":2,"y":1}]}');
  });
});
