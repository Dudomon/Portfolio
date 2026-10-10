import { z } from 'zod';

// `.env` files commonly contain `KEY=` for unused settings; treat blank as unset.
const optionalString = z.preprocess(
  (value) => (typeof value === 'string' && value.trim() === '' ? undefined : value),
  z.string().optional(),
);

const envSchema = z.object({
  NODE_ENV: z.enum(['development', 'production', 'test']).default('development'),
  PORT: z.coerce.number().int().positive().default(3000),
  LOG_LEVEL: z.enum(['fatal', 'error', 'warn', 'info', 'debug', 'trace']).default('info'),
  CORS_ORIGINS: optionalString,

  DATABASE_URL: z.string().min(1),
  REDIS_URL: z.string().min(1),

  ADMIN_TOKEN: z.string().min(32, 'ADMIN_TOKEN must be at least 32 characters'),
  API_KEY_PEPPER: z.string().min(32, 'API_KEY_PEPPER must be at least 32 characters'),

  OPENAI_API_KEY: optionalString,
  OPENAI_BASE_URL: z.url().default('https://api.openai.com/v1'),
  ANTHROPIC_API_KEY: optionalString,
  ANTHROPIC_BASE_URL: z.url().default('https://api.anthropic.com'),
  AZURE_OPENAI_ENDPOINT: optionalString,
  AZURE_OPENAI_API_KEY: optionalString,
  AZURE_OPENAI_API_VERSION: z.string().default('2024-10-21'),
  /** Enables Bedrock; credentials come from the standard AWS provider chain. */
  BEDROCK_REGION: optionalString,

  /** 0 disables the exact-match response cache. */
  RESPONSE_CACHE_TTL_SECONDS: z.coerce.number().int().nonnegative().default(0),
  USAGE_FLUSH_INTERVAL_MS: z.coerce.number().int().positive().default(2_000),
  USAGE_FLUSH_BATCH_SIZE: z.coerce.number().int().positive().default(200),
  USAGE_BUFFER_MAX: z.coerce.number().int().positive().default(10_000),
  /** How long a resolved API key is trusted before re-reading the database. */
  API_KEY_CACHE_TTL_MS: z.coerce.number().int().nonnegative().default(30_000),
});

export type AppEnv = z.infer<typeof envSchema>;

export function parseEnv(source: NodeJS.ProcessEnv): AppEnv {
  const result = envSchema.safeParse(source);
  if (!result.success) {
    const details = result.error.issues
      .map((issue) => `  ${issue.path.join('.')}: ${issue.message}`)
      .join('\n');
    throw new Error(`Invalid environment configuration:\n${details}`);
  }
  return result.data;
}
