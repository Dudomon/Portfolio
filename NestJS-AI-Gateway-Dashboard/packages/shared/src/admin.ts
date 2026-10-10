import { z } from 'zod';
import { providerIdSchema } from './config';

/** Contracts for the admin API consumed by the dashboard (via the SDK). */

const isoDate = z.iso.date();

export const dateRangeQuerySchema = z
  .object({
    from: isoDate.optional(),
    to: isoDate.optional(),
  })
  .refine((q) => !q.from || !q.to || q.from <= q.to, {
    message: '`from` must not be after `to`',
    path: ['from'],
  });
export type DateRangeQuery = z.infer<typeof dateRangeQuerySchema>;

export const aggregateDimensionSchema = z.enum(['key', 'model', 'day']);
export type AggregateDimension = z.infer<typeof aggregateDimensionSchema>;

export const aggregateQuerySchema = z.object({
  groupBy: aggregateDimensionSchema,
  from: isoDate.optional(),
  to: isoDate.optional(),
});
export type AggregateQuery = z.infer<typeof aggregateQuerySchema>;

export interface UsageSummary {
  requests: number;
  errors: number;
  errorRate: number;
  cacheHits: number;
  costUsd: number;
  promptTokens: number;
  completionTokens: number;
  p95LatencyMs: number | null;
}

export interface TimeseriesPoint {
  /** UTC day, YYYY-MM-DD. */
  date: string;
  requests: number;
  errors: number;
  costUsd: number;
  p95LatencyMs: number | null;
}

export interface ProviderStats {
  provider: string;
  requests: number;
  errors: number;
  errorRate: number;
  costUsd: number;
  p95LatencyMs: number | null;
}

export interface AggregateRow {
  /** Key ID, `provider/model`, or YYYY-MM-DD depending on the dimension. */
  group: string;
  label: string;
  requests: number;
  costUsd: number;
  promptTokens: number;
  completionTokens: number;
}

export interface ApiKeyView {
  id: string;
  name: string;
  project: string;
  prefix: string;
  monthlyBudgetUsd: number | null;
  rateLimitPerMinute: number;
  monthToDateSpendUsd: number;
  createdAt: string;
  revokedAt: string | null;
}

export const createApiKeyRequestSchema = z.object({
  name: z.string().trim().min(1).max(100),
  project: z.string().trim().min(1).max(100),
  monthlyBudgetUsd: z.number().positive().max(1_000_000).nullable().optional(),
  rateLimitPerMinute: z.number().int().min(1).max(100_000).default(60),
});
export type CreateApiKeyRequest = z.input<typeof createApiKeyRequestSchema>;

export const updateApiKeyRequestSchema = z
  .object({
    name: z.string().trim().min(1).max(100).optional(),
    monthlyBudgetUsd: z.number().positive().max(1_000_000).nullable().optional(),
    rateLimitPerMinute: z.number().int().min(1).max(100_000).optional(),
  })
  .refine((body) => Object.keys(body).length > 0, { message: 'Nothing to update' });
export type UpdateApiKeyRequest = z.infer<typeof updateApiKeyRequestSchema>;

export interface CreateApiKeyResponse {
  key: ApiKeyView;
  /** Plaintext key. Returned exactly once; only its hash is stored. */
  secret: string;
}

export type CircuitState = 'closed' | 'open' | 'half_open';

export interface RouteTargetView {
  provider: string;
  model: string;
  /** False when the provider has no credentials configured. */
  enabled: boolean;
  circuit: CircuitState;
  inputPerMillionUsd: number | null;
  outputPerMillionUsd: number | null;
}

export interface RouteView {
  alias: string;
  description: string;
  targets: RouteTargetView[];
}

export const requestStatusSchema = z.enum(['success', 'error']);
export type RequestStatus = z.infer<typeof requestStatusSchema>;

export const requestLogQuerySchema = z.object({
  apiKeyId: z.string().min(1).optional(),
  provider: providerIdSchema.optional(),
  alias: z.string().min(1).optional(),
  status: requestStatusSchema.optional(),
  from: isoDate.optional(),
  to: isoDate.optional(),
  cursor: z.string().min(1).optional(),
  limit: z.coerce.number().int().min(1).max(200).default(50),
});

/** Client-side view of the request log filters (query params are strings on the wire). */
export interface RequestLogFilters {
  apiKeyId?: string;
  provider?: string;
  alias?: string;
  status?: RequestStatus;
  from?: string;
  to?: string;
  cursor?: string;
  limit?: number;
}

export interface RequestLogEntry {
  id: string;
  requestId: string;
  createdAt: string;
  apiKeyId: string;
  keyName: string;
  alias: string;
  provider: string | null;
  model: string | null;
  status: RequestStatus;
  errorCode: string | null;
  promptTokens: number;
  completionTokens: number;
  costUsd: number;
  latencyMs: number;
  cached: boolean;
  streamed: boolean;
  attempts: number;
}

export interface RequestLogPage {
  items: RequestLogEntry[];
  nextCursor: string | null;
}
