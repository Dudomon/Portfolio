/**
 * Runtime schemas for every payload crossing the API boundary, mirroring
 * docs/CONTRACTS.md. Responses are parsed (not cast) so contract drift fails
 * loudly at the edge instead of deep inside a component.
 */
import { z } from 'zod';

import { isDecimalString } from '@/lib/decimal';

export const DOCUMENT_STATUSES = [
  'uploaded',
  'queued',
  'processing',
  'extracted',
  'needs_review',
  'approved',
  'rejected',
  'exported',
  'failed',
] as const;

export const DocumentStatusSchema = z.enum(DOCUMENT_STATUSES);
export type DocumentStatus = z.infer<typeof DocumentStatusSchema>;

export const DocTypeSchema = z.enum(['invoice', 'nfe']);
export type DocType = z.infer<typeof DocTypeSchema>;

export const RoleSchema = z.enum(['admin', 'reviewer', 'viewer']);
export type Role = z.infer<typeof RoleSchema>;

export const FieldSourceSchema = z.enum(['llm', 'rule', 'human']);
export type FieldSource = z.infer<typeof FieldSourceSchema>;

export const DecimalStringSchema = z
  .string()
  .refine(isDecimalString, { message: 'Expected a decimal string' });

const IsoDateTimeSchema = z.string().datetime({ offset: true });
const UuidSchema = z.string().uuid();

export type JsonValue =
  string | number | boolean | null | JsonValue[] | { [key: string]: JsonValue };

/**
 * The contract types `value`, `before` and `after` as `unknown`. `z.unknown()`
 * would infer those keys as optional, so this keeps them required while
 * typing them as what they are on the wire: any JSON value, null included.
 */
const AnyJsonSchema = z.custom<JsonValue>((value) => value !== undefined, {
  message: 'Required',
});

const ActorSchema = z.object({ id: z.string(), name: z.string() });

export const DocumentSummarySchema = z.object({
  id: UuidSchema,
  filename: z.string(),
  doc_type: DocTypeSchema.nullable(),
  status: DocumentStatusSchema,
  supplier_name: z.string().nullable(),
  total_amount: DecimalStringSchema.nullable(),
  currency: z.string().nullable(),
  overall_confidence: z.number().min(0).max(1).nullable(),
  uploaded_by: ActorSchema,
  created_at: IsoDateTimeSchema,
  updated_at: IsoDateTimeSchema,
});
export type DocumentSummary = z.infer<typeof DocumentSummarySchema>;

export const FieldValueSchema = z.object({
  value: AnyJsonSchema,
  confidence: z.number().min(0).max(1),
  source: FieldSourceSchema,
});
export type FieldValue = z.infer<typeof FieldValueSchema>;

export const ExtractionSchema = z.object({
  id: z.string(),
  version: z.number().int(),
  model: z.string().nullable(),
  prompt_version: z.string().nullable(),
  fields: z.record(z.string(), FieldValueSchema),
  overall_confidence: z.number().min(0).max(1),
  cost_usd_micros: z.number().int().nonnegative(),
  latency_ms: z.number().nonnegative(),
  created_at: IsoDateTimeSchema,
});
export type Extraction = z.infer<typeof ExtractionSchema>;

export const DocumentDetailSchema = DocumentSummarySchema.extend({
  content_type: z.string(),
  sha256: z.string(),
  size_bytes: z.number().int().nonnegative(),
  extraction: ExtractionSchema.nullable(),
  rejection_reason: z.string().nullable(),
  version: z.number().int(),
});
export type DocumentDetail = z.infer<typeof DocumentDetailSchema>;

export const AuditEventSchema = z.object({
  id: z.string(),
  actor: ActorSchema.nullable(),
  action: z.string(),
  diff: z.record(z.string(), z.object({ before: AnyJsonSchema, after: AnyJsonSchema })).nullable(),
  request_id: z.string().nullable(),
  created_at: IsoDateTimeSchema,
});
export type AuditEvent = z.infer<typeof AuditEventSchema>;
export const AuditEventListSchema = z.array(AuditEventSchema);

const countSchema = z.number().int().nonnegative();

export const StatsSchema = z.object({
  by_status: z.object({
    uploaded: countSchema,
    queued: countSchema,
    processing: countSchema,
    extracted: countSchema,
    needs_review: countSchema,
    approved: countSchema,
    rejected: countSchema,
    exported: countSchema,
    failed: countSchema,
  }) satisfies z.ZodType<Record<DocumentStatus, number>>,
  auto_approval_rate: z.number().min(0).max(1),
  avg_confidence: z.number().min(0).max(1),
  p95_processing_ms: z.number().nonnegative(),
  llm_cost_usd: DecimalStringSchema,
});
export type Stats = z.infer<typeof StatsSchema>;

export function pageSchema<T extends z.ZodTypeAny>(item: T) {
  return z.object({
    items: z.array(item),
    next_cursor: z.string().nullable(),
  });
}
export interface Page<T> {
  items: T[];
  next_cursor: string | null;
}
export const DocumentPageSchema = pageSchema(DocumentSummarySchema);

export const TokenResponseSchema = z.object({
  access_token: z.string().min(1),
  token_type: z.string(),
  expires_in: z.number().int().positive(),
});
export type TokenResponse = z.infer<typeof TokenResponseSchema>;

/** JWT claims as specified by the contract (`sub`, `tid`, `role`, `exp`). */
export const TokenClaimsSchema = z.object({
  sub: z.string(),
  tid: z.string(),
  role: RoleSchema,
  exp: z.number().int(),
});
export type TokenClaims = z.infer<typeof TokenClaimsSchema>;

/**
 * The contract names a `User` response for GET /me without defining it. Only
 * the fields the UI renders are required; unknown keys pass through.
 */
export const UserSchema = z
  .object({
    id: z.string(),
    name: z.string(),
    email: z.string(),
    role: RoleSchema,
  })
  .passthrough();
export type User = z.infer<typeof UserSchema>;

export const ProblemDetailsSchema = z
  .object({
    type: z.string().default('about:blank'),
    title: z.string(),
    status: z.number().int(),
    detail: z.string().nullish(),
    instance: z.string().nullish(),
    request_id: z.string().nullish(),
  })
  .passthrough();
export type ProblemDetails = z.infer<typeof ProblemDetailsSchema>;

/** Known extracted field keys (contract: "Extracted field set"). */
export const LineItemSchema = z.object({
  description: z.string(),
  quantity: DecimalStringSchema,
  unit_price: DecimalStringSchema,
  total: DecimalStringSchema,
});
export type LineItem = z.infer<typeof LineItemSchema>;
