# LedgerFlow — Service Contracts

Single source of truth for the boundaries between `apps/web`, `services/api` and `services/worker`.
Any change here requires matching changes in every consumer.

## Domain

LedgerFlow replaces manual invoice keying into a legacy ERP. Users upload supplier invoices
(PDF/image or Brazilian NF-e XML). A worker extracts structured fields, low-confidence results
land in a human review queue, approved documents are exported to the ERP.

### Roles (RBAC)

| Role       | Can |
|------------|-----|
| `admin`    | everything, incl. tenant settings and audit log |
| `reviewer` | upload, view, correct extraction, approve / reject |
| `viewer`   | read-only |

All data is tenant-scoped. `tenant_id` comes from the JWT, never from the request body.

### Document lifecycle

```
uploaded ──► queued ──► processing ──► extracted ──► approved ──► exported
                            │              │
                            │              └──► needs_review ──► approved | rejected
                            └──► failed (after max retries; can be re-queued by admin)
```

`extracted` = all fields confidence >= tenant threshold (default 0.85) → auto-approvable.
Anything lower → `needs_review`.

### Extracted field set (`doc_type = "invoice" | "nfe"`)

| key               | type            | notes |
|-------------------|-----------------|-------|
| `supplier_name`   | string          | |
| `supplier_tax_id` | string          | CNPJ (14 digits) or foreign VAT id |
| `invoice_number`  | string          | |
| `issue_date`      | ISO date        | |
| `due_date`        | ISO date / null | |
| `currency`        | ISO 4217        | |
| `total_amount`    | decimal string  | never float; e.g. `"1530.40"` |
| `tax_amount`      | decimal string  | |
| `nfe_access_key`  | string / null   | 44 digits, NF-e only |
| `line_items`      | array           | `{description, quantity, unit_price, total}` decimal strings |

Each field is wrapped: `{ "value": <...>, "confidence": 0..1, "source": "llm" | "rule" | "human" }`.

## REST API (`/api/v1`, JSON, FastAPI)

Auth: `Authorization: Bearer <JWT>` (HS256 in dev, RS256/JWKS in prod). Claims: `sub`, `tid` (tenant), `role`, `exp`.
Errors: RFC 9457 problem+json `{type, title, status, detail, instance, request_id}`.
Pagination: cursor-based `?limit=50&cursor=<opaque>` → `{ items, next_cursor }`.
Concurrency: `ETag` on document reads; mutating document endpoints require `If-Match` → `412` on mismatch.
Every response carries `X-Request-ID`.

| Method | Path | Role | Body / Query | Response |
|---|---|---|---|---|
| POST | `/auth/token` | — | `{email, password}` | `{access_token, token_type, expires_in}` |
| GET | `/me` | any | | `User` |
| GET | `/documents` | any | `status`, `doc_type`, `q`, `min_confidence`, `limit`, `cursor`, `sort=-created_at` | `Page<DocumentSummary>` |
| POST | `/documents` | reviewer+ | multipart `file` (≤ 15 MB; pdf, png, jpeg, xml) | `202 DocumentSummary` |
| GET | `/documents/{id}` | any | | `DocumentDetail` + `ETag` |
| GET | `/documents/{id}/file` | any | | `307` to presigned S3 URL (5 min) |
| PATCH | `/documents/{id}/fields` | reviewer+ | `{fields: {key: value}}` + `If-Match` | `DocumentDetail` (creates new extraction version, `source=human`) |
| POST | `/documents/{id}/approve` | reviewer+ | `If-Match` | `DocumentDetail` |
| POST | `/documents/{id}/reject` | reviewer+ | `{reason}` + `If-Match` | `DocumentDetail` |
| POST | `/documents/{id}/retry` | admin | | `202` |
| GET | `/documents/{id}/audit` | any | | `AuditEvent[]` |
| GET | `/stats` | any | `from`, `to` | `Stats` |
| POST | `/internal/extractions` | service token | `ExtractionResult` | `204` (idempotent on `event_id`) |
| POST | `/internal/documents/{id}/status` | service token | `{status, error?}` | `204` |

### Schemas

```ts
type DocumentStatus = "uploaded"|"queued"|"processing"|"extracted"|"needs_review"|"approved"|"rejected"|"exported"|"failed";

interface DocumentSummary {
  id: string;               // uuid
  filename: string;
  doc_type: "invoice" | "nfe" | null;
  status: DocumentStatus;
  supplier_name: string | null;
  total_amount: string | null;
  currency: string | null;
  overall_confidence: number | null;
  uploaded_by: { id: string; name: string };
  created_at: string;       // ISO-8601 UTC
  updated_at: string;
}

interface FieldValue { value: unknown; confidence: number; source: "llm" | "rule" | "human"; }

interface Extraction {
  id: string;
  version: number;
  model: string | null;            // e.g. "gpt-4o-mini-2024-07-18", null for rule-based
  prompt_version: string | null;
  fields: Record<string, FieldValue>;
  overall_confidence: number;
  cost_usd_micros: number;
  latency_ms: number;
  created_at: string;
}

interface DocumentDetail extends DocumentSummary {
  content_type: string;
  sha256: string;
  size_bytes: number;
  extraction: Extraction | null;   // latest version
  rejection_reason: string | null;
  version: number;                 // optimistic lock; ETag = W/"<version>"
}

interface AuditEvent {
  id: string; actor: { id: string; name: string } | null;  // null = system
  action: string;  // "document.uploaded" | "document.fields_corrected" | "document.approved" | ...
  diff: Record<string, { before: unknown; after: unknown }> | null;
  request_id: string | null; created_at: string;
}

interface Stats {
  by_status: Record<DocumentStatus, number>;
  auto_approval_rate: number;     // 0..1
  avg_confidence: number;
  p95_processing_ms: number;
  llm_cost_usd: string;           // decimal string
}
```

## Messaging (RabbitMQ)

API never publishes directly from the request transaction: it writes to an `outbox` table in the
same DB transaction and a relay publishes (at-least-once). Consumers must be idempotent on `event_id`.

- Exchange `ledgerflow.events` (topic, durable)
- Queue `worker.document-uploaded` bound to `document.uploaded`, `x-dead-letter-exchange=ledgerflow.dlx`
- Retry: `worker.document-uploaded.retry` with per-message TTL (exponential: 5s, 30s, 2m), max 3 attempts → DLQ `worker.document-uploaded.dlq`

```json
// routing key: document.uploaded
{
  "event_id": "uuid",
  "event_type": "document.uploaded",
  "occurred_at": "2026-10-09T12:00:00Z",
  "tenant_id": "uuid",
  "document_id": "uuid",
  "storage_key": "tenants/<tid>/documents/<id>/<sha256>",
  "content_type": "application/pdf",
  "sha256": "hex",
  "attempt": 1,
  "trace_parent": "00-...-...-01"
}
```

### `ExtractionResult` (worker → `POST /internal/extractions`)

```json
{
  "event_id": "uuid (same as triggering event)",
  "document_id": "uuid",
  "tenant_id": "uuid",
  "doc_type": "nfe",
  "model": "gpt-4o-mini-2024-07-18 | null",
  "prompt_version": "invoice-v3 | null",
  "fields": { "<key>": { "value": "...", "confidence": 0.97, "source": "llm" } },
  "cost_usd_micros": 412,
  "latency_ms": 1830
}
```

## Storage

S3-compatible (MinIO locally). Key `tenants/{tenant_id}/documents/{document_id}/{sha256}`.
Server-side encryption on. Uploads deduplicated per tenant by `sha256` (returns existing document, `200`).

## Database (PostgreSQL 16) — owned by `services/api`

`tenants`, `users`, `documents`, `extractions` (append-only, versioned), `audit_events` (append-only),
`outbox`, `processed_events` (idempotency). Every tenant-owned table has `tenant_id` + composite index;
Row-Level Security policies enforce `tenant_id = current_setting('app.tenant_id')`.
