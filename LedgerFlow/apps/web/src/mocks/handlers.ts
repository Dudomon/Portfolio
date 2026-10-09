/**
 * MSW handlers implementing the REST contract (docs/CONTRACTS.md) against an
 * in-memory store. They back `npm run dev:mocks`, Storybook-free demos and tests.
 */
import { delay, http, HttpResponse, type DefaultBodyType, type StrictRequest } from 'msw';

import {
  DOCUMENT_STATUSES,
  etagForVersion,
  type AuditEvent,
  type DocumentDetail,
  type DocumentStatus,
  type DocumentSummary,
  type FieldValue,
  type JsonValue,
  type Role,
  type Stats,
  type TokenClaims,
} from '@/api';
import { toDecimalString } from '@/lib/decimal';

import { db } from './db';
import { renderInvoicePdf, renderNfeXml } from './files';
import {
  TENANT_ID,
  USERS,
  buildExtractionFields,
  overallConfidence,
  type MockUser,
} from './fixtures';

const API = '*/api/v1';
const TOKEN_TTL_SECONDS = 3600;
const MAX_UPLOAD_BYTES = 15 * 1024 * 1024;
const UPLOAD_TYPES = new Set([
  'application/pdf',
  'image/png',
  'image/jpeg',
  'application/xml',
  'text/xml',
]);
const REVIEWABLE = new Set<DocumentStatus>(['needs_review', 'extracted']);
const ROLE_RANK: Record<Role, number> = { viewer: 1, reviewer: 2, admin: 3 };

/** Artificial latency in the browser so loading states are visible; none in tests. */
const latency = () => (import.meta.env.MODE === 'test' ? Promise.resolve() : delay(250));

type Request = StrictRequest<DefaultBodyType>;

function requestId(request: Request): string {
  return request.headers.get('X-Request-ID') ?? crypto.randomUUID();
}

function problem(request: Request, status: number, title: string, detail?: string) {
  const id = requestId(request);
  return HttpResponse.json(
    {
      type: 'about:blank',
      title,
      status,
      detail: detail ?? null,
      instance: new URL(request.url).pathname,
      request_id: id,
    },
    { status, headers: { 'Content-Type': 'application/problem+json', 'X-Request-ID': id } },
  );
}

function json(request: Request, body: unknown, init: { status?: number; etag?: string } = {}) {
  const headers: Record<string, string> = { 'X-Request-ID': requestId(request) };
  if (init.etag) headers.ETag = init.etag;
  return HttpResponse.json(body as DefaultBodyType, { status: init.status ?? 200, headers });
}

const base64Url = (value: string) =>
  btoa(value).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');

export function createMockToken(user: MockUser, ttlSeconds = TOKEN_TTL_SECONDS): string {
  const claims: TokenClaims = {
    sub: user.id,
    tid: TENANT_ID,
    role: user.role,
    exp: Math.floor(Date.now() / 1000) + ttlSeconds,
  };
  const header = base64Url(JSON.stringify({ alg: 'none', typ: 'JWT' }));
  return `${header}.${base64Url(JSON.stringify(claims))}.mock-signature`;
}

/** Resolves the caller from the bearer token, or returns the error response to send. */
function authenticate(request: Request, minimum: Role = 'viewer'): MockUser | Response {
  const token = request.headers.get('Authorization')?.replace(/^Bearer\s+/i, '');
  const payload = token?.split('.')[1];
  let user: MockUser | undefined;
  if (payload) {
    try {
      const claims = JSON.parse(atob(payload.replace(/-/g, '+').replace(/_/g, '/'))) as TokenClaims;
      if (claims.exp * 1000 > Date.now())
        user = USERS.find((candidate) => candidate.id === claims.sub);
    } catch {
      user = undefined;
    }
  }
  if (!user) return problem(request, 401, 'Not authenticated', 'A valid bearer token is required.');
  if (ROLE_RANK[user.role] < ROLE_RANK[minimum]) {
    return problem(request, 403, 'Forbidden', `This action requires the ${minimum} role.`);
  }
  return user;
}

function toSummary(detail: DocumentDetail): DocumentSummary {
  const { id, filename, doc_type, status, supplier_name, total_amount, currency } = detail;
  return {
    id,
    filename,
    doc_type,
    status,
    supplier_name,
    total_amount,
    currency,
    overall_confidence: detail.overall_confidence,
    uploaded_by: detail.uploaded_by,
    created_at: detail.created_at,
    updated_at: detail.updated_at,
  };
}

function auditEvent(
  request: Request,
  actor: MockUser | null,
  action: string,
  diff: AuditEvent['diff'] = null,
): AuditEvent {
  return {
    id: crypto.randomUUID(),
    actor: actor ? { id: actor.id, name: actor.name } : null,
    action,
    diff,
    request_id: actor ? requestId(request) : null,
    created_at: new Date().toISOString(),
  };
}

/**
 * Shared checks for mutating document endpoints: existence, optimistic lock
 * (428 without If-Match, 412 on mismatch) and lifecycle state.
 */
function guardMutation(request: Request, id: string, { reviewable = true } = {}) {
  const doc = db.find(id);
  if (!doc) return problem(request, 404, 'Document not found');
  const ifMatch = request.headers.get('If-Match');
  if (!ifMatch) return problem(request, 428, 'Precondition required', 'Send the ETag as If-Match.');
  if (ifMatch !== etagForVersion(doc.detail.version)) {
    return problem(
      request,
      412,
      'Precondition failed',
      'The document was modified by someone else.',
    );
  }
  if (reviewable && !REVIEWABLE.has(doc.detail.status)) {
    return problem(request, 409, 'Conflict', `A ${doc.detail.status} document cannot be changed.`);
  }
  return doc;
}

const encodeCursor = (offset: number) => btoa(String(offset));
const decodeCursor = (cursor: string | null) => {
  if (!cursor) return 0;
  const offset = Number(atob(cursor));
  return Number.isInteger(offset) && offset >= 0 ? offset : 0;
};

async function sha256Hex(buffer: ArrayBuffer): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', buffer);
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, '0')).join('');
}

/** Moves a freshly uploaded document through the pipeline, as the worker would. */
function simulateProcessing(id: string, isXml: boolean) {
  setTimeout(() => {
    if (!db.find(id)) return;
    db.update(id, (detail) => ({ ...detail, status: 'processing' }));
  }, 1500);
  setTimeout(() => {
    if (!db.find(id)) return;
    const supplier = isXml
      ? { name: 'Metalúrgica Paulista Ltda', taxId: '12345678000195', currency: 'BRL', nfe: true }
      : { name: 'Nordwind Logistik GmbH', taxId: 'DE811569869', currency: 'EUR', nfe: false };
    const fields = buildExtractionFields(Math.random, supplier, db.count(), Date.now(), true);
    const confidence = overallConfidence(fields);
    db.update(
      id,
      (detail) => ({
        ...detail,
        doc_type: isXml ? 'nfe' : 'invoice',
        status: 'needs_review',
        supplier_name: supplier.name,
        currency: supplier.currency,
        total_amount: fields.total_amount?.value as string,
        overall_confidence: confidence,
        extraction: {
          id: crypto.randomUUID(),
          version: 1,
          model: isXml ? null : 'gpt-4o-mini-2024-07-18',
          prompt_version: isXml ? null : 'invoice-v3',
          fields,
          overall_confidence: confidence,
          cost_usd_micros: isXml ? 0 : 412,
          latency_ms: isXml ? 60 : 1830,
          created_at: new Date().toISOString(),
        },
      }),
      {
        id: crypto.randomUUID(),
        actor: null,
        action: 'document.extracted',
        diff: null,
        request_id: null,
        created_at: new Date().toISOString(),
      },
    );
  }, 4500);
}

export const handlers = [
  http.post(`${API}/auth/token`, async ({ request }) => {
    await latency();
    const body = (await request.json().catch(() => null)) as {
      email?: string;
      password?: string;
    } | null;
    const user = USERS.find((candidate) => candidate.email === body?.email?.toLowerCase());
    if (!user || !body?.password) {
      return problem(request, 401, 'Invalid credentials', 'Email or password is incorrect.');
    }
    return json(request, {
      access_token: createMockToken(user),
      token_type: 'bearer',
      expires_in: TOKEN_TTL_SECONDS,
    });
  }),

  http.get(`${API}/me`, ({ request }) => {
    const user = authenticate(request);
    if (user instanceof Response) return user;
    return json(request, user);
  }),

  http.get(`${API}/documents`, async ({ request }) => {
    const user = authenticate(request);
    if (user instanceof Response) return user;
    await latency();
    const url = new URL(request.url);
    const status = url.searchParams.get('status');
    const docType = url.searchParams.get('doc_type');
    const q = url.searchParams.get('q')?.toLowerCase();
    const minConfidence = Number(url.searchParams.get('min_confidence') ?? 0);
    const ascending = url.searchParams.get('sort') === 'created_at';
    const limit = Math.min(Number(url.searchParams.get('limit') ?? 50) || 50, 100);
    const offset = decodeCursor(url.searchParams.get('cursor'));

    const matches = db
      .all()
      .filter((doc) => !status || doc.status === status)
      .filter((doc) => !docType || doc.doc_type === docType)
      .filter((doc) => !minConfidence || (doc.overall_confidence ?? 0) >= minConfidence)
      .filter((doc) => {
        if (!q) return true;
        const invoiceNumber = doc.extraction?.fields.invoice_number?.value;
        return [
          doc.filename,
          doc.supplier_name,
          typeof invoiceNumber === 'string' ? invoiceNumber : null,
        ].some((text) => text?.toLowerCase().includes(q));
      })
      .sort((a, b) => (ascending ? 1 : -1) * a.created_at.localeCompare(b.created_at));

    const items = matches.slice(offset, offset + limit).map(toSummary);
    const nextOffset = offset + limit;
    return json(request, {
      items,
      next_cursor: nextOffset < matches.length ? encodeCursor(nextOffset) : null,
    });
  }),

  http.post(`${API}/documents`, async ({ request }) => {
    const user = authenticate(request, 'reviewer');
    if (user instanceof Response) return user;
    await latency();
    const form = await request.formData();
    const file = form.get('file');
    if (!(file instanceof File))
      return problem(request, 422, 'Missing file', 'Send a multipart "file" part.');
    if (file.size > MAX_UPLOAD_BYTES)
      return problem(request, 413, 'File too large', 'The limit is 15 MB.');
    const isXml = file.name.toLowerCase().endsWith('.xml') || file.type.includes('xml');
    if (!UPLOAD_TYPES.has(file.type) && !isXml) {
      return problem(request, 415, 'Unsupported file type', 'Upload a PDF, PNG, JPEG or XML file.');
    }

    const sha256 = await sha256Hex(await file.arrayBuffer());
    const existing = db.findBySha(sha256);
    if (existing) return json(request, toSummary(existing.detail));

    const now = new Date().toISOString();
    const detail: DocumentDetail = {
      id: crypto.randomUUID(),
      filename: file.name,
      doc_type: null,
      status: 'queued',
      supplier_name: null,
      total_amount: null,
      currency: null,
      overall_confidence: null,
      uploaded_by: { id: user.id, name: user.name },
      created_at: now,
      updated_at: now,
      content_type: isXml ? 'application/xml' : file.type,
      sha256,
      size_bytes: file.size,
      extraction: null,
      rejection_reason: null,
      version: 1,
    };
    db.insert({ detail, audit: [auditEvent(request, user, 'document.uploaded')] });
    simulateProcessing(detail.id, isXml);
    return json(request, toSummary(detail), { status: 202 });
  }),

  http.get(`${API}/documents/:id`, async ({ request, params }) => {
    const user = authenticate(request);
    if (user instanceof Response) return user;
    await latency();
    const doc = db.find(String(params.id));
    if (!doc) return problem(request, 404, 'Document not found');
    return json(request, doc.detail, { etag: etagForVersion(doc.detail.version) });
  }),

  http.get(`${API}/documents/:id/file`, ({ request, params }) => {
    const user = authenticate(request);
    if (user instanceof Response) return user;
    if (!db.find(String(params.id))) return problem(request, 404, 'Document not found');
    // Contract: 307 to a short-lived presigned object-storage URL.
    return new HttpResponse(null, {
      status: 307,
      headers: {
        Location: `/mock-storage/${String(params.id)}`,
        'X-Request-ID': requestId(request),
      },
    });
  }),

  http.get('*/mock-storage/:id', ({ params }) => {
    const doc = db.find(String(params.id));
    if (!doc) return new HttpResponse(null, { status: 404 });
    if (doc.detail.content_type === 'application/xml') {
      return new HttpResponse(renderNfeXml(doc.detail), {
        headers: { 'Content-Type': 'application/xml' },
      });
    }
    return new HttpResponse(renderInvoicePdf(doc.detail), {
      headers: { 'Content-Type': 'application/pdf' },
    });
  }),

  http.patch(`${API}/documents/:id/fields`, async ({ request, params }) => {
    const user = authenticate(request, 'reviewer');
    if (user instanceof Response) return user;
    await latency();
    const id = String(params.id);
    const doc = guardMutation(request, id);
    if (doc instanceof Response) return doc;
    const body = (await request.json()) as { fields?: Record<string, JsonValue> };
    const corrections = Object.entries(body.fields ?? {});
    if (corrections.length === 0 || !doc.detail.extraction) {
      return problem(request, 422, 'Nothing to correct');
    }

    const previous = doc.detail.extraction;
    const fields: Record<string, FieldValue> = { ...previous.fields };
    const diff: NonNullable<AuditEvent['diff']> = {};
    for (const [key, value] of corrections) {
      diff[key] = { before: previous.fields[key]?.value ?? null, after: value };
      fields[key] = { value, confidence: 1, source: 'human' };
    }
    const confidence = overallConfidence(fields);
    const updated = db.update(
      id,
      (detail) => ({
        ...detail,
        supplier_name:
          typeof fields.supplier_name?.value === 'string' ? fields.supplier_name.value : null,
        total_amount:
          typeof fields.total_amount?.value === 'string' ? fields.total_amount.value : null,
        currency: typeof fields.currency?.value === 'string' ? fields.currency.value : null,
        overall_confidence: confidence,
        extraction: {
          ...previous,
          id: crypto.randomUUID(),
          version: previous.version + 1,
          fields,
          overall_confidence: confidence,
          created_at: new Date().toISOString(),
        },
      }),
      auditEvent(request, user, 'document.fields_corrected', diff),
    );
    return json(request, updated, { etag: etagForVersion(updated.version) });
  }),

  http.post(`${API}/documents/:id/approve`, async ({ request, params }) => {
    const user = authenticate(request, 'reviewer');
    if (user instanceof Response) return user;
    await latency();
    const id = String(params.id);
    const doc = guardMutation(request, id);
    if (doc instanceof Response) return doc;
    const updated = db.update(
      id,
      (detail) => ({ ...detail, status: 'approved' }),
      auditEvent(request, user, 'document.approved', {
        status: { before: doc.detail.status, after: 'approved' },
      }),
    );
    return json(request, updated, { etag: etagForVersion(updated.version) });
  }),

  http.post(`${API}/documents/:id/reject`, async ({ request, params }) => {
    const user = authenticate(request, 'reviewer');
    if (user instanceof Response) return user;
    await latency();
    const id = String(params.id);
    const doc = guardMutation(request, id);
    if (doc instanceof Response) return doc;
    const body = (await request.json().catch(() => ({}))) as { reason?: string };
    const reason = body.reason?.trim();
    if (!reason)
      return problem(request, 422, 'Reason required', 'Explain why the document is rejected.');
    const updated = db.update(
      id,
      (detail) => ({ ...detail, status: 'rejected', rejection_reason: reason }),
      auditEvent(request, user, 'document.rejected', {
        status: { before: doc.detail.status, after: 'rejected' },
      }),
    );
    return json(request, updated, { etag: etagForVersion(updated.version) });
  }),

  http.post(`${API}/documents/:id/retry`, async ({ request, params }) => {
    const user = authenticate(request, 'admin');
    if (user instanceof Response) return user;
    await latency();
    const id = String(params.id);
    const doc = db.find(id);
    if (!doc) return problem(request, 404, 'Document not found');
    if (doc.detail.status !== 'failed')
      return problem(request, 409, 'Only failed documents can be retried');
    db.update(
      id,
      (detail) => ({ ...detail, status: 'queued' }),
      auditEvent(request, user, 'document.retried'),
    );
    simulateProcessing(id, doc.detail.content_type === 'application/xml');
    return new HttpResponse(null, { status: 202, headers: { 'X-Request-ID': requestId(request) } });
  }),

  http.get(`${API}/documents/:id/audit`, async ({ request, params }) => {
    const user = authenticate(request);
    if (user instanceof Response) return user;
    await latency();
    const doc = db.find(String(params.id));
    if (!doc) return problem(request, 404, 'Document not found');
    return json(request, doc.audit);
  }),

  http.get(`${API}/stats`, async ({ request }) => {
    const user = authenticate(request);
    if (user instanceof Response) return user;
    await latency();
    const url = new URL(request.url);
    const from = url.searchParams.get('from') ?? '';
    const to = url.searchParams.get('to');
    const docs = db.all().filter((doc) => doc.created_at >= from && (!to || doc.created_at < to));

    const byStatus = Object.fromEntries(DOCUMENT_STATUSES.map((status) => [status, 0])) as Record<
      DocumentStatus,
      number
    >;
    docs.forEach((doc) => (byStatus[doc.status] += 1));

    const extractions = docs.flatMap((doc) => (doc.extraction ? [doc.extraction] : []));
    const finished = docs.filter((doc) => doc.status === 'approved' || doc.status === 'exported');
    const autoApproved = finished.filter(
      (doc) =>
        !db.find(doc.id)?.audit.some((event) => event.action === 'document.fields_corrected'),
    );
    const latencies = extractions.map((extraction) => extraction.latency_ms).sort((a, b) => a - b);
    const p95 = latencies[Math.max(0, Math.ceil(latencies.length * 0.95) - 1)] ?? 0;
    const costMicros = extractions.reduce(
      (sum, extraction) => sum + BigInt(extraction.cost_usd_micros),
      0n,
    );

    const stats: Stats = {
      by_status: byStatus,
      auto_approval_rate: finished.length === 0 ? 0 : autoApproved.length / finished.length,
      avg_confidence:
        extractions.length === 0
          ? 0
          : extractions.reduce((sum, extraction) => sum + extraction.overall_confidence, 0) /
            extractions.length,
      p95_processing_ms: p95,
      llm_cost_usd: toDecimalString({ coefficient: costMicros, scale: 6 }),
    };
    return json(request, stats);
  }),
];
