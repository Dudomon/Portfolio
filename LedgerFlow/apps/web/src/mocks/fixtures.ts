/**
 * Deterministic demo data for the mock API. A seeded PRNG keeps the dataset
 * identical across reloads and test runs.
 */
import type {
  AuditEvent,
  DocumentDetail,
  DocumentStatus,
  FieldValue,
  JsonValue,
  LineItem,
  Role,
} from '@/api';
import { addDecimals, multiplyDecimals, roundDecimal } from '@/lib/decimal';

export interface MockUser {
  id: string;
  name: string;
  email: string;
  role: Role;
}

export const TENANT_ID = '7d0f6a52-9a8e-4c1b-9f43-2f1d5c0e8a11';

export const USERS: readonly MockUser[] = [
  { id: 'u-admin', name: 'Ana Admin', email: 'admin@demo.test', role: 'admin' },
  { id: 'u-reviewer', name: 'Rafael Reviewer', email: 'reviewer@demo.test', role: 'reviewer' },
  { id: 'u-viewer', name: 'Vera Viewer', email: 'viewer@demo.test', role: 'viewer' },
];

export function mockUuid(namespace: number, index: number): string {
  const tail = index.toString(16).padStart(12, '0');
  return `00000000-0000-4000-8${namespace.toString(16).padStart(3, '0')}-${tail}`;
}

function mulberry32(seed: number) {
  let state = seed;
  return () => {
    state |= 0;
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

interface Supplier {
  name: string;
  taxId: string;
  currency: string;
  nfe: boolean;
}

const SUPPLIERS: readonly Supplier[] = [
  { name: 'Metalúrgica Paulista Ltda', taxId: '12345678000195', currency: 'BRL', nfe: true },
  { name: 'Transportes Rio Verde SA', taxId: '98765432000110', currency: 'BRL', nfe: true },
  { name: 'Papelaria Central ME', taxId: '45678912000133', currency: 'BRL', nfe: false },
  { name: 'Nordwind Logistik GmbH', taxId: 'DE811569869', currency: 'EUR', nfe: false },
  { name: 'Atlas Industrial Supply Inc', taxId: 'US471234567', currency: 'USD', nfe: false },
  { name: 'Química Fina do Sul Ltda', taxId: '33445566000177', currency: 'BRL', nfe: true },
];

const PRODUCTS = [
  'Steel bracket 40mm',
  'Hydraulic hose 1/2"',
  'Freight São Paulo → Curitiba',
  'A4 paper, 500 sheets',
  'Industrial solvent 20L',
  'Pallet rental (monthly)',
  'Maintenance service hours',
  'Safety gloves, pair',
] as const;

const STATUS_CYCLE: readonly DocumentStatus[] = [
  'needs_review',
  'needs_review',
  'approved',
  'needs_review',
  'exported',
  'extracted',
  'needs_review',
  'rejected',
  'processing',
  'approved',
  'failed',
  'queued',
];

const HOUR_MS = 3_600_000;
const BASE_TIME = Date.parse('2026-10-08T18:00:00Z');

export interface MockDocument {
  detail: DocumentDetail;
  audit: AuditEvent[];
}

export function field(
  value: JsonValue,
  confidence: number,
  source: FieldValue['source'] = 'llm',
): FieldValue {
  return { value, confidence, source };
}

function pick<T>(random: () => number, items: readonly T[]): T {
  const item = items[Math.floor(random() * items.length)];
  if (item === undefined) throw new Error('pick() from an empty list');
  return item;
}

function confidence(random: () => number, low: boolean): number {
  const value = low ? 0.35 + random() * 0.45 : 0.86 + random() * 0.13;
  return Math.round(value * 100) / 100;
}

function lineItems(random: () => number): LineItem[] {
  const count = 1 + Math.floor(random() * 4);
  return Array.from({ length: count }, () => {
    const quantity = String(1 + Math.floor(random() * 12));
    const unitPrice = (5 + random() * 400).toFixed(2);
    return {
      description: pick(random, PRODUCTS),
      quantity,
      unit_price: unitPrice,
      total: roundDecimal(multiplyDecimals(quantity, unitPrice), 2),
    };
  });
}

export function buildExtractionFields(
  random: () => number,
  supplier: Supplier,
  index: number,
  createdAt: number,
  needsReview: boolean,
): Record<string, FieldValue> {
  const lines = lineItems(random);
  const net = addDecimals(...lines.map((line) => line.total));
  const tax = roundDecimal(multiplyDecimals(net, supplier.currency === 'BRL' ? '0.18' : '0.19'), 2);
  const total = addDecimals(net, tax);
  const issue = new Date(createdAt - 3 * 24 * HOUR_MS).toISOString().slice(0, 10);
  const due = new Date(createdAt + 27 * 24 * HOUR_MS).toISOString().slice(0, 10);
  // A couple of fields per review document are deliberately uncertain.
  const lowKeys = needsReview
    ? new Set([
        pick(random, ['supplier_tax_id', 'invoice_number', 'due_date']),
        pick(random, ['total_amount', 'tax_amount', 'line_items']),
      ])
    : new Set<string>();
  const c = (key: string) => confidence(random, lowKeys.has(key));

  const fields: Record<string, FieldValue> = {
    supplier_name: field(supplier.name, c('supplier_name')),
    supplier_tax_id: field(supplier.taxId, c('supplier_tax_id'), supplier.nfe ? 'rule' : 'llm'),
    invoice_number: field(
      `${supplier.nfe ? 'NF' : 'INV'}-${(4000 + index).toString()}`,
      c('invoice_number'),
    ),
    issue_date: field(issue, c('issue_date')),
    due_date: field(due, c('due_date')),
    currency: field(supplier.currency, c('currency'), 'rule'),
    total_amount: field(total, c('total_amount')),
    tax_amount: field(tax, c('tax_amount')),
    line_items: field(lines, c('line_items')),
  };
  if (supplier.nfe) {
    const key = Array.from({ length: 44 }, () => Math.floor(random() * 10)).join('');
    fields.nfe_access_key = field(key, 0.99, 'rule');
  }
  return fields;
}

export function overallConfidence(fields: Record<string, FieldValue>): number {
  const values = Object.values(fields).map((value) => value.confidence);
  return values.length === 0 ? 0 : Math.min(...values);
}

export function createFixtures(count = 36): MockDocument[] {
  const random = mulberry32(42);
  return Array.from({ length: count }, (_, index) => {
    const supplier = pick(random, SUPPLIERS);
    const status = STATUS_CYCLE[index % STATUS_CYCLE.length] ?? 'needs_review';
    const createdAt = BASE_TIME - index * 5 * HOUR_MS;
    const uploader = USERS[index % 2 === 0 ? 1 : 0] ?? USERS[0]!;
    const id = mockUuid(1, index + 1);
    const docType = supplier.nfe ? 'nfe' : 'invoice';
    const extracted = !['queued', 'processing', 'failed', 'uploaded'].includes(status);
    const fields = extracted
      ? buildExtractionFields(random, supplier, index, createdAt, status === 'needs_review')
      : null;
    const createdIso = new Date(createdAt).toISOString();
    const updatedIso = new Date(createdAt + HOUR_MS).toISOString();
    const filename = supplier.nfe
      ? `nfe-${(35_000 + index).toString()}.xml`
      : `invoice-${(4000 + index).toString()}.pdf`;

    const detail: DocumentDetail = {
      id,
      filename,
      doc_type: extracted ? docType : null,
      status,
      supplier_name: fields ? supplier.name : null,
      total_amount: fields ? (fields.total_amount?.value as string) : null,
      currency: fields ? supplier.currency : null,
      overall_confidence: fields ? overallConfidence(fields) : null,
      uploaded_by: { id: uploader.id, name: uploader.name },
      created_at: createdIso,
      updated_at: updatedIso,
      content_type: supplier.nfe ? 'application/xml' : 'application/pdf',
      sha256: id.replace(/-/g, '').padEnd(64, '0'),
      size_bytes: 40_000 + Math.floor(random() * 900_000),
      extraction: fields
        ? {
            id: mockUuid(2, index + 1),
            version: 1,
            model: supplier.nfe ? null : 'gpt-4o-mini-2024-07-18',
            prompt_version: supplier.nfe ? null : 'invoice-v3',
            fields,
            overall_confidence: overallConfidence(fields),
            cost_usd_micros: supplier.nfe ? 0 : 300 + Math.floor(random() * 500),
            latency_ms: supplier.nfe
              ? 40 + Math.floor(random() * 60)
              : 1200 + Math.floor(random() * 2400),
            created_at: updatedIso,
          }
        : null,
      rejection_reason: status === 'rejected' ? 'Duplicate of an invoice already paid.' : null,
      version: 1,
    };

    const audit: AuditEvent[] = [
      {
        id: mockUuid(3, index * 10 + 1),
        actor: { id: uploader.id, name: uploader.name },
        action: 'document.uploaded',
        diff: null,
        request_id: mockUuid(4, index * 10 + 1),
        created_at: createdIso,
      },
    ];
    if (fields) {
      audit.push({
        id: mockUuid(3, index * 10 + 2),
        actor: null,
        action: 'document.extracted',
        diff: null,
        request_id: null,
        created_at: updatedIso,
      });
    }
    if (status === 'approved' || status === 'exported' || status === 'rejected') {
      const reviewer = USERS[1]!;
      audit.push({
        id: mockUuid(3, index * 10 + 4),
        actor: { id: reviewer.id, name: reviewer.name },
        action: status === 'rejected' ? 'document.rejected' : 'document.approved',
        diff: null,
        request_id: mockUuid(4, index * 10 + 4),
        created_at: updatedIso,
      });
    }
    if (status === 'failed') {
      audit.push({
        id: mockUuid(3, index * 10 + 3),
        actor: null,
        action: 'document.failed',
        diff: null,
        request_id: null,
        created_at: updatedIso,
      });
    }
    return { detail, audit };
  });
}
