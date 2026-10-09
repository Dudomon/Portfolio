import type { ApiClient, UploadOptions } from './client';
import {
  AuditEventListSchema,
  DocumentDetailSchema,
  DocumentPageSchema,
  DocumentSummarySchema,
  StatsSchema,
  TokenResponseSchema,
  UserSchema,
  type DocType,
  type DocumentDetail,
  type DocumentStatus,
  type LineItem,
} from './schemas';

export type DocumentSort = 'created_at' | '-created_at';

export interface DocumentFilters {
  status?: DocumentStatus;
  doc_type?: DocType;
  q?: string;
  min_confidence?: number;
  sort?: DocumentSort;
}

export interface DocumentListParams extends DocumentFilters {
  limit?: number;
  cursor?: string;
}

export interface StatsRange {
  from: string;
  to: string;
}

/** Field values as sent in `PATCH /documents/{id}/fields` (unwrapped, no confidence/source). */
export interface CorrectableFields {
  supplier_name: string;
  supplier_tax_id: string;
  invoice_number: string;
  issue_date: string;
  due_date: string | null;
  currency: string;
  total_amount: string;
  tax_amount: string;
  nfe_access_key: string | null;
  line_items: LineItem[];
}

export type FieldCorrections = Partial<CorrectableFields>;

/** A resource together with the entity tag to send back as `If-Match`. */
export interface Versioned<T> {
  data: T;
  etag: string;
}

/** Contract: `ETag = W/"<version>"`. */
export const etagForVersion = (version: number) => `W/"${version}"`;

function versioned(response: {
  data: DocumentDetail;
  etag: string | null;
}): Versioned<DocumentDetail> {
  // The header is invisible cross-origin unless the API exposes it via CORS;
  // the contract makes it derivable from the body, so fall back to that.
  return { data: response.data, etag: response.etag ?? etagForVersion(response.data.version) };
}

const documentPath = (id: string) => `/documents/${encodeURIComponent(id)}`;

export function createLedgerFlowApi(client: ApiClient) {
  return {
    login: async (credentials: { email: string; password: string }) =>
      (
        await client.request('/auth/token', TokenResponseSchema, {
          method: 'POST',
          json: credentials,
          anonymous: true,
        })
      ).data,

    getMe: async (signal?: AbortSignal) =>
      (await client.request('/me', UserSchema, { signal })).data,

    listDocuments: async (params: DocumentListParams, signal?: AbortSignal) =>
      (
        await client.request('/documents', DocumentPageSchema, {
          query: { ...params },
          signal,
        })
      ).data,

    getDocument: async (id: string, signal?: AbortSignal) =>
      versioned(await client.request(documentPath(id), DocumentDetailSchema, { signal })),

    getDocumentFile: (id: string, signal?: AbortSignal) =>
      client.requestBlob(`${documentPath(id)}/file`, { signal }),

    uploadDocument: async (file: File, options?: UploadOptions) => {
      const body = new FormData();
      body.append('file', file);
      const response = await client.upload('/documents', DocumentSummarySchema, body, options);
      // Contract: re-uploading identical bytes returns the existing document with 200 (not 202).
      return { document: response.data, duplicate: response.status === 200 };
    },

    correctFields: async (id: string, fields: FieldCorrections, etag: string) =>
      versioned(
        await client.request(`${documentPath(id)}/fields`, DocumentDetailSchema, {
          method: 'PATCH',
          json: { fields },
          ifMatch: etag,
        }),
      ),

    approve: async (id: string, etag: string) =>
      versioned(
        await client.request(`${documentPath(id)}/approve`, DocumentDetailSchema, {
          method: 'POST',
          ifMatch: etag,
        }),
      ),

    reject: async (id: string, reason: string, etag: string) =>
      versioned(
        await client.request(`${documentPath(id)}/reject`, DocumentDetailSchema, {
          method: 'POST',
          json: { reason },
          ifMatch: etag,
        }),
      ),

    retry: (id: string) => client.requestVoid(`${documentPath(id)}/retry`, { method: 'POST' }),

    getAudit: async (id: string, signal?: AbortSignal) =>
      (await client.request(`${documentPath(id)}/audit`, AuditEventListSchema, { signal })).data,

    getStats: async (range: StatsRange, signal?: AbortSignal) =>
      (await client.request('/stats', StatsSchema, { query: { ...range }, signal })).data,
  };
}

export type LedgerFlowApi = ReturnType<typeof createLedgerFlowApi>;
