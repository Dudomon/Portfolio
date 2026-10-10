import type {
  AggregateDimension,
  AggregateRow,
  ApiKeyView,
  CreateApiKeyRequest,
  CreateApiKeyResponse,
  DateRangeQuery,
  ProviderStats,
  RequestLogFilters,
  RequestLogPage,
  RouteView,
  TimeseriesPoint,
  UpdateApiKeyRequest,
  UsageSummary,
} from '@aigw/shared';
import { HttpClient, type HttpOptions } from './http';

export interface GatewayAdminClientOptions extends Omit<HttpOptions, 'token'> {
  adminToken: string;
}

/** Client for the gateway's admin API (usage analytics and key management). */
export class GatewayAdminClient {
  private readonly http: HttpClient;

  constructor({ adminToken, ...options }: GatewayAdminClientOptions) {
    this.http = new HttpClient({ ...options, token: adminToken });
  }

  usageSummary(range: DateRangeQuery = {}): Promise<UsageSummary> {
    return this.get('/admin/usage/summary', { ...range });
  }

  usageTimeseries(range: DateRangeQuery = {}): Promise<TimeseriesPoint[]> {
    return this.get('/admin/usage/timeseries', { ...range });
  }

  providerStats(range: DateRangeQuery = {}): Promise<ProviderStats[]> {
    return this.get('/admin/usage/providers', { ...range });
  }

  usageAggregate(groupBy: AggregateDimension, range: DateRangeQuery = {}): Promise<AggregateRow[]> {
    return this.get('/admin/usage/aggregate', { groupBy, ...range });
  }

  listRequests(filters: RequestLogFilters = {}): Promise<RequestLogPage> {
    return this.get('/admin/requests', { ...filters });
  }

  listRoutes(): Promise<RouteView[]> {
    return this.get('/admin/routes');
  }

  listKeys(): Promise<ApiKeyView[]> {
    return this.get('/admin/keys');
  }

  createKey(body: CreateApiKeyRequest): Promise<CreateApiKeyResponse> {
    return this.http.json({ method: 'POST', path: '/admin/keys', body, retry: false });
  }

  updateKey(id: string, body: UpdateApiKeyRequest): Promise<ApiKeyView> {
    return this.http.json({
      method: 'PATCH',
      path: `/admin/keys/${encodeURIComponent(id)}`,
      body,
      retry: false,
    });
  }

  /** Revocation is idempotent, so it is safe to retry. */
  revokeKey(id: string): Promise<ApiKeyView> {
    return this.http.json({ method: 'DELETE', path: `/admin/keys/${encodeURIComponent(id)}` });
  }

  private get<T>(path: string, query?: Record<string, string | number | undefined>): Promise<T> {
    return this.http.json<T>({ method: 'GET', path, ...(query ? { query } : {}) });
  }
}
