import { Injectable } from '@nestjs/common';
import { Prisma } from '@prisma/client';
import type {
  AggregateDimension,
  AggregateRow,
  DateRangeQuery,
  ProviderStats,
  RequestLogEntry,
  RequestLogPage,
  TimeseriesPoint,
  UsageSummary,
  requestLogQuerySchema,
} from '@aigw/shared';
import type { z } from 'zod';
import { PrismaService } from '../infra/prisma.service';

const DAY_MS = 24 * 60 * 60 * 1000;
const DEFAULT_RANGE_DAYS = 30;

interface Range {
  start: Date;
  /** Exclusive. */
  end: Date;
}

/**
 * Converts inclusive YYYY-MM-DD bounds into a half-open UTC interval,
 * defaulting to the last 30 days.
 */
export function toRange(query: DateRangeQuery, now = new Date()): Range {
  const today = Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate());
  const end = query.to ? Date.parse(`${query.to}T00:00:00Z`) + DAY_MS : today + DAY_MS;
  const start = query.from
    ? Date.parse(`${query.from}T00:00:00Z`)
    : end - DEFAULT_RANGE_DAYS * DAY_MS;
  return { start: new Date(start), end: new Date(end) };
}

/** Postgres returns numerics/bigints in forms JSON cannot carry; normalize them. */
function num(value: unknown): number {
  if (value === null || value === undefined) return 0;
  if (typeof value === 'number') return value;
  if (typeof value === 'bigint' || typeof value === 'string') return Number(value);
  if (value instanceof Prisma.Decimal) return value.toNumber();
  return Number.NaN;
}

function nullableNum(value: unknown): number | null {
  return value === null || value === undefined ? null : Math.round(num(value));
}

/**
 * Read-side queries for the dashboard. Aggregations run in PostgreSQL
 * (percentile_cont, FILTER) instead of loading rows into Node; the indexes
 * on usage_records(created_at, ...) keep them range scans.
 *
 * p95 latency excludes cached responses and failures, which would otherwise
 * drag the percentile toward meaningless near-zero or timeout values.
 */
@Injectable()
export class UsageAnalyticsService {
  constructor(private readonly prisma: PrismaService) {}

  async summary(query: DateRangeQuery): Promise<UsageSummary> {
    const { start, end } = toRange(query);
    const [row] = await this.prisma.$queryRaw<Record<string, unknown>[]>`
      SELECT
        count(*) AS requests,
        count(*) FILTER (WHERE status = 'error') AS errors,
        count(*) FILTER (WHERE cached) AS cache_hits,
        coalesce(sum(cost_usd), 0) AS cost_usd,
        coalesce(sum(prompt_tokens), 0) AS prompt_tokens,
        coalesce(sum(completion_tokens), 0) AS completion_tokens,
        percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms)
          FILTER (WHERE status = 'success' AND NOT cached) AS p95
      FROM usage_records
      WHERE created_at >= ${start} AND created_at < ${end}`;

    const requests = num(row?.requests);
    const errors = num(row?.errors);
    return {
      requests,
      errors,
      errorRate: requests === 0 ? 0 : errors / requests,
      cacheHits: num(row?.cache_hits),
      costUsd: num(row?.cost_usd),
      promptTokens: num(row?.prompt_tokens),
      completionTokens: num(row?.completion_tokens),
      p95LatencyMs: nullableNum(row?.p95),
    };
  }

  async timeseries(query: DateRangeQuery): Promise<TimeseriesPoint[]> {
    const { start, end } = toRange(query);
    const rows = await this.prisma.$queryRaw<Record<string, unknown>[]>`
      SELECT
        to_char(date_trunc('day', created_at), 'YYYY-MM-DD') AS day,
        count(*) AS requests,
        count(*) FILTER (WHERE status = 'error') AS errors,
        coalesce(sum(cost_usd), 0) AS cost_usd,
        percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms)
          FILTER (WHERE status = 'success' AND NOT cached) AS p95
      FROM usage_records
      WHERE created_at >= ${start} AND created_at < ${end}
      GROUP BY 1
      ORDER BY 1`;

    const byDay = new Map(rows.map((row) => [String(row.day), row]));
    // Emit every day in the range so charts show gaps as zeros, not as
    // interpolated lines across missing days.
    const points: TimeseriesPoint[] = [];
    for (let t = start.getTime(); t < end.getTime(); t += DAY_MS) {
      const date = new Date(t).toISOString().slice(0, 10);
      const row = byDay.get(date);
      points.push({
        date,
        requests: num(row?.requests),
        errors: num(row?.errors),
        costUsd: num(row?.cost_usd),
        p95LatencyMs: nullableNum(row?.p95),
      });
    }
    return points;
  }

  async providers(query: DateRangeQuery): Promise<ProviderStats[]> {
    const { start, end } = toRange(query);
    const rows = await this.prisma.$queryRaw<Record<string, unknown>[]>`
      SELECT
        provider,
        count(*) AS requests,
        count(*) FILTER (WHERE status = 'error') AS errors,
        coalesce(sum(cost_usd), 0) AS cost_usd,
        percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms)
          FILTER (WHERE status = 'success') AS p95
      FROM usage_records
      WHERE created_at >= ${start} AND created_at < ${end}
        AND provider IS NOT NULL
      GROUP BY provider
      ORDER BY requests DESC`;

    return rows.map((row) => {
      const requests = num(row.requests);
      const errors = num(row.errors);
      return {
        provider: String(row.provider),
        requests,
        errors,
        errorRate: requests === 0 ? 0 : errors / requests,
        costUsd: num(row.cost_usd),
        p95LatencyMs: nullableNum(row.p95),
      };
    });
  }

  async aggregate(dimension: AggregateDimension, query: DateRangeQuery): Promise<AggregateRow[]> {
    const { start, end } = toRange(query);
    const groupExpression = {
      key: Prisma.sql`u.api_key_id`,
      model: Prisma.sql`coalesce(u.provider || '/' || u.model, 'unrouted')`,
      day: Prisma.sql`to_char(date_trunc('day', u.created_at), 'YYYY-MM-DD')`,
    }[dimension];
    const labelExpression = dimension === 'key' ? Prisma.sql`max(k.name)` : groupExpression;

    const rows = await this.prisma.$queryRaw<Record<string, unknown>[]>`
      SELECT
        ${groupExpression} AS grp,
        ${labelExpression} AS label,
        count(*) AS requests,
        coalesce(sum(u.cost_usd), 0) AS cost_usd,
        coalesce(sum(u.prompt_tokens), 0) AS prompt_tokens,
        coalesce(sum(u.completion_tokens), 0) AS completion_tokens
      FROM usage_records u
      JOIN api_keys k ON k.id = u.api_key_id
      WHERE u.created_at >= ${start} AND u.created_at < ${end}
      GROUP BY 1
      ORDER BY ${dimension === 'day' ? Prisma.sql`1` : Prisma.sql`cost_usd DESC`}`;

    return rows.map((row) => ({
      group: String(row.grp),
      label: String(row.label),
      requests: num(row.requests),
      costUsd: num(row.cost_usd),
      promptTokens: num(row.prompt_tokens),
      completionTokens: num(row.completion_tokens),
    }));
  }

  async requests(query: z.output<typeof requestLogQuerySchema>): Promise<RequestLogPage> {
    const where: Prisma.UsageRecordWhereInput = {
      ...(query.apiKeyId ? { apiKeyId: query.apiKeyId } : {}),
      ...(query.provider ? { provider: query.provider } : {}),
      ...(query.alias ? { alias: query.alias } : {}),
      ...(query.status ? { status: query.status } : {}),
    };
    if (query.from || query.to) {
      const { start, end } = toRange({
        ...(query.from ? { from: query.from } : { from: '1970-01-01' }),
        ...(query.to ? { to: query.to } : {}),
      });
      where.createdAt = { gte: start, lt: end };
    }

    // Keyset pagination on (created_at, id): stable under concurrent inserts,
    // unlike OFFSET, and cheap at any depth.
    const rows = await this.prisma.usageRecord.findMany({
      where,
      include: { apiKey: { select: { name: true } } },
      orderBy: [{ createdAt: 'desc' }, { id: 'desc' }],
      take: query.limit + 1,
      ...(query.cursor ? { cursor: { id: query.cursor }, skip: 1 } : {}),
    });

    const page = rows.slice(0, query.limit);
    const items: RequestLogEntry[] = page.map((row) => ({
      id: row.id,
      requestId: row.requestId,
      createdAt: row.createdAt.toISOString(),
      apiKeyId: row.apiKeyId,
      keyName: row.apiKey.name,
      alias: row.alias,
      provider: row.provider,
      model: row.model,
      status: row.status,
      errorCode: row.errorCode,
      promptTokens: row.promptTokens,
      completionTokens: row.completionTokens,
      costUsd: row.costUsd.toNumber(),
      latencyMs: row.latencyMs,
      cached: row.cached,
      streamed: row.streamed,
      attempts: row.attempts,
    }));
    return {
      items,
      nextCursor: rows.length > query.limit ? (page.at(-1)?.id ?? null) : null,
    };
  }
}
