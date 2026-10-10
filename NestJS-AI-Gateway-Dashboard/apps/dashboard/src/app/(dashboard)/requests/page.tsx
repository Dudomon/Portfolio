import type { Metadata } from 'next';
import Link from 'next/link';
import { providerIdSchema, type RequestLogFilters } from '@aigw/shared';
import { PageHeader } from '@/components/PageHeader';
import ui from '@/components/ui.module.css';
import { withAdmin } from '@/lib/admin';
import { formatLatency, formatTimestamp, formatUsd } from '@/lib/format';
import styles from './requests.module.css';

export const metadata: Metadata = { title: 'Request log' };

type SearchParams = Record<string, string | string[] | undefined>;

const FILTER_KEYS = ['apiKeyId', 'provider', 'alias', 'status', 'from', 'to'] as const;
const DATE = /^\d{4}-\d{2}-\d{2}$/;

/** Keeps only well-formed filter values so a hand-edited URL cannot cause a 400. */
function parseFilters(params: SearchParams): RequestLogFilters {
  const one = (key: string): string | undefined => {
    const value = params[key];
    const text = Array.isArray(value) ? value[0] : value;
    return text?.trim() || undefined;
  };
  const provider = one('provider');
  const status = one('status');
  const from = one('from');
  const to = one('to');
  return {
    apiKeyId: one('apiKeyId'),
    alias: one('alias'),
    provider: provider && providerIdSchema.safeParse(provider).success ? provider : undefined,
    status: status === 'success' || status === 'error' ? status : undefined,
    from: from && DATE.test(from) ? from : undefined,
    to: to && DATE.test(to) ? to : undefined,
    cursor: one('cursor'),
    limit: 50,
  };
}

function olderHref(filters: RequestLogFilters, cursor: string): string {
  const query = new URLSearchParams();
  for (const key of FILTER_KEYS) {
    const value = filters[key];
    if (value) query.set(key, value);
  }
  query.set('cursor', cursor);
  return `/requests?${query}`;
}

export default async function RequestsPage({
  searchParams,
}: {
  searchParams: Promise<SearchParams>;
}) {
  const filters = parseFilters(await searchParams);
  const [page, keys, routes] = await withAdmin((admin) =>
    Promise.all([admin.listRequests(filters), admin.listKeys(), admin.listRoutes()]),
  );

  return (
    <>
      <PageHeader title="Request log">One row per client request, newest first.</PageHeader>

      <form className={styles.filters} method="get" aria-label="Filter requests">
        <label className={ui.field}>
          API key
          <select className={ui.input} name="apiKeyId" defaultValue={filters.apiKeyId ?? ''}>
            <option value="">All keys</option>
            {keys.map((key) => (
              <option key={key.id} value={key.id}>
                {key.name} ({key.project})
              </option>
            ))}
          </select>
        </label>
        <label className={ui.field}>
          Model alias
          <select className={ui.input} name="alias" defaultValue={filters.alias ?? ''}>
            <option value="">All aliases</option>
            {routes.map((route) => (
              <option key={route.alias} value={route.alias}>
                {route.alias}
              </option>
            ))}
          </select>
        </label>
        <label className={ui.field}>
          Provider
          <select className={ui.input} name="provider" defaultValue={filters.provider ?? ''}>
            <option value="">All providers</option>
            {providerIdSchema.options.map((provider) => (
              <option key={provider} value={provider}>
                {provider}
              </option>
            ))}
          </select>
        </label>
        <label className={ui.field}>
          Status
          <select className={ui.input} name="status" defaultValue={filters.status ?? ''}>
            <option value="">Any status</option>
            <option value="success">Success</option>
            <option value="error">Error</option>
          </select>
        </label>
        <label className={ui.field}>
          From
          <input className={ui.input} type="date" name="from" defaultValue={filters.from ?? ''} />
        </label>
        <label className={ui.field}>
          To
          <input className={ui.input} type="date" name="to" defaultValue={filters.to ?? ''} />
        </label>
        <div className={styles.filterActions}>
          <button className={ui.button} type="submit">
            Apply
          </button>
          <Link className={ui.buttonSecondary} href="/requests">
            Reset
          </Link>
        </div>
      </form>

      <div className={ui.tableWrap}>
        <table className={ui.table}>
          <caption>
            {page.items.length} requests{filters.cursor ? ' (older page)' : ''}
          </caption>
          <thead>
            <tr>
              <th scope="col">Time</th>
              <th scope="col">Key</th>
              <th scope="col">Alias</th>
              <th scope="col">Served by</th>
              <th scope="col">Status</th>
              <th scope="col" className={ui.num}>
                Tokens in / out
              </th>
              <th scope="col" className={ui.num}>
                Cost
              </th>
              <th scope="col" className={ui.num}>
                Latency
              </th>
              <th scope="col" className={ui.num}>
                Attempts
              </th>
              <th scope="col">Request ID</th>
            </tr>
          </thead>
          <tbody>
            {page.items.length === 0 ? (
              <tr>
                <td colSpan={10} className={ui.empty}>
                  No requests match these filters.
                </td>
              </tr>
            ) : (
              page.items.map((item) => (
                <tr key={item.id}>
                  <td>{formatTimestamp(item.createdAt)}</td>
                  <td>{item.keyName}</td>
                  <td className={ui.mono}>{item.alias}</td>
                  <td>
                    {item.cached ? (
                      <span className={ui.badge}>Cache</span>
                    ) : item.provider ? (
                      <>
                        {item.provider}{' '}
                        <span className={`${ui.mono} ${ui.muted}`}>{item.model}</span>
                      </>
                    ) : (
                      <span className={ui.muted}>—</span>
                    )}
                    {item.streamed ? <span className={styles.tag}>stream</span> : null}
                  </td>
                  <td>
                    {item.status === 'success' ? (
                      <span className={ui.badgeGood}>
                        <span aria-hidden="true">●</span> OK
                      </span>
                    ) : (
                      <span className={ui.badgeCritical} title={item.errorCode ?? undefined}>
                        <span aria-hidden="true">▲</span> {item.errorCode ?? 'Error'}
                      </span>
                    )}
                  </td>
                  <td className={ui.num}>
                    {item.promptTokens} / {item.completionTokens}
                  </td>
                  <td className={ui.num}>{formatUsd(item.costUsd)}</td>
                  <td className={ui.num}>{formatLatency(item.latencyMs)}</td>
                  <td className={ui.num}>{item.attempts}</td>
                  <td className={ui.mono}>{item.requestId}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {page.nextCursor ? (
        <nav aria-label="Pagination" className={styles.pagination}>
          <Link className={ui.buttonSecondary} href={olderHref(filters, page.nextCursor)}>
            Older requests
          </Link>
        </nav>
      ) : null}
    </>
  );
}
