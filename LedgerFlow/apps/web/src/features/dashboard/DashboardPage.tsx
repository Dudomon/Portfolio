import { useMemo, type ReactNode } from 'react';
import { Link, useSearchParams } from 'react-router-dom';

import { DOCUMENT_STATUSES, describeError, useStats, type StatsRange } from '@/api';
import { formatDuration, formatMoney, formatPercent } from '@/lib/format';
import { Button, Callout, Select, Skeleton } from '@/ui';

import { STATUS_META } from '../documents';
import styles from './DashboardPage.module.css';

const RANGES = [
  { value: '7', label: 'Last 7 days' },
  { value: '30', label: 'Last 30 days' },
  { value: '90', label: 'Last 90 days' },
] as const;
type RangeDays = (typeof RANGES)[number]['value'];
const DEFAULT_RANGE: RangeDays = '30';

const DAY_MS = 86_400_000;

/**
 * Whole UTC days, so the range (and therefore the query key) is stable for the
 * rest of the day instead of changing on every render.
 */
export function rangeForDays(days: number, now = Date.now()): StatsRange {
  const startOfTomorrow = Math.floor(now / DAY_MS) * DAY_MS + DAY_MS;
  return {
    from: new Date(startOfTomorrow - days * DAY_MS).toISOString(),
    to: new Date(startOfTomorrow).toISOString(),
  };
}

export function DashboardPage() {
  const [params, setParams] = useSearchParams();
  const rangeParam = params.get('range');
  const days: RangeDays =
    RANGES.find((range) => range.value === rangeParam)?.value ?? DEFAULT_RANGE;
  const range = useMemo(() => rangeForDays(Number(days)), [days]);
  const stats = useStats(range);

  const total = stats.data ? Object.values(stats.data.by_status).reduce((a, b) => a + b, 0) : 0;

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <div>
          <h1 className={styles.title}>Dashboard</h1>
          <p className={styles.muted}>Throughput and extraction quality.</p>
        </div>
        <Select
          label="Time range"
          hideLabel
          options={RANGES}
          value={days}
          onChange={(event) =>
            setParams(event.target.value === DEFAULT_RANGE ? {} : { range: event.target.value }, {
              replace: true,
            })
          }
          className={styles.range}
        />
      </header>

      {stats.isError && (
        <Callout
          tone="danger"
          title="Statistics could not be loaded"
          actions={
            <Button size="sm" onClick={() => void stats.refetch()}>
              Retry
            </Button>
          }
        >
          {describeError(stats.error)}
        </Callout>
      )}

      <section aria-label="Key figures" className={styles.tiles} aria-busy={stats.isPending}>
        <StatTile label="Documents" value={stats.data && total.toLocaleString()} />
        <StatTile
          label="Awaiting review"
          value={stats.data?.by_status.needs_review.toLocaleString()}
          footer={<Link to="/documents?status=needs_review">Open queue</Link>}
        />
        <StatTile
          label="Auto-approval rate"
          value={stats.data && formatPercent(stats.data.auto_approval_rate)}
        />
        <StatTile
          label="Average confidence"
          value={stats.data && formatPercent(stats.data.avg_confidence)}
        />
        <StatTile
          label="Processing time (p95)"
          value={stats.data && formatDuration(stats.data.p95_processing_ms)}
        />
        <StatTile
          label="LLM cost"
          value={stats.data && formatMoney(stats.data.llm_cost_usd, 'USD')}
        />
      </section>

      <section className={styles.card} aria-labelledby="by-status-title">
        <h2 id="by-status-title" className={styles.cardTitle}>
          Documents by status
        </h2>
        {stats.data ? (
          <ul className={styles.bars}>
            {DOCUMENT_STATUSES.map((status) => {
              const count = stats.data.by_status[status];
              const share = total === 0 ? 0 : count / total;
              return (
                <li key={status} className={styles.barRow}>
                  <Link to={`/documents?status=${status}`} className={styles.barLabel}>
                    {STATUS_META[status].label}
                  </Link>
                  <span className={styles.barTrack} aria-hidden="true">
                    <span
                      className={styles.barFill}
                      data-tone={STATUS_META[status].tone}
                      style={{ width: `${share * 100}%` }}
                    />
                  </span>
                  <span className={styles.barValue}>{count.toLocaleString()}</span>
                </li>
              );
            })}
          </ul>
        ) : (
          !stats.isError && <Skeleton shape="rect" height="14rem" />
        )}
      </section>
    </div>
  );
}

function StatTile({
  label,
  value,
  footer,
}: {
  label: string;
  value: string | undefined;
  footer?: ReactNode;
}) {
  return (
    <div className={styles.tile}>
      <dl className={styles.tileBody}>
        <dt className={styles.tileLabel}>{label}</dt>
        <dd className={styles.tileValue}>{value ?? <Skeleton width="5rem" height="1.75rem" />}</dd>
      </dl>
      {footer && <div className={styles.tileFooter}>{footer}</div>}
    </div>
  );
}
