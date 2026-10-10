import type { Metadata } from 'next';
import { MetricChart } from '@/components/charts/MetricChart';
import { PageHeader } from '@/components/PageHeader';
import { StatCard } from '@/components/StatCard';
import ui from '@/components/ui.module.css';
import { withAdmin } from '@/lib/admin';
import { formatCount, formatDay, formatLatency, formatPercent, formatUsd } from '@/lib/format';
import styles from './overview.module.css';

export const metadata: Metadata = { title: 'Overview' };

export default async function OverviewPage() {
  // Independent queries; fetch in parallel so the page waits for the slowest, not the sum.
  const [summary, series, providers] = await withAdmin((admin) =>
    Promise.all([admin.usageSummary(), admin.usageTimeseries(), admin.providerStats()]),
  );

  const byDay = (pick: (p: (typeof series)[number]) => number | null) =>
    series.map((point) => ({ label: formatDay(point.date), value: pick(point) }));

  return (
    <>
      <PageHeader title="Overview">Last 30 days, UTC.</PageHeader>

      <dl className={styles.stats}>
        <StatCard label="Spend" value={formatUsd(summary.costUsd)} />
        <StatCard
          label="Requests"
          value={formatCount(summary.requests)}
          detail={`${formatCount(summary.cacheHits)} served from cache`}
        />
        <StatCard
          label="p95 latency"
          value={formatLatency(summary.p95LatencyMs)}
          detail="Successful, uncached requests"
        />
        <StatCard
          label="Error rate"
          value={formatPercent(summary.errorRate)}
          detail={`${formatCount(summary.errors)} failed requests`}
        />
      </dl>

      <div className={styles.grid}>
        <section className={ui.panel} aria-labelledby="spend-title">
          <h2 id="spend-title" className={ui.panelTitle}>
            Spend per day
          </h2>
          <p className={ui.panelHint}>Estimated from token usage and the configured price table.</p>
          <MetricChart
            kind="bar"
            data={byDay((p) => p.costUsd)}
            measure="Spend"
            labelHeader="Day"
            unit="usd"
          />
        </section>

        <section className={ui.panel} aria-labelledby="requests-title">
          <h2 id="requests-title" className={ui.panelTitle}>
            Requests per day
          </h2>
          <p className={ui.panelHint}>All requests, including failures and cache hits.</p>
          <MetricChart
            kind="bar"
            data={byDay((p) => p.requests)}
            measure="Requests"
            labelHeader="Day"
            unit="count"
          />
        </section>

        <section className={ui.panel} aria-labelledby="latency-title">
          <h2 id="latency-title" className={ui.panelTitle}>
            p95 latency per day
          </h2>
          <p className={ui.panelHint}>
            Full response time for successful, uncached requests; gaps mean no traffic.
          </p>
          <MetricChart
            kind="line"
            data={byDay((p) => p.p95LatencyMs)}
            measure="p95 latency"
            labelHeader="Day"
            unit="latency"
          />
        </section>

        <section className={ui.panel} aria-labelledby="errors-title">
          <h2 id="errors-title" className={ui.panelTitle}>
            Error rate by provider
          </h2>
          <p className={ui.panelHint}>
            Share of requests whose final attempt on that provider failed.
          </p>
          {providers.length === 0 ? (
            <p className={ui.empty}>No routed requests in this period.</p>
          ) : (
            <MetricChart
              kind="bar"
              data={providers.map((p) => ({ label: p.provider, value: p.errorRate }))}
              measure="Error rate"
              labelHeader="Provider"
              unit="percent"
            />
          )}
        </section>
      </div>
    </>
  );
}
