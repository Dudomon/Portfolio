'use client';

import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  type TooltipContentProps,
  type TooltipValueType,
} from 'recharts';
import { formatCount, formatLatency, formatPercent, formatUsd } from '@/lib/format';
import ui from '../ui.module.css';
import styles from './charts.module.css';

export interface MetricPoint {
  /** Category or x-axis label (already formatted for display). */
  label: string;
  /** null renders as a gap (e.g. no successful requests that day). */
  value: number | null;
}

interface MetricChartProps {
  kind: 'bar' | 'line';
  data: MetricPoint[];
  /** Names the measure for the tooltip and the table header. */
  measure: string;
  /** Header for the label column of the table view. */
  labelHeader: string;
  /** A formatter name rather than a function: props from server components must be serializable. */
  unit: keyof typeof FORMATTERS;
}

const FORMATTERS = {
  usd: formatUsd,
  count: formatCount,
  latency: formatLatency,
  percent: formatPercent,
} satisfies Record<string, (value: number) => string>;

const AXIS_TICK = { fill: 'var(--text-muted)', fontSize: 12 };

/**
 * One single-series chart (so no legend: the panel title names the series),
 * with a hover tooltip and an equivalent data table for screen readers and
 * anyone who prefers exact numbers. Colors come from CSS custom properties so
 * the chart follows the light/dark theme.
 */
export function MetricChart({ kind, data, measure, labelHeader, unit }: MetricChartProps) {
  const format = FORMATTERS[unit];
  const tooltip = (props: TooltipContentProps<TooltipValueType, string | number>) => {
    const point = props.payload?.[0];
    if (!props.active || !point || typeof point.value !== 'number') return null;
    return (
      <div className={styles.tooltip}>
        <div className={styles.tooltipLabel}>{props.label}</div>
        <div className={styles.tooltipValue}>
          {measure}: {format(point.value)}
        </div>
      </div>
    );
  };

  const axes = (
    <>
      <CartesianGrid vertical={false} stroke="var(--grid)" />
      <XAxis
        dataKey="label"
        tick={AXIS_TICK}
        tickLine={false}
        axisLine={{ stroke: 'var(--border)' }}
        minTickGap={16}
      />
      <YAxis
        tick={AXIS_TICK}
        tickLine={false}
        axisLine={false}
        width={64}
        tickFormatter={(value: number) => format(value)}
      />
      <Tooltip content={tooltip} cursor={{ fill: 'var(--surface-2)', stroke: 'var(--border)' }} />
    </>
  );

  return (
    <div>
      <div className={styles.chart} aria-hidden="true">
        <ResponsiveContainer width="100%" height="100%">
          {kind === 'bar' ? (
            <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
              {axes}
              <Bar
                dataKey="value"
                fill="var(--series-1)"
                radius={[4, 4, 0, 0]}
                maxBarSize={28}
                isAnimationActive={false}
              />
            </BarChart>
          ) : (
            <LineChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
              {axes}
              <Line
                dataKey="value"
                stroke="var(--series-1)"
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4, stroke: 'var(--surface-1)', strokeWidth: 2 }}
                connectNulls={false}
                isAnimationActive={false}
              />
            </LineChart>
          )}
        </ResponsiveContainer>
      </div>
      <details className={styles.tableToggle}>
        <summary>View as table</summary>
        <div className={ui.tableWrap}>
          <table className={ui.table}>
            <thead>
              <tr>
                <th scope="col">{labelHeader}</th>
                <th scope="col" className={ui.num}>
                  {measure}
                </th>
              </tr>
            </thead>
            <tbody>
              {data.map((point) => (
                <tr key={point.label}>
                  <td>{point.label}</td>
                  <td className={ui.num}>{point.value === null ? '—' : format(point.value)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </div>
  );
}
