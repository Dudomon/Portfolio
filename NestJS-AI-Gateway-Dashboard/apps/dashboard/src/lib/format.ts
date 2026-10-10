const usd = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' });
const usdPrecise = new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency: 'USD',
  maximumSignificantDigits: 3,
});
const compact = new Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 1 });
const percent = new Intl.NumberFormat('en-US', { style: 'percent', maximumFractionDigits: 1 });

/** Dollars with cents; sub-cent amounts keep 3 significant digits instead of rounding to $0.00. */
export function formatUsd(value: number): string {
  if (value !== 0 && Math.abs(value) < 0.01) return usdPrecise.format(value);
  return usd.format(value);
}

export function formatCount(value: number): string {
  return compact.format(value);
}

export function formatPercent(ratio: number): string {
  return percent.format(ratio);
}

export function formatLatency(ms: number | null): string {
  if (ms === null) return '—';
  return ms >= 1_000 ? `${(ms / 1_000).toFixed(2)} s` : `${Math.round(ms)} ms`;
}

/** Formats an ISO timestamp in UTC so server and client render identical text. */
export function formatTimestamp(iso: string): string {
  return `${iso.slice(0, 10)} ${iso.slice(11, 19)} UTC`;
}

/** "2025-03-07" -> "Mar 7" for chart axes. */
export function formatDay(day: string): string {
  const date = new Date(`${day}T00:00:00Z`);
  return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', timeZone: 'UTC' });
}
