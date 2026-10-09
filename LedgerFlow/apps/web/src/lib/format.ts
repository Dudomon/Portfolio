import { isDecimalString } from './decimal';

const moneyFormatters = new Map<string, Intl.NumberFormat>();

function moneyFormatter(currency: string, locale?: string): Intl.NumberFormat {
  const key = `${locale ?? ''}|${currency}`;
  let formatter = moneyFormatters.get(key);
  if (!formatter) {
    formatter = new Intl.NumberFormat(locale, { style: 'currency', currency });
    moneyFormatters.set(key, formatter);
  }
  return formatter;
}

/**
 * Formats a decimal string as money. The string goes to Intl as-is (ES2023
 * Intl.NumberFormat v3), which formats it exactly instead of rounding through
 * a binary float first.
 */
export function formatMoney(amount: string | null, currency: string | null, locale?: string) {
  if (amount === null || !isDecimalString(amount)) return '—';
  if (currency === null) return amount;
  try {
    return moneyFormatter(currency, locale).format(amount as Intl.StringNumericLiteral);
  } catch {
    // Unknown ISO 4217 codes throw a RangeError; the raw value is still useful.
    return `${amount} ${currency}`;
  }
}

const percentFormatter = new Intl.NumberFormat(undefined, {
  style: 'percent',
  maximumFractionDigits: 0,
});

export function formatPercent(ratio: number | null): string {
  return ratio === null ? '—' : percentFormatter.format(ratio);
}

const dateTimeFormatter = new Intl.DateTimeFormat(undefined, {
  dateStyle: 'medium',
  timeStyle: 'short',
});

export function formatDateTime(iso: string): string {
  return dateTimeFormatter.format(new Date(iso));
}

const compactNumber = new Intl.NumberFormat(undefined, { maximumFractionDigits: 1 });

export function formatDuration(ms: number): string {
  return ms < 1000 ? `${compactNumber.format(ms)} ms` : `${compactNumber.format(ms / 1000)} s`;
}

const fileSizeUnits = ['B', 'KB', 'MB', 'GB'] as const;

export function formatFileSize(bytes: number): string {
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < fileSizeUnits.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${compactNumber.format(value)} ${fileSizeUnits[unit] ?? 'B'}`;
}
