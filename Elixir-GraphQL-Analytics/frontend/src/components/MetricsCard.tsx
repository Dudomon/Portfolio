import { memo } from "react";

/**
 * Individual metric display card.
 *
 * Design considerations:
 * - Large value for quick scanning
 * - Change indicator shows trend at a glance
 * - Color coding: green for positive, red for negative
 * - Accessible: color is not the only indicator (arrow + number)
 *
 * Format types:
 * - number: Integer with thousand separators (1,234)
 * - currency: Dollar format ($1,234.56)
 * - decimal: Float with 2 decimal places (12.34)
 * - percent: Percentage (12.3%)
 */

type FormatType = "number" | "currency" | "decimal" | "percent";

interface MetricsCardProps {
  title: string;
  value: number | string;
  change?: number | null;
  format: FormatType;
}

export const MetricsCard = memo(function MetricsCard({
  title,
  value,
  change,
  format,
}: MetricsCardProps): JSX.Element {
  const formattedValue = formatValue(value, format);
  const changeIndicator = change != null ? formatChange(change) : null;

  return (
    <article className="metrics-card">
      <h3 className="metrics-card__title">{title}</h3>

      <div className="metrics-card__value">{formattedValue}</div>

      {changeIndicator && (
        <div
          className={`metrics-card__change metrics-card__change--${changeIndicator.direction}`}
          aria-label={`${changeIndicator.direction === "up" ? "Increased" : "Decreased"} by ${Math.abs(change!)}%`}
        >
          <span className="metrics-card__arrow">{changeIndicator.arrow}</span>
          <span className="metrics-card__change-value">
            {changeIndicator.formatted}
          </span>
        </div>
      )}
    </article>
  );
});

function formatValue(value: number | string, format: FormatType): string {
  const numValue = typeof value === "string" ? parseFloat(value) : value;

  if (isNaN(numValue)) {
    return String(value);
  }

  switch (format) {
    case "number":
      return new Intl.NumberFormat("en-US", {
        maximumFractionDigits: 0,
      }).format(numValue);

    case "currency":
      return new Intl.NumberFormat("en-US", {
        style: "currency",
        currency: "USD",
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
      }).format(numValue);

    case "decimal":
      return new Intl.NumberFormat("en-US", {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
      }).format(numValue);

    case "percent":
      return new Intl.NumberFormat("en-US", {
        style: "percent",
        minimumFractionDigits: 1,
        maximumFractionDigits: 1,
      }).format(numValue / 100);

    default:
      return String(value);
  }
}

interface ChangeIndicator {
  direction: "up" | "down" | "neutral";
  arrow: string;
  formatted: string;
}

function formatChange(change: number): ChangeIndicator {
  const absChange = Math.abs(change);
  const formatted = `${absChange.toFixed(1)}%`;

  if (change > 0) {
    return { direction: "up", arrow: "\u2191", formatted };
  }

  if (change < 0) {
    return { direction: "down", arrow: "\u2193", formatted };
  }

  return { direction: "neutral", arrow: "\u2192", formatted };
}
