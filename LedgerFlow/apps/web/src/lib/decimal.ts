/**
 * Exact arithmetic on decimal strings (the API's money representation).
 *
 * Values are held as a scaled BigInt (`coefficient × 10^-scale`) so that
 * "0.1" + "0.2" is exactly "0.3". Only the operations the UI needs are provided.
 */

const DECIMAL_PATTERN = /^([+-])?(\d+)(?:\.(\d+))?$/;

export interface Decimal {
  readonly coefficient: bigint;
  readonly scale: number;
}

type DecimalInput = string | Decimal;

export function isDecimalString(value: string): boolean {
  return DECIMAL_PATTERN.test(value.trim());
}

export function parseDecimal(value: string): Decimal {
  const match = DECIMAL_PATTERN.exec(value.trim());
  if (!match) {
    throw new RangeError(`Invalid decimal string: "${value}"`);
  }
  const [, sign, integerPart = '0', fractionPart = ''] = match;
  const magnitude = BigInt(integerPart + fractionPart);
  return {
    coefficient: sign === '-' ? -magnitude : magnitude,
    scale: fractionPart.length,
  };
}

export function toDecimalString({ coefficient, scale }: Decimal): string {
  const negative = coefficient < 0n;
  const digits = (negative ? -coefficient : coefficient).toString().padStart(scale + 1, '0');
  const integerPart = digits.slice(0, digits.length - scale);
  const fractionPart = scale > 0 ? `.${digits.slice(digits.length - scale)}` : '';
  return `${negative ? '-' : ''}${integerPart}${fractionPart}`;
}

const toDecimal = (value: DecimalInput): Decimal =>
  typeof value === 'string' ? parseDecimal(value) : value;

function rescale(value: Decimal, scale: number): bigint {
  return value.coefficient * 10n ** BigInt(scale - value.scale);
}

function align(a: Decimal, b: Decimal): [bigint, bigint, number] {
  const scale = Math.max(a.scale, b.scale);
  return [rescale(a, scale), rescale(b, scale), scale];
}

export function addDecimals(...values: DecimalInput[]): string {
  const sum = values.map(toDecimal).reduce<Decimal>(
    (acc, value) => {
      const [x, y, scale] = align(acc, value);
      return { coefficient: x + y, scale };
    },
    { coefficient: 0n, scale: 0 },
  );
  return toDecimalString(sum);
}

export function subtractDecimals(a: DecimalInput, b: DecimalInput): string {
  const [x, y, scale] = align(toDecimal(a), toDecimal(b));
  return toDecimalString({ coefficient: x - y, scale });
}

export function multiplyDecimals(a: DecimalInput, b: DecimalInput): string {
  const x = toDecimal(a);
  const y = toDecimal(b);
  return toDecimalString({ coefficient: x.coefficient * y.coefficient, scale: x.scale + y.scale });
}

/** Rounds half away from zero, the convention used on invoices. */
export function roundDecimal(value: DecimalInput, scale: number): string {
  const decimal = toDecimal(value);
  if (decimal.scale <= scale) {
    return toDecimalString({ coefficient: rescale(decimal, scale), scale });
  }
  const divisor = 10n ** BigInt(decimal.scale - scale);
  const negative = decimal.coefficient < 0n;
  const magnitude = negative ? -decimal.coefficient : decimal.coefficient;
  let quotient = magnitude / divisor;
  if ((magnitude % divisor) * 2n >= divisor) {
    quotient += 1n;
  }
  return toDecimalString({ coefficient: negative ? -quotient : quotient, scale });
}

export function compareDecimals(a: DecimalInput, b: DecimalInput): -1 | 0 | 1 {
  const [x, y] = align(toDecimal(a), toDecimal(b));
  if (x === y) return 0;
  return x < y ? -1 : 1;
}

export function decimalsEqual(a: DecimalInput, b: DecimalInput): boolean {
  return compareDecimals(a, b) === 0;
}
