import { z } from 'zod';

import {
  LineItemSchema,
  type CorrectableFields,
  type DocType,
  type DocumentDetail,
  type FieldCorrections,
  type LineItem,
} from '@/api';
import {
  addDecimals,
  decimalsEqual,
  isDecimalString,
  multiplyDecimals,
  parseDecimal,
  roundDecimal,
  subtractDecimals,
} from '@/lib/decimal';

/**
 * Contract default for auto-approval. The tenant-specific value is not exposed
 * by the API, so the UI flags against the default.
 */
export const CONFIDENCE_THRESHOLD = 0.85;

export type ScalarFieldKey = Exclude<keyof CorrectableFields, 'line_items'>;

export interface FieldDefinition {
  key: ScalarFieldKey;
  label: string;
  type: 'text' | 'date' | 'decimal';
  /** Only shown for NF-e documents (or when the extractor found a value anyway). */
  nfeOnly?: boolean;
  placeholder?: string;
}

export const FIELD_GROUPS: readonly { title: string; fields: readonly FieldDefinition[] }[] = [
  {
    title: 'Supplier',
    fields: [
      { key: 'supplier_name', label: 'Supplier name', type: 'text' },
      {
        key: 'supplier_tax_id',
        label: 'Supplier tax ID',
        type: 'text',
        placeholder: '14-digit CNPJ or VAT id',
      },
    ],
  },
  {
    title: 'Invoice',
    fields: [
      { key: 'invoice_number', label: 'Invoice number', type: 'text' },
      { key: 'issue_date', label: 'Issue date', type: 'date' },
      { key: 'due_date', label: 'Due date', type: 'date' },
      { key: 'nfe_access_key', label: 'NF-e access key', type: 'text', nfeOnly: true },
    ],
  },
  {
    title: 'Amounts',
    fields: [
      { key: 'currency', label: 'Currency', type: 'text', placeholder: 'BRL' },
      { key: 'total_amount', label: 'Total amount', type: 'decimal' },
      { key: 'tax_amount', label: 'Tax amount', type: 'decimal' },
    ],
  },
];

/** Form state: every scalar is an input string ('' = empty); conversion happens on submit. */
export type ReviewFormValues = Record<ScalarFieldKey, string> & { line_items: LineItem[] };

const ISO_DATE = /^\d{4}-\d{2}-\d{2}$/;
const CNPJ = /^\d{14}$/;
const VAT_ID = /^[A-Z]{2}[0-9A-Z+*.]{2,12}$/;

const isIsoDate = (value: string) =>
  ISO_DATE.test(value) && !Number.isNaN(Date.parse(`${value}T00:00:00Z`));

const requiredText = (message: string) => z.string().trim().min(1, message);
const decimal = (label: string) =>
  z
    .string()
    .trim()
    .min(1, `Enter the ${label}.`)
    .refine(isDecimalString, 'Use digits with a dot for decimals, e.g. 1530.40');

const LineItemFormSchema = z.object({
  description: requiredText('Enter a description.'),
  quantity: decimal('quantity'),
  unit_price: decimal('unit price'),
  total: decimal('line total'),
});

export function createReviewSchema(docType: DocType | null) {
  return z
    .object({
      supplier_name: requiredText('Enter the supplier name.'),
      supplier_tax_id: z
        .string()
        .trim()
        .refine(
          (value) => CNPJ.test(value) || VAT_ID.test(value),
          'Enter a 14-digit CNPJ (digits only) or a VAT id such as DE123456789.',
        ),
      invoice_number: requiredText('Enter the invoice number.'),
      issue_date: z.string().refine(isIsoDate, 'Enter a valid date.'),
      due_date: z
        .string()
        .refine((value) => value === '' || isIsoDate(value), 'Enter a valid date.'),
      currency: z
        .string()
        .trim()
        .regex(/^[A-Za-z]{3}$/, 'Use a 3-letter ISO 4217 code, e.g. BRL.'),
      total_amount: decimal('total amount'),
      tax_amount: decimal('tax amount'),
      nfe_access_key: z
        .string()
        .trim()
        .refine(
          (value) =>
            docType === 'nfe' ? /^\d{44}$/.test(value) : value === '' || /^\d{44}$/.test(value),
          'The access key has 44 digits.',
        ),
      line_items: z.array(LineItemFormSchema),
    })
    .refine((values) => values.due_date === '' || values.due_date >= values.issue_date, {
      path: ['due_date'],
      message: 'Due date cannot be before the issue date.',
    }) satisfies z.ZodType<ReviewFormValues>;
}

function asInputString(value: unknown): string {
  if (typeof value === 'string') return value;
  if (typeof value === 'number') return String(value);
  return '';
}

export function toFormValues(document: DocumentDetail): ReviewFormValues {
  const fields = document.extraction?.fields ?? {};
  const read = (key: ScalarFieldKey) => asInputString(fields[key]?.value);
  const lineItems = LineItemSchema.array().safeParse(fields.line_items?.value);
  return {
    supplier_name: read('supplier_name'),
    supplier_tax_id: read('supplier_tax_id'),
    invoice_number: read('invoice_number'),
    issue_date: read('issue_date'),
    due_date: read('due_date'),
    currency: read('currency'),
    total_amount: read('total_amount'),
    tax_amount: read('tax_amount'),
    nfe_access_key: read('nfe_access_key'),
    line_items: lineItems.success ? lineItems.data : [],
  };
}

function toFieldValues(values: ReviewFormValues): CorrectableFields {
  const nullable = (value: string) => (value.trim() === '' ? null : value.trim());
  return {
    supplier_name: values.supplier_name.trim(),
    supplier_tax_id: values.supplier_tax_id.trim(),
    invoice_number: values.invoice_number.trim(),
    issue_date: values.issue_date,
    due_date: nullable(values.due_date),
    currency: values.currency.trim().toUpperCase(),
    total_amount: values.total_amount.trim(),
    tax_amount: values.tax_amount.trim(),
    nfe_access_key: nullable(values.nfe_access_key),
    line_items: values.line_items.map((item) => ({
      description: item.description.trim(),
      quantity: item.quantity.trim(),
      unit_price: item.unit_price.trim(),
      total: item.total.trim(),
    })),
  };
}

/**
 * Only fields whose normalised value actually changed are sent, so untouched
 * fields keep their machine source and confidence on the server.
 */
export function toCorrections(
  values: ReviewFormValues,
  original: ReviewFormValues,
): FieldCorrections {
  const next = toFieldValues(values);
  const before = toFieldValues(original);
  const changed = (Object.keys(next) as (keyof CorrectableFields)[]).filter(
    (key) => JSON.stringify(next[key]) !== JSON.stringify(before[key]),
  );
  return Object.fromEntries(changed.map((key) => [key, next[key]]));
}

/** The rounded `quantity × unit_price` when it disagrees with the stated line total. */
export function expectedLineTotal(item: LineItem): string | null {
  const { quantity, unit_price, total } = item;
  if (![quantity, unit_price, total].every((value) => isDecimalString(value))) return null;
  const expected = roundDecimal(multiplyDecimals(quantity, unit_price), parseDecimal(total).scale);
  return decimalsEqual(expected, total) ? null : expected;
}

export interface LineItemsCheck {
  sum: string;
  /** True when the lines add up to the total, either gross or net of tax. */
  matchesTotal: boolean;
}

export function checkLineItems(values: ReviewFormValues): LineItemsCheck | null {
  const totals = values.line_items.map((item) => item.total.trim());
  if (totals.length === 0 || !totals.every(isDecimalString)) return null;
  const sum = addDecimals(...totals);
  const { total_amount: total, tax_amount: tax } = values;
  if (!isDecimalString(total)) return { sum, matchesTotal: false };
  const matchesTotal =
    decimalsEqual(sum, total) ||
    (isDecimalString(tax) && decimalsEqual(sum, subtractDecimals(total, tax)));
  return { sum, matchesTotal };
}
