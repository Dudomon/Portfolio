import { useFieldArray, useFormContext, useWatch } from 'react-hook-form';

import type { FieldValue } from '@/api';
import { cx } from '@/lib/cx';
import { formatMoney } from '@/lib/format';
import { Button, IconButton, PlusIcon, TextField, TrashIcon } from '@/ui';

import { checkLineItems, expectedLineTotal, type ReviewFormValues } from './fieldModel';
import { FieldMeta, isLowConfidence } from './ReviewField';
import styles from './ReviewForm.module.css';

const EMPTY_LINE = { description: '', quantity: '1', unit_price: '', total: '' };

const COLUMNS = [
  { key: 'description', label: 'Description', numeric: false },
  { key: 'quantity', label: 'Qty', numeric: true },
  { key: 'unit_price', label: 'Unit price', numeric: true },
  { key: 'total', label: 'Total', numeric: true },
] as const;

export function LineItemsEditor({
  meta,
  readOnly,
}: {
  meta: FieldValue | undefined;
  readOnly: boolean;
}) {
  const {
    control,
    register,
    formState: { errors, dirtyFields },
  } = useFormContext<ReviewFormValues>();
  const { fields, append, remove } = useFieldArray({ control, name: 'line_items' });
  const values = useWatch({ control }) as ReviewFormValues;
  const edited = dirtyFields.line_items !== undefined;
  const check = checkLineItems(values);
  const currency = values.currency.trim().toUpperCase() || null;

  return (
    <section
      className={cx(styles.group, !edited && isLowConfidence(meta) && styles.lowConfidence)}
      aria-labelledby="line-items-title"
    >
      <div className={styles.groupHeader}>
        <h3 id="line-items-title" className={styles.groupTitle}>
          Line items
        </h3>
        <FieldMeta meta={meta} edited={edited} label="Line items" />
      </div>

      {fields.length === 0 ? (
        <p className={styles.muted}>No line items were extracted.</p>
      ) : (
        <div className={styles.lineTableScroller}>
          <table className={styles.lineTable}>
            <thead>
              <tr>
                {COLUMNS.map((column) => (
                  <th key={column.key} scope="col" className={cx(column.numeric && styles.numeric)}>
                    {column.label}
                  </th>
                ))}
                {!readOnly && (
                  <th scope="col">
                    <span className="visually-hidden">Actions</span>
                  </th>
                )}
              </tr>
            </thead>
            <tbody>
              {fields.map((field, index) => {
                const lineErrors = errors.line_items?.[index];
                const line = values.line_items[index];
                const expected = line ? expectedLineTotal(line) : null;
                return (
                  <tr key={field.id}>
                    {COLUMNS.map((column) => (
                      <td key={column.key}>
                        <TextField
                          label={`Line ${index + 1} ${column.label.toLowerCase()}`}
                          hideLabel
                          size="sm"
                          readOnly={readOnly}
                          inputMode={column.numeric ? 'decimal' : undefined}
                          inputClassName={cx(column.numeric && styles.numeric)}
                          error={lineErrors?.[column.key]?.message}
                          hint={
                            column.key === 'total' && expected
                              ? `Qty × unit price = ${expected}`
                              : undefined
                          }
                          {...register(`line_items.${index}.${column.key}`)}
                        />
                      </td>
                    ))}
                    {!readOnly && (
                      <td>
                        <IconButton
                          label={`Remove line ${index + 1}`}
                          icon={<TrashIcon />}
                          size="sm"
                          onClick={() => remove(index)}
                        />
                      </td>
                    )}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      <div className={styles.lineFooter}>
        {!readOnly && (
          <Button size="sm" startIcon={<PlusIcon />} onClick={() => append(EMPTY_LINE)}>
            Add line
          </Button>
        )}
        {check && (
          <p className={cx(styles.lineSum, !check.matchesTotal && styles.warningText)}>
            Lines sum to {formatMoney(check.sum, currency)}
            {!check.matchesTotal && ', which matches neither the total nor the total net of tax'}
          </p>
        )}
      </div>
    </section>
  );
}
