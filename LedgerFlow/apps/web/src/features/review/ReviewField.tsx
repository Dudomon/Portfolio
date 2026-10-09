import { useFormContext } from 'react-hook-form';

import type { FieldSource, FieldValue } from '@/api';
import { cx } from '@/lib/cx';
import { Badge, ConfidenceMeter, TextField } from '@/ui';

import { CONFIDENCE_THRESHOLD, type FieldDefinition, type ReviewFormValues } from './fieldModel';
import styles from './ReviewForm.module.css';

const SOURCE_LABEL: Record<FieldSource, string> = {
  llm: 'LLM',
  rule: 'Rule',
  human: 'Human',
};

export function FieldMeta({
  meta,
  edited,
  label,
}: {
  meta: FieldValue | undefined;
  edited: boolean;
  label: string;
}) {
  return (
    <span className={styles.meta}>
      {edited ? (
        <Badge tone="accent">Edited</Badge>
      ) : (
        meta && (
          <Badge tone={meta.source === 'human' ? 'accent' : 'neutral'} title="Value source">
            {SOURCE_LABEL[meta.source]}
          </Badge>
        )
      )}
      <ConfidenceMeter
        value={edited ? null : (meta?.confidence ?? null)}
        threshold={CONFIDENCE_THRESHOLD}
        label={`${label} confidence`}
      />
    </span>
  );
}

export function isLowConfidence(meta: FieldValue | undefined): boolean {
  return meta !== undefined && meta.confidence < CONFIDENCE_THRESHOLD;
}

export function ReviewField({
  definition,
  meta,
  readOnly,
}: {
  definition: FieldDefinition;
  meta: FieldValue | undefined;
  readOnly: boolean;
}) {
  const {
    register,
    formState: { errors, dirtyFields },
  } = useFormContext<ReviewFormValues>();
  const edited = Boolean(dirtyFields[definition.key]);
  const low = !edited && isLowConfidence(meta);

  return (
    <TextField
      label={definition.label}
      type={definition.type === 'date' ? 'date' : 'text'}
      inputMode={definition.type === 'decimal' ? 'decimal' : undefined}
      placeholder={definition.placeholder}
      readOnly={readOnly}
      labelAccessory={<FieldMeta meta={meta} edited={edited} label={definition.label} />}
      hint={low ? 'Low confidence: check against the document.' : undefined}
      error={errors[definition.key]?.message}
      className={cx(styles.field, low && styles.lowConfidence)}
      {...register(definition.key)}
    />
  );
}
