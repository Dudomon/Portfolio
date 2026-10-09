import { zodResolver } from '@hookform/resolvers/zod';
import { useEffect, useId } from 'react';
import { useForm } from 'react-hook-form';
import { z } from 'zod';

import { Button, Dialog, TextField } from '@/ui';

const RejectSchema = z.object({
  reason: z
    .string()
    .trim()
    .min(3, 'Explain why, so the supplier can be asked for a corrected document.')
    .max(500, 'Keep the reason under 500 characters.'),
});
type RejectValues = z.infer<typeof RejectSchema>;

export function RejectDialog({
  open,
  pending,
  onCancel,
  onConfirm,
}: {
  open: boolean;
  pending: boolean;
  onCancel: () => void;
  onConfirm: (reason: string) => void;
}) {
  const {
    register,
    handleSubmit,
    reset,
    formState: { errors },
  } = useForm<RejectValues>({ resolver: zodResolver(RejectSchema), defaultValues: { reason: '' } });
  const formId = useId();

  // Start from a blank reason every time the dialog opens.
  useEffect(() => {
    if (!open) reset();
  }, [open, reset]);

  return (
    <Dialog
      open={open}
      onClose={onCancel}
      title="Reject document"
      description="Rejected documents are not exported to the ERP."
      dismissible={!pending}
      footer={
        <>
          <Button onClick={onCancel} disabled={pending}>
            Cancel
          </Button>
          <Button type="submit" form={formId} variant="danger" loading={pending}>
            Reject
          </Button>
        </>
      }
    >
      <form id={formId} noValidate onSubmit={handleSubmit(({ reason }) => onConfirm(reason))}>
        <TextField
          label="Reason"
          placeholder="e.g. Duplicate of invoice 4711"
          error={errors.reason?.message}
          {...register('reason')}
        />
      </form>
    </Dialog>
  );
}
