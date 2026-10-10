'use server';

import { APIError } from '@aigw/sdk';
import { revalidatePath } from 'next/cache';
import type { CreateKeyFormState } from '@/components/CreateKeyForm';
import { withAdmin } from '@/lib/admin';
import { textField } from '@/lib/form-data';

function optionalNumber(formData: FormData, name: string): number | undefined {
  const text = textField(formData, name);
  return text === '' ? undefined : Number(text);
}

export async function createKey(
  _previous: CreateKeyFormState,
  formData: FormData,
): Promise<CreateKeyFormState> {
  const name = textField(formData, 'name');
  const project = textField(formData, 'project');
  const budget = optionalNumber(formData, 'monthlyBudgetUsd');
  const rateLimit = optionalNumber(formData, 'rateLimitPerMinute');

  try {
    const created = await withAdmin((admin) =>
      admin.createKey({
        name,
        project,
        monthlyBudgetUsd: budget ?? null,
        ...(rateLimit !== undefined ? { rateLimitPerMinute: rateLimit } : {}),
      }),
    );
    revalidatePath('/keys');
    return { secret: created.secret, keyName: created.key.name, error: null };
  } catch (error) {
    // Validation errors come back from the gateway with a precise message.
    if (error instanceof APIError && error.status === 400) {
      return { secret: null, keyName: null, error: error.message };
    }
    throw error;
  }
}

export async function revokeKey(id: string): Promise<void> {
  await withAdmin((admin) => admin.revokeKey(id));
  revalidatePath('/keys');
}
