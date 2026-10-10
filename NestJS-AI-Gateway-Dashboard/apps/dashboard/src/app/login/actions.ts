'use server';

import { AuthenticationError } from '@aigw/sdk';
import { cookies } from 'next/headers';
import { redirect } from 'next/navigation';
import { SESSION_COOKIE, adminClient } from '@/lib/admin';
import { textField } from '@/lib/form-data';

export interface LoginState {
  error: string | null;
}

const SESSION_SECONDS = 8 * 60 * 60;

/**
 * Verifies the admin token against the gateway before storing it, so a typo
 * shows an error here instead of a redirect loop later.
 */
export async function login(_previous: LoginState, formData: FormData): Promise<LoginState> {
  const token = textField(formData, 'token');
  if (!token) return { error: 'Enter the admin token.' };

  try {
    await adminClient(token).listRoutes();
  } catch (error) {
    if (error instanceof AuthenticationError)
      return { error: 'That token was rejected by the gateway.' };
    return { error: 'The gateway is unreachable. Check GATEWAY_URL and try again.' };
  }

  (await cookies()).set(SESSION_COOKIE, token, {
    httpOnly: true,
    secure: process.env.NODE_ENV === 'production',
    sameSite: 'strict',
    path: '/',
    maxAge: SESSION_SECONDS,
  });
  redirect('/');
}

export async function logout(): Promise<void> {
  (await cookies()).delete(SESSION_COOKIE);
  redirect('/login');
}
