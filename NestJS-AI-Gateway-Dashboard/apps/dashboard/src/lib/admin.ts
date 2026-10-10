import 'server-only';
import { AuthenticationError, GatewayAdminClient } from '@aigw/sdk';
import { cookies } from 'next/headers';
import { redirect } from 'next/navigation';

export const SESSION_COOKIE = 'aigw_admin';

export function gatewayUrl(): string {
  return process.env.GATEWAY_URL ?? 'http://localhost:3000';
}

export function adminClient(token: string): GatewayAdminClient {
  // Dashboard pages render on demand; fail quickly rather than hold the page.
  return new GatewayAdminClient({ baseUrl: gatewayUrl(), adminToken: token, timeoutMs: 10_000 });
}

/**
 * Runs an admin API call with the token from the session cookie. The token
 * never reaches the browser: it lives in an httpOnly cookie and is only read
 * here, in server components and server actions. A rejected token sends the
 * user back to the login page.
 */
export async function withAdmin<T>(call: (client: GatewayAdminClient) => Promise<T>): Promise<T> {
  const token = (await cookies()).get(SESSION_COOKIE)?.value;
  if (!token) redirect('/login');
  try {
    return await call(adminClient(token));
  } catch (error) {
    if (error instanceof AuthenticationError) redirect('/login?expired=1');
    throw error;
  }
}
