import { NextResponse, type NextRequest } from 'next/server';

// Kept in sync with SESSION_COOKIE in lib/admin.ts; that module is
// server-only and cannot be imported into the edge middleware bundle.
const SESSION_COOKIE = 'aigw_admin';

/**
 * Cheap presence check so unauthenticated visitors land on /login instead of
 * a failed render. The token itself is validated by the gateway on every
 * admin API call.
 */
export function middleware(request: NextRequest): NextResponse {
  if (request.cookies.has(SESSION_COOKIE)) return NextResponse.next();
  const login = new URL('/login', request.url);
  return NextResponse.redirect(login);
}

export const config = {
  matcher: ['/((?!login|_next/static|_next/image|favicon.ico).*)'],
};
