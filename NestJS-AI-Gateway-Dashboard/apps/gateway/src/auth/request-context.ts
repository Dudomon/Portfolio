import { createParamDecorator, type ExecutionContext } from '@nestjs/common';
import type { Request } from 'express';
import type { AuthenticatedKey } from './api-key.service';

declare module 'express-serve-static-core' {
  interface Request {
    /** Set by ApiKeyGuard on authenticated routes. */
    apiKey?: AuthenticatedKey;
  }
}

/** Injects the caller's API key; only valid behind ApiKeyGuard. */
export const CurrentKey = createParamDecorator((_data: unknown, ctx: ExecutionContext) => {
  const key = ctx.switchToHttp().getRequest<Request>().apiKey;
  if (!key) throw new Error('CurrentKey used on a route without ApiKeyGuard');
  return key;
});

/** Extracts a bearer token from the Authorization header. */
export function bearerToken(request: Request): string | null {
  const header = request.headers.authorization;
  if (!header) return null;
  const [scheme, token] = header.split(' ');
  return scheme?.toLowerCase() === 'bearer' && token ? token : null;
}
