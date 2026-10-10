import { createHash, timingSafeEqual } from 'node:crypto';
import { Inject, Injectable, type CanActivate, type ExecutionContext } from '@nestjs/common';
import type { Request } from 'express';
import { ApiError } from '../common/api-error';
import { ENV } from '../config/config.module';
import type { AppEnv } from '../config/env';
import { bearerToken } from './request-context';

/**
 * Protects the admin API with a static bearer token. Both sides are hashed
 * before comparison so timingSafeEqual sees equal-length inputs and the
 * token's length is not revealed through timing.
 */
@Injectable()
export class AdminGuard implements CanActivate {
  private readonly expected: Buffer;

  constructor(@Inject(ENV) env: AppEnv) {
    this.expected = digest(env.ADMIN_TOKEN);
  }

  canActivate(context: ExecutionContext): boolean {
    const token = bearerToken(context.switchToHttp().getRequest<Request>());
    if (!token || !timingSafeEqual(digest(token), this.expected)) {
      throw new ApiError(401, 'authentication_error', 'Invalid admin token', 'invalid_admin_token');
    }
    return true;
  }
}

function digest(value: string): Buffer {
  return createHash('sha256').update(value).digest();
}
