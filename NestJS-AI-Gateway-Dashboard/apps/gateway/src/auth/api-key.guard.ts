import { Injectable, type CanActivate, type ExecutionContext } from '@nestjs/common';
import type { Request } from 'express';
import { ApiError } from '../common/api-error';
import { MetricsService } from '../observability/metrics.service';
import { ApiKeyService } from './api-key.service';
import { bearerToken } from './request-context';

@Injectable()
export class ApiKeyGuard implements CanActivate {
  constructor(
    private readonly keys: ApiKeyService,
    private readonly metrics: MetricsService,
  ) {}

  async canActivate(context: ExecutionContext): Promise<boolean> {
    const request = context.switchToHttp().getRequest<Request>();
    const token = bearerToken(request);
    const key = token ? await this.keys.authenticate(token) : null;
    if (!key) {
      this.metrics.rejections.inc({ reason: 'unauthenticated' });
      throw ApiError.unauthorized();
    }
    request.apiKey = key;
    return true;
  }
}
