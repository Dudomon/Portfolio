import {
  Injectable,
  type CallHandler,
  type ExecutionContext,
  type NestInterceptor,
} from '@nestjs/common';
import type { Request, Response } from 'express';
import type { Observable } from 'rxjs';
import { MetricsService } from './metrics.service';

/**
 * Records request duration per route template (not raw URL, to keep label
 * cardinality bounded). Measured on the response's `close` event rather
 * than the handler's observable so streamed responses are timed until the
 * last byte. Requests rejected by guards never reach interceptors; those are
 * counted by `gateway_rejections_total` instead.
 */
@Injectable()
export class HttpMetricsInterceptor implements NestInterceptor {
  constructor(private readonly metrics: MetricsService) {}

  intercept(context: ExecutionContext, next: CallHandler): Observable<unknown> {
    const http = context.switchToHttp();
    const request = http.getRequest<Request>();
    const response = http.getResponse<Response>();
    const route = `${context.getClass().name}.${context.getHandler().name}`;
    const stop = this.metrics.httpDuration.startTimer({ method: request.method, route });

    response.once('close', () => stop({ status: String(response.statusCode) }));
    return next.handle();
  }
}
