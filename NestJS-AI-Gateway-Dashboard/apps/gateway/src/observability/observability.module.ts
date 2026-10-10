import { randomUUID } from 'node:crypto';
import type { IncomingMessage, ServerResponse } from 'node:http';
import { Global, Injectable, Module, type OnApplicationShutdown } from '@nestjs/common';
import { APP_INTERCEPTOR } from '@nestjs/core';
import { trace } from '@opentelemetry/api';
import { LoggerModule } from 'nestjs-pino';
import { ENV } from '../config/config.module';
import type { AppEnv } from '../config/env';
import { shutdownTracing } from '../tracing';
import { HttpMetricsInterceptor } from './http-metrics.interceptor';
import { MetricsService } from './metrics.service';
import { ObservabilityController } from './observability.controller';

const SAFE_REQUEST_ID = /^[\w.:-]{1,128}$/;

/**
 * Honors an inbound x-request-id (so IDs propagate across services) only if
 * it is well-formed, which prevents log injection via the header, and echoes
 * the ID back to the client for support correlation.
 */
function requestId(req: IncomingMessage, res: ServerResponse): string {
  const inbound = req.headers['x-request-id'];
  const id = typeof inbound === 'string' && SAFE_REQUEST_ID.test(inbound) ? inbound : randomUUID();
  res.setHeader('x-request-id', id);
  return id;
}

/** Flushes pending spans after the HTTP server has stopped accepting requests. */
@Injectable()
class TracingLifecycle implements OnApplicationShutdown {
  onApplicationShutdown(): Promise<void> {
    return shutdownTracing();
  }
}

@Global()
@Module({
  imports: [
    LoggerModule.forRootAsync({
      inject: [ENV],
      useFactory: (env: AppEnv) => ({
        pinoHttp: {
          level: env.LOG_LEVEL,
          genReqId: requestId,
          // Never log credentials, even at debug level.
          redact: ['req.headers.authorization', 'req.headers["api-key"]', 'req.headers.cookie'],
          // Correlate logs with traces when OpenTelemetry is active.
          mixin: () => {
            const span = trace.getActiveSpan()?.spanContext();
            return span ? { trace_id: span.traceId, span_id: span.spanId } : {};
          },
          autoLogging: { ignore: (req) => req.url === '/metrics' || req.url === '/healthz' },
          ...(env.NODE_ENV === 'development'
            ? { transport: { target: 'pino-pretty', options: { singleLine: true } } }
            : {}),
        },
      }),
    }),
  ],
  controllers: [ObservabilityController],
  providers: [
    MetricsService,
    TracingLifecycle,
    { provide: APP_INTERCEPTOR, useClass: HttpMetricsInterceptor },
  ],
  exports: [MetricsService],
})
export class ObservabilityModule {}
