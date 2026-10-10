import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-http';
import { ExpressInstrumentation } from '@opentelemetry/instrumentation-express';
import { HttpInstrumentation } from '@opentelemetry/instrumentation-http';
import { NestInstrumentation } from '@opentelemetry/instrumentation-nestjs-core';
import { NodeSDK } from '@opentelemetry/sdk-node';

/**
 * OpenTelemetry bootstrap, run as a side effect of importing this module.
 * main.ts imports it before Nest, Express or http are loaded so the
 * instrumentations can patch them. Tracing is opt-in: without
 * OTEL_EXPORTER_OTLP_ENDPOINT nothing starts and the @opentelemetry/api
 * calls elsewhere are no-ops.
 */
const sdk = process.env.OTEL_EXPORTER_OTLP_ENDPOINT
  ? new NodeSDK({
      serviceName: process.env.OTEL_SERVICE_NAME ?? 'ai-gateway',
      // Reads OTEL_EXPORTER_OTLP_ENDPOINT / _HEADERS from the environment.
      traceExporter: new OTLPTraceExporter(),
      instrumentations: [
        // Provider calls use fetch (undici), which the http instrumentation
        // does not see; the router creates explicit spans for them instead.
        new HttpInstrumentation({ ignoreIncomingRequestHook: (req) => req.url === '/metrics' }),
        new ExpressInstrumentation(),
        new NestInstrumentation(),
      ],
    })
  : undefined;

sdk?.start();

/** Flushes buffered spans; called from the Nest shutdown sequence. */
export async function shutdownTracing(): Promise<void> {
  await sdk?.shutdown();
}
