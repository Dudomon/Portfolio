package observability

import (
	"context"
	"net/http"

	"go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp"
	"go.opentelemetry.io/otel/propagation"
)

// The W3C propagator is passed explicitly rather than installed with
// otel.SetTextMapPropagator: the worker only needs trace context and should
// not mutate process-wide state to get it.

// ContextWithTraceParent returns ctx carrying the remote span context from a
// W3C traceparent value, as published by the API's outbox relay. An empty or
// malformed value leaves ctx without a span context.
func ContextWithTraceParent(ctx context.Context, traceParent string) context.Context {
	if traceParent == "" {
		return ctx
	}
	return propagation.TraceContext{}.Extract(ctx, propagation.MapCarrier{"traceparent": traceParent})
}

// TracingTransport wraps base so outgoing requests carry the traceparent of
// the context they were made with. Without an SDK tracer provider the client
// span is non-recording and the upstream span context is forwarded as-is,
// which keeps the API's trace continuous without the worker exporting spans.
func TracingTransport(base http.RoundTripper) http.RoundTripper {
	return otelhttp.NewTransport(base, otelhttp.WithPropagators(propagation.TraceContext{}))
}
