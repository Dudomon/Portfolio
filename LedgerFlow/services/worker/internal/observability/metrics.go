package observability

import (
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/collectors"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

const namespace = "ledgerflow_worker"

// Metrics implements the narrow recorder interfaces declared by the queue,
// pipeline and llm packages.
type Metrics struct {
	processed  *prometheus.CounterVec
	failed     *prometheus.CounterVec
	duplicates prometheus.Counter
	extraction *prometheus.HistogramVec
	llmCost    *prometheus.CounterVec
	llmTokens  *prometheus.CounterVec
}

// NewRegistry returns a registry with Go runtime and process collectors. A
// private registry (instead of prometheus.DefaultRegisterer) keeps tests
// independent and the exposed metric set explicit.
func NewRegistry() *prometheus.Registry {
	reg := prometheus.NewRegistry()
	reg.MustRegister(
		collectors.NewGoCollector(),
		collectors.NewProcessCollector(collectors.ProcessCollectorOpts{}),
	)
	return reg
}

// NewMetrics registers the worker's metrics on reg.
func NewMetrics(reg prometheus.Registerer) *Metrics {
	f := promauto.With(reg)
	return &Metrics{
		processed: f.NewCounterVec(prometheus.CounterOpts{
			Namespace: namespace,
			Name:      "messages_processed_total",
			Help:      "Messages handled successfully and acknowledged.",
		}, []string{"outcome"}),
		failed: f.NewCounterVec(prometheus.CounterOpts{
			Namespace: namespace,
			Name:      "messages_failed_total",
			Help:      "Messages whose handling failed, by what happened next (retry, dead_letter, requeued).",
		}, []string{"outcome"}),
		duplicates: f.NewCounter(prometheus.CounterOpts{
			Namespace: namespace,
			Name:      "duplicate_events_total",
			Help:      "Redelivered events skipped by the local idempotency cache.",
		}),
		extraction: f.NewHistogramVec(prometheus.HistogramOpts{
			Namespace: namespace,
			Name:      "extraction_duration_seconds",
			Help:      "Time spent extracting fields from one document.",
			// Rules finish in milliseconds; model calls take seconds and can
			// stretch to minutes with provider retries.
			Buckets: []float64{0.005, 0.025, 0.1, 0.5, 1, 2.5, 5, 10, 20, 40, 80, 160},
		}, []string{"extractor", "outcome"}),
		llmCost: f.NewCounterVec(prometheus.CounterOpts{
			Namespace: namespace,
			Name:      "llm_cost_usd_micros_total",
			Help:      "Estimated LLM spend in micro-USD, from token usage and the configured price table.",
		}, []string{"model"}),
		llmTokens: f.NewCounterVec(prometheus.CounterOpts{
			Namespace: namespace,
			Name:      "llm_tokens_total",
			Help:      "LLM tokens billed, by direction.",
		}, []string{"model", "kind"}),
	}
}

// MessageProcessed counts an acknowledged message.
func (m *Metrics) MessageProcessed(outcome string) { m.processed.WithLabelValues(outcome).Inc() }

// MessageFailed counts a failed handling attempt.
func (m *Metrics) MessageFailed(outcome string) { m.failed.WithLabelValues(outcome).Inc() }

// DuplicateEvent counts an event skipped as already processed.
func (m *Metrics) DuplicateEvent() { m.duplicates.Inc() }

// ObserveExtraction records extraction latency.
func (m *Metrics) ObserveExtraction(extractor, outcome string, d time.Duration) {
	m.extraction.WithLabelValues(extractor, outcome).Observe(d.Seconds())
}

// RecordLLMUsage records tokens and cost of one billed model request.
func (m *Metrics) RecordLLMUsage(model string, promptTokens, completionTokens, costMicros int64) {
	m.llmTokens.WithLabelValues(model, "prompt").Add(float64(promptTokens))
	m.llmTokens.WithLabelValues(model, "completion").Add(float64(completionTokens))
	m.llmCost.WithLabelValues(model).Add(float64(costMicros))
}
