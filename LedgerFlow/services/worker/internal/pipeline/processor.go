// Package pipeline turns a document.uploaded event into a submitted
// extraction: fetch, verify, extract, report.
package pipeline

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"time"

	"github.com/Dudomon/ledgerflow/services/worker/internal/apiclient"
	"github.com/Dudomon/ledgerflow/services/worker/internal/extract"
	"github.com/Dudomon/ledgerflow/services/worker/internal/observability"
	"github.com/Dudomon/ledgerflow/services/worker/internal/queue"
	"github.com/Dudomon/ledgerflow/services/worker/internal/storage"
)

// maxStatusError bounds the failure reason stored on the document.
const maxStatusError = 500

// ObjectStore fetches a document and verifies its digest.
type ObjectStore interface {
	Fetch(ctx context.Context, key, sha256Hex string) ([]byte, error)
}

// API is the subset of the LedgerFlow API the pipeline reports to.
type API interface {
	SubmitExtraction(ctx context.Context, r apiclient.ExtractionResult) error
	UpdateStatus(ctx context.Context, documentID string, u apiclient.StatusUpdate) error
}

// Deduplicator is a local cache of processed event IDs.
type Deduplicator interface {
	Seen(eventID string) bool
	Add(eventID string)
}

// Metrics receives pipeline-level measurements.
type Metrics interface {
	ObserveExtraction(extractor, outcome string, d time.Duration)
	DuplicateEvent()
}

// Processor implements queue.Handler.
type Processor struct {
	store     ObjectStore
	extractor extract.Extractor
	api       API
	dedup     Deduplicator
	metrics   Metrics
	log       *slog.Logger
}

// New wires a processor.
func New(store ObjectStore, extractor extract.Extractor, api API, dedup Deduplicator, metrics Metrics, log *slog.Logger) *Processor {
	return &Processor{store: store, extractor: extractor, api: api, dedup: dedup, metrics: metrics, log: log}
}

// Handle implements queue.Handler. Errors wrapping queue.ErrPermanent go
// straight to the DLQ; anything else is retried with backoff.
func (p *Processor) Handle(ctx context.Context, d queue.Delivery) error {
	ev, err := decodeEvent(d.Body)
	if err != nil {
		return permanent(err)
	}
	ctx = observability.ContextWithTraceParent(ctx, ev.TraceParent)
	log := p.log.With("event_id", ev.EventID, "document_id", ev.DocumentID,
		"tenant_id", ev.TenantID, "attempt", d.Attempt)

	if p.dedup.Seen(ev.EventID) {
		p.metrics.DuplicateEvent()
		log.InfoContext(ctx, "event already processed by this replica; skipping")
		return nil
	}

	if err := p.api.UpdateStatus(ctx, ev.DocumentID, apiclient.StatusUpdate{Status: apiclient.StatusProcessing}); err != nil {
		// 409 means the document already moved past "processing" (e.g. a
		// redelivery after another replica finished). Submitting again is
		// still safe because the API deduplicates on event_id.
		if !apiclient.IsConflict(err) {
			return classify(fmt.Errorf("mark processing: %w", err))
		}
		log.InfoContext(ctx, "status transition rejected; continuing idempotently", "error", err)
	}

	content, err := p.store.Fetch(ctx, ev.StorageKey, ev.SHA256)
	if err != nil {
		return classify(fmt.Errorf("fetch document: %w", err))
	}

	start := time.Now()
	res, err := p.extractor.Extract(ctx, extract.Document{Content: content, ContentType: ev.ContentType})
	latency := time.Since(start)
	if err != nil {
		p.metrics.ObserveExtraction("none", "error", latency)
		return classify(fmt.Errorf("extract: %w", err))
	}
	p.metrics.ObserveExtraction(res.Extractor, "success", latency)

	err = p.api.SubmitExtraction(ctx, apiclient.ExtractionResult{
		EventID:       ev.EventID,
		DocumentID:    ev.DocumentID,
		TenantID:      ev.TenantID,
		DocType:       res.DocType,
		Model:         nullable(res.Model),
		PromptVersion: nullable(res.PromptVersion),
		Fields:        res.Fields,
		CostUSDMicros: res.CostUSDMicros,
		LatencyMS:     latency.Milliseconds(),
	})
	if err != nil {
		return classify(fmt.Errorf("submit extraction: %w", err))
	}

	// Only cache after the API has the result: caching earlier would turn a
	// failed submit into a silently dropped document on redelivery.
	p.dedup.Add(ev.EventID)
	log.InfoContext(ctx, "extraction submitted",
		"extractor", res.Extractor, "doc_type", res.DocType, "min_confidence", minConfidence(res.Fields),
		"cost_usd_micros", res.CostUSDMicros, "latency_ms", latency.Milliseconds())
	return nil
}

// DeadLettered implements queue.Handler by marking the document failed, so it
// surfaces in the UI and an admin can re-queue it.
func (p *Processor) DeadLettered(ctx context.Context, d queue.Delivery, cause error) {
	ev, err := decodeEvent(d.Body)
	if err != nil {
		// No trustworthy document ID to update; the DLQ copy is the record.
		p.log.WarnContext(ctx, "dead-lettered message is not a valid event", "error", err)
		return
	}
	ctx = observability.ContextWithTraceParent(ctx, ev.TraceParent)
	reason := fmt.Sprintf("extraction failed after %d attempt(s): %v", d.Attempt, cause)
	if len(reason) > maxStatusError {
		reason = strings.ToValidUTF8(reason[:maxStatusError], "")
	}
	update := apiclient.StatusUpdate{Status: apiclient.StatusFailed, Error: reason}
	if err := p.api.UpdateStatus(ctx, ev.DocumentID, update); err != nil {
		p.log.ErrorContext(ctx, "could not mark document failed", "document_id", ev.DocumentID, "error", err)
	}
}

// classify marks errors that will fail identically on every attempt.
func classify(err error) error {
	switch {
	case errors.Is(err, storage.ErrNotFound),
		errors.Is(err, storage.ErrChecksumMismatch),
		errors.Is(err, storage.ErrTooLarge),
		errors.Is(err, extract.ErrUnsupportedContentType),
		apiclient.IsPermanent(err):
		return permanent(err)
	default:
		return err
	}
}

func permanent(err error) error {
	return fmt.Errorf("%w: %w", queue.ErrPermanent, err)
}

func nullable(s string) *string {
	if s == "" {
		return nil
	}
	return &s
}

func minConfidence(f extract.Fields) float64 {
	if len(f) == 0 {
		return 0
	}
	lowest := 1.0
	for _, v := range f {
		lowest = min(lowest, v.Confidence)
	}
	return lowest
}
