package pipeline

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/Dudomon/ledgerflow/services/worker/internal/apiclient"
	"github.com/Dudomon/ledgerflow/services/worker/internal/extract"
	"github.com/Dudomon/ledgerflow/services/worker/internal/queue"
	"github.com/Dudomon/ledgerflow/services/worker/internal/storage"
)

const (
	tenantID   = "5a1c2e3d-4b5f-4a6b-8c7d-9e0f1a2b3c4d"
	documentID = "0b6f7a2e-6f43-4a7e-a1b1-2f3f8f1d9c11"
	eventID    = "8d0f8c52-3c1e-4c4b-9f5e-0f8a4b1f6a10"
	digest     = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
)

func validEvent() Event {
	return Event{
		EventID:     eventID,
		EventType:   "document.uploaded",
		OccurredAt:  time.Date(2026, 10, 9, 12, 0, 0, 0, time.UTC),
		TenantID:    tenantID,
		DocumentID:  documentID,
		StorageKey:  fmt.Sprintf("tenants/%s/documents/%s/%s", tenantID, documentID, digest),
		ContentType: "application/pdf",
		SHA256:      digest,
		Attempt:     1,
		TraceParent: "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
	}
}

func delivery(t *testing.T, ev Event) queue.Delivery {
	t.Helper()
	b, err := json.Marshal(ev)
	if err != nil {
		t.Fatal(err)
	}
	return queue.Delivery{MessageID: ev.EventID, Body: b, Attempt: 1}
}

type fakeStore struct {
	err   error
	calls int
}

func (f *fakeStore) Fetch(context.Context, string, string) ([]byte, error) {
	f.calls++
	return []byte("%PDF-1.7"), f.err
}

type fakeExtractor struct {
	res   extract.Result
	err   error
	calls int
}

func (f *fakeExtractor) Extract(context.Context, extract.Document) (extract.Result, error) {
	f.calls++
	return f.res, f.err
}

type fakeAPI struct {
	statusErr   error
	submitErr   error
	statuses    []apiclient.StatusUpdate
	submissions []apiclient.ExtractionResult
}

func (f *fakeAPI) SubmitExtraction(_ context.Context, r apiclient.ExtractionResult) error {
	f.submissions = append(f.submissions, r)
	return f.submitErr
}

func (f *fakeAPI) UpdateStatus(_ context.Context, _ string, u apiclient.StatusUpdate) error {
	f.statuses = append(f.statuses, u)
	return f.statusErr
}

type fakeDedup map[string]bool

func (f fakeDedup) Seen(id string) bool { return f[id] }
func (f fakeDedup) Add(id string)       { f[id] = true }

type nopMetrics struct{}

func (nopMetrics) ObserveExtraction(string, string, time.Duration) {}
func (nopMetrics) DuplicateEvent()                                 {}

type harness struct {
	store     *fakeStore
	extractor *fakeExtractor
	api       *fakeAPI
	dedup     fakeDedup
	proc      *Processor
}

func newHarness() *harness {
	h := &harness{
		store: &fakeStore{},
		extractor: &fakeExtractor{res: extract.Result{
			DocType:       extract.DocTypeInvoice,
			Fields:        extract.Fields{extract.KeyCurrency: {Value: "BRL", Confidence: 0.9, Source: extract.SourceLLM}},
			Extractor:     "llm",
			Model:         "gpt-4o-mini-2024-07-18",
			PromptVersion: "invoice-v3",
			CostUSDMicros: 270,
		}},
		api:   &fakeAPI{},
		dedup: fakeDedup{},
	}
	h.proc = New(h.store, h.extractor, h.api, h.dedup, nopMetrics{}, slog.New(slog.NewTextHandler(io.Discard, nil)))
	return h
}

func TestHandleSuccess(t *testing.T) {
	h := newHarness()

	if err := h.proc.Handle(context.Background(), delivery(t, validEvent())); err != nil {
		t.Fatalf("Handle: %v", err)
	}

	if len(h.api.statuses) != 1 || h.api.statuses[0].Status != apiclient.StatusProcessing {
		t.Fatalf("statuses = %+v, want [processing]", h.api.statuses)
	}
	if len(h.api.submissions) != 1 {
		t.Fatalf("submissions = %d, want 1", len(h.api.submissions))
	}
	got := h.api.submissions[0]
	if got.EventID != eventID || got.DocumentID != documentID || got.TenantID != tenantID ||
		got.Model == nil || *got.Model != "gpt-4o-mini-2024-07-18" ||
		got.PromptVersion == nil || *got.PromptVersion != "invoice-v3" || got.CostUSDMicros != 270 {
		t.Fatalf("submission = %+v", got)
	}
	if !h.dedup[eventID] {
		t.Fatal("event not recorded as processed")
	}
}

func TestHandleRuleResultSubmitsNullModel(t *testing.T) {
	h := newHarness()
	h.extractor.res = extract.Result{DocType: extract.DocTypeNFe, Fields: extract.Fields{}, Extractor: "nfe-rules"}

	if err := h.proc.Handle(context.Background(), delivery(t, validEvent())); err != nil {
		t.Fatalf("Handle: %v", err)
	}
	if s := h.api.submissions[0]; s.Model != nil || s.PromptVersion != nil {
		t.Fatalf("rule result must submit null model/prompt_version, got %v/%v", s.Model, s.PromptVersion)
	}
}

func TestHandleDuplicateSkipsWork(t *testing.T) {
	h := newHarness()
	h.dedup[eventID] = true

	if err := h.proc.Handle(context.Background(), delivery(t, validEvent())); err != nil {
		t.Fatalf("Handle: %v", err)
	}
	if h.store.calls != 0 || h.extractor.calls != 0 || len(h.api.submissions) != 0 {
		t.Fatal("duplicate event must not be reprocessed")
	}
}

func TestHandleErrorClassification(t *testing.T) {
	tests := []struct {
		name          string
		mutateEvent   func(*Event)
		setup         func(*harness)
		wantPermanent bool
		wantCached    bool
	}{
		{
			name:          "storage key for another tenant",
			mutateEvent:   func(e *Event) { e.StorageKey = "tenants/other/documents/x/" + digest },
			wantPermanent: true,
		},
		{
			name:          "unknown event type",
			mutateEvent:   func(e *Event) { e.EventType = "document.deleted" },
			wantPermanent: true,
		},
		{
			name:          "checksum mismatch",
			setup:         func(h *harness) { h.store.err = fmt.Errorf("wrapped: %w", storage.ErrChecksumMismatch) },
			wantPermanent: true,
		},
		{
			name:          "unsupported content",
			setup:         func(h *harness) { h.extractor.err = extract.ErrUnsupportedContentType },
			wantPermanent: true,
		},
		{
			name:  "model outage is retried",
			setup: func(h *harness) { h.extractor.err = errors.New("llm: provider returned 503") },
		},
		{
			name: "api unavailable on submit is retried and not cached",
			setup: func(h *harness) {
				h.api.submitErr = &apiclient.StatusError{StatusCode: http.StatusServiceUnavailable}
			},
		},
		{
			name: "api rejects payload permanently",
			setup: func(h *harness) {
				h.api.submitErr = &apiclient.StatusError{StatusCode: http.StatusUnprocessableEntity}
			},
			wantPermanent: true,
		},
		{
			name: "status conflict on processing is tolerated",
			setup: func(h *harness) {
				h.api.statusErr = &apiclient.StatusError{StatusCode: http.StatusConflict}
			},
			wantCached: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := newHarness()
			if tt.setup != nil {
				tt.setup(h)
			}
			ev := validEvent()
			if tt.mutateEvent != nil {
				tt.mutateEvent(&ev)
			}

			err := h.proc.Handle(context.Background(), delivery(t, ev))

			if tt.wantCached {
				if err != nil || !h.dedup[eventID] {
					t.Fatalf("err = %v, cached = %v; want success", err, h.dedup[eventID])
				}
				return
			}
			if err == nil {
				t.Fatal("expected an error")
			}
			if got := errors.Is(err, queue.ErrPermanent); got != tt.wantPermanent {
				t.Fatalf("permanent = %v, want %v (err: %v)", got, tt.wantPermanent, err)
			}
			if h.dedup[eventID] {
				t.Fatal("failed event must not be cached as processed")
			}
		})
	}
}

func TestDeadLetteredMarksDocumentFailed(t *testing.T) {
	h := newHarness()
	d := delivery(t, validEvent())
	d.Attempt = 4

	h.proc.DeadLettered(context.Background(), d, errors.New(strings.Repeat("x", 2000)))

	if len(h.api.statuses) != 1 {
		t.Fatalf("statuses = %+v", h.api.statuses)
	}
	got := h.api.statuses[0]
	if got.Status != apiclient.StatusFailed || len(got.Error) > maxStatusError ||
		!strings.HasPrefix(got.Error, "extraction failed after 4 attempt(s)") {
		t.Fatalf("status update = %+v", got)
	}
}

func TestDeadLetteredIgnoresUndecodableMessage(t *testing.T) {
	h := newHarness()
	h.proc.DeadLettered(context.Background(), queue.Delivery{Body: []byte("not json")}, errors.New("boom"))
	if len(h.api.statuses) != 0 {
		t.Fatal("must not update status without a valid document id")
	}
}
