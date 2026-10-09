package apiclient

import (
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Dudomon/ledgerflow/services/worker/internal/backoff"
	"github.com/Dudomon/ledgerflow/services/worker/internal/extract"
)

func newTestClient(t *testing.T, baseURL string) *Client {
	t.Helper()
	c, err := New(Config{
		BaseURL:      baseURL + "/api/v1",
		ServiceToken: "svc-token",
		MaxAttempts:  3,
		Backoff:      backoff.Policy{Base: time.Millisecond, Max: 5 * time.Millisecond},
		HTTPClient:   &http.Client{Timeout: 2 * time.Second},
	}, slog.New(slog.NewTextHandler(io.Discard, nil)))
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func TestSubmitExtractionRetriesTransientFailure(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v1/internal/extractions" || r.Header.Get("Authorization") != "Bearer svc-token" {
			t.Errorf("unexpected request %s auth=%q", r.URL.Path, r.Header.Get("Authorization"))
		}
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode body: %v", err)
		}
		// Contract: model and prompt_version are explicit nulls for rule-based results.
		if v, ok := body["model"]; !ok || v != nil {
			t.Errorf("model = %v (present=%v), want explicit null", v, ok)
		}
		if calls.Add(1) == 1 {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		w.WriteHeader(http.StatusNoContent)
	}))
	defer srv.Close()

	err := newTestClient(t, srv.URL).SubmitExtraction(context.Background(), ExtractionResult{
		EventID:    "8d0f8c52-3c1e-4c4b-9f5e-0f8a4b1f6a10",
		DocumentID: "0b6f7a2e-6f43-4a7e-a1b1-2f3f8f1d9c11",
		TenantID:   "5a1c2e3d-4b5f-4a6b-8c7d-9e0f1a2b3c4d",
		DocType:    extract.DocTypeNFe,
		Fields:     extract.Fields{},
	})
	if err != nil {
		t.Fatalf("SubmitExtraction: %v", err)
	}
	if got := calls.Load(); got != 2 {
		t.Fatalf("calls = %d, want 2", got)
	}
}

func TestUpdateStatusPermanentFailureIsNotRetried(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		if r.URL.Path != "/api/v1/internal/documents/0b6f7a2e-6f43-4a7e-a1b1-2f3f8f1d9c11/status" {
			t.Errorf("path = %s", r.URL.Path)
		}
		w.WriteHeader(http.StatusConflict)
	}))
	defer srv.Close()

	err := newTestClient(t, srv.URL).UpdateStatus(context.Background(),
		"0b6f7a2e-6f43-4a7e-a1b1-2f3f8f1d9c11", StatusUpdate{Status: StatusProcessing})
	if !IsConflict(err) || !IsPermanent(err) {
		t.Fatalf("error = %v, want permanent conflict", err)
	}
	if got := calls.Load(); got != 1 {
		t.Fatalf("calls = %d, want 1", got)
	}
}
