package llm

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Dudomon/ledgerflow/services/worker/internal/backoff"
	"github.com/Dudomon/ledgerflow/services/worker/internal/extract"
)

const (
	testModel    = "gpt-4o-mini"
	servedModel  = "gpt-4o-mini-2024-07-18"
	testAPIKey   = "test-key"
	validContent = `{
		"supplier_name":   {"value": "Ferragens Boa Vista Ltda", "confidence": 0.97},
		"supplier_tax_id": {"value": "11.222.333/0001-81", "confidence": 0.95},
		"invoice_number":  {"value": "12345", "confidence": 0.9},
		"issue_date":      {"value": "2026-09-30", "confidence": 0.93},
		"due_date":        {"value": null, "confidence": 0.8},
		"currency":        {"value": "brl", "confidence": 0.99},
		"total_amount":    {"value": "414.6", "confidence": 0.96},
		"tax_amount":      {"value": "98.27", "confidence": 0.7},
		"nfe_access_key":  {"value": "3526 0911 2223 3300 0181 5500 1000 0123 4514 7382 9105", "confidence": 0.9},
		"line_items":      {"value": [{"description": "PARAFUSO SEXTAVADO", "quantity": "200.0000", "unit_price": "0.85", "total": "170"}], "confidence": 0.85}
	}`
)

type usageCall struct {
	model                    string
	prompt, completion, cost int64
}

type fakeUsage struct {
	mu    sync.Mutex
	calls []usageCall
}

func (f *fakeUsage) RecordLLMUsage(model string, prompt, completion, cost int64) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.calls = append(f.calls, usageCall{model: model, prompt: prompt, completion: completion, cost: cost})
}

func completionBody(t *testing.T, content string) []byte {
	t.Helper()
	b, err := json.Marshal(map[string]any{
		"id":    "chatcmpl-test",
		"model": servedModel,
		"choices": []any{map[string]any{
			"index":         0,
			"message":       map[string]any{"role": "assistant", "content": content, "refusal": nil},
			"finish_reason": "stop",
		}},
		"usage": map[string]any{"prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200},
	})
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func newTestClient(t *testing.T, baseURL string, usage UsageRecorder) *Client {
	t.Helper()
	c, err := New(Config{
		BaseURL:     baseURL,
		APIKey:      testAPIKey,
		Model:       testModel,
		Timeout:     2 * time.Second,
		MaxAttempts: 3,
		Backoff:     backoff.Policy{Base: time.Millisecond, Max: 5 * time.Millisecond},
	}, usage, slog.New(slog.NewTextHandler(io.Discard, nil)))
	if err != nil {
		t.Fatal(err)
	}
	return c
}

var pdfDoc = extract.Document{ContentType: "application/pdf", Content: []byte("%PDF-1.7 test")}

func TestExtractSuccess(t *testing.T) {
	body := completionBody(t, validContent)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/v1/chat/completions" {
			t.Errorf("unexpected request %s %s", r.Method, r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer "+testAPIKey {
			t.Errorf("Authorization = %q", got)
		}
		var req struct {
			Model          string `json:"model"`
			ResponseFormat struct {
				Type       string `json:"type"`
				JSONSchema struct {
					Name   string          `json:"name"`
					Strict bool            `json:"strict"`
					Schema json.RawMessage `json:"schema"`
				} `json:"json_schema"`
			} `json:"response_format"`
			Messages []struct {
				Role    string          `json:"role"`
				Content json.RawMessage `json:"content"`
			} `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode request: %v", err)
		}
		if req.Model != testModel || req.ResponseFormat.Type != "json_schema" ||
			!req.ResponseFormat.JSONSchema.Strict || req.ResponseFormat.JSONSchema.Name != schemaName ||
			len(req.ResponseFormat.JSONSchema.Schema) == 0 {
			t.Errorf("structured output not requested correctly: %+v", req.ResponseFormat)
		}
		if len(req.Messages) != 2 || req.Messages[0].Role != "system" {
			t.Errorf("messages = %+v", req.Messages)
			http.Error(w, "bad request shape", http.StatusBadRequest)
			return
		}
		var parts []contentPart
		if err := json.Unmarshal(req.Messages[1].Content, &parts); err != nil {
			t.Errorf("user content: %v", err)
		}
		if len(parts) != 2 || parts[1].Type != "file" || parts[1].File == nil ||
			parts[1].File.FileData != "data:application/pdf;base64,JVBERi0xLjcgdGVzdA==" {
			t.Errorf("pdf not attached as file part: %+v", parts)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(body)
	}))
	defer srv.Close()

	usage := &fakeUsage{}
	res, err := newTestClient(t, srv.URL+"/v1", usage).Extract(context.Background(), pdfDoc)
	if err != nil {
		t.Fatalf("Extract: %v", err)
	}

	// 1000 prompt tokens * 0.15 + 200 completion tokens * 0.60 = 270 micro-USD.
	if res.Model != servedModel || res.PromptVersion != PromptVersion || res.CostUSDMicros != 270 {
		t.Fatalf("metadata = model %q prompt %q cost %d", res.Model, res.PromptVersion, res.CostUSDMicros)
	}
	if res.DocType != extract.DocTypeNFe {
		t.Fatalf("DocType = %q, want nfe (valid access key present)", res.DocType)
	}
	wantValues := map[string]any{
		extract.KeySupplierTaxID: "11222333000181",
		extract.KeyCurrency:      "BRL",
		extract.KeyTotalAmount:   "414.60",
		extract.KeyDueDate:       nil,
		extract.KeyNFeAccessKey:  "35260911222333000181550010000123451473829105",
	}
	for k, want := range wantValues {
		if got := res.Fields[k]; got.Value != want || got.Source != extract.SourceLLM {
			t.Errorf("%s = %#v (%s), want %#v (llm)", k, got.Value, got.Source, want)
		}
	}
	if got := res.Fields[extract.KeySupplierName].Confidence; got != 0.97 {
		t.Errorf("model confidence not preserved: %v", got)
	}
	items, ok := res.Fields[extract.KeyLineItems].Value.([]extract.LineItem)
	if !ok || len(items) != 1 || items[0].Quantity != "200" || items[0].Total != "170.00" {
		t.Errorf("line items = %#v", res.Fields[extract.KeyLineItems].Value)
	}
	if len(usage.calls) != 1 || usage.calls[0] != (usageCall{model: servedModel, prompt: 1000, completion: 200, cost: 270}) {
		t.Errorf("usage = %+v", usage.calls)
	}
}

func TestExtractRetriesRateLimit(t *testing.T) {
	body := completionBody(t, validContent)
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if calls.Add(1) == 1 {
			w.Header().Set("Retry-After", "0")
			http.Error(w, `{"error":{"message":"rate limited"}}`, http.StatusTooManyRequests)
			return
		}
		_, _ = w.Write(body)
	}))
	defer srv.Close()

	if _, err := newTestClient(t, srv.URL, &fakeUsage{}).Extract(context.Background(), pdfDoc); err != nil {
		t.Fatalf("Extract: %v", err)
	}
	if got := calls.Load(); got != 2 {
		t.Fatalf("calls = %d, want 2", got)
	}
}

func TestExtractFailures(t *testing.T) {
	tests := []struct {
		name      string
		status    int
		content   string
		wantErr   error
		wantCode  int
		wantCalls int32
		wantUsage int
	}{
		{
			name:      "malformed model json is not retried in-process",
			status:    http.StatusOK,
			content:   `{"supplier_name": {"value": "Ferr`,
			wantErr:   ErrMalformedOutput,
			wantCalls: 1,
			wantUsage: 1, // the tokens were still billed
		},
		{
			name:      "fields outside the schema are rejected",
			status:    http.StatusOK,
			content:   `{"supplier_name": {"value": "x", "confidence": 1}, "surprise": true}`,
			wantErr:   ErrMalformedOutput,
			wantCalls: 1,
			wantUsage: 1,
		},
		{
			name:      "server errors exhaust attempts",
			status:    http.StatusServiceUnavailable,
			wantCode:  http.StatusServiceUnavailable,
			wantCalls: 3,
		},
		{
			name:      "bad request is not retried",
			status:    http.StatusBadRequest,
			wantCode:  http.StatusBadRequest,
			wantCalls: 1,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body := completionBody(t, tt.content)
			var calls atomic.Int32
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				calls.Add(1)
				if tt.status != http.StatusOK {
					http.Error(w, `{"error":{"message":"nope"}}`, tt.status)
					return
				}
				_, _ = w.Write(body)
			}))
			defer srv.Close()

			usage := &fakeUsage{}
			_, err := newTestClient(t, srv.URL, usage).Extract(context.Background(), pdfDoc)
			if err == nil {
				t.Fatal("expected an error")
			}
			if tt.wantErr != nil && !errors.Is(err, tt.wantErr) {
				t.Fatalf("error = %v, want %v", err, tt.wantErr)
			}
			if tt.wantCode != 0 {
				var se *StatusError
				if !errors.As(err, &se) || se.StatusCode != tt.wantCode {
					t.Fatalf("error = %v, want StatusError %d", err, tt.wantCode)
				}
			}
			if got := calls.Load(); got != tt.wantCalls {
				t.Fatalf("calls = %d, want %d", got, tt.wantCalls)
			}
			if len(usage.calls) != tt.wantUsage {
				t.Fatalf("usage records = %d, want %d", len(usage.calls), tt.wantUsage)
			}
		})
	}
}

func TestExtractRejectsUnsupportedContent(t *testing.T) {
	c := newTestClient(t, "http://127.0.0.1:1", &fakeUsage{})
	_, err := c.Extract(context.Background(), extract.Document{ContentType: "application/zip", Content: []byte("PK")})
	if !errors.Is(err, extract.ErrUnsupportedContentType) {
		t.Fatalf("error = %v, want ErrUnsupportedContentType", err)
	}
}

func TestSchemaCoversContractFields(t *testing.T) {
	var schema struct {
		Required             []string `json:"required"`
		AdditionalProperties bool     `json:"additionalProperties"`
	}
	if err := json.Unmarshal(responseSchema, &schema); err != nil {
		t.Fatalf("schema is not valid JSON: %v", err)
	}
	want := []string{
		extract.KeySupplierName, extract.KeySupplierTaxID, extract.KeyInvoiceNumber,
		extract.KeyIssueDate, extract.KeyDueDate, extract.KeyCurrency, extract.KeyTotalAmount,
		extract.KeyTaxAmount, extract.KeyNFeAccessKey, extract.KeyLineItems,
	}
	if len(schema.Required) != len(want) || schema.AdditionalProperties {
		t.Fatalf("schema required = %v, additionalProperties = %v", schema.Required, schema.AdditionalProperties)
	}
	got := make(map[string]bool, len(schema.Required))
	for _, k := range schema.Required {
		got[k] = true
	}
	for _, k := range want {
		if !got[k] {
			t.Errorf("schema does not require %q", k)
		}
	}
}
