// Package apiclient calls the LedgerFlow API's internal, service-token
// protected endpoints.
package apiclient

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"time"

	"github.com/Dudomon/ledgerflow/services/worker/internal/backoff"
	"github.com/Dudomon/ledgerflow/services/worker/internal/extract"
)

const errorBodyBytes = 512

// Status is a document lifecycle state the worker may set.
type Status string

// Statuses the worker reports; extracted/needs_review are decided by the API
// from the submitted confidences and the tenant threshold.
const (
	StatusProcessing Status = "processing"
	StatusFailed     Status = "failed"
)

// ExtractionResult is the body of POST /internal/extractions.
type ExtractionResult struct {
	EventID       string         `json:"event_id"`
	DocumentID    string         `json:"document_id"`
	TenantID      string         `json:"tenant_id"`
	DocType       string         `json:"doc_type"`
	Model         *string        `json:"model"`
	PromptVersion *string        `json:"prompt_version"`
	Fields        extract.Fields `json:"fields"`
	CostUSDMicros int64          `json:"cost_usd_micros"`
	LatencyMS     int64          `json:"latency_ms"`
}

// StatusUpdate is the body of POST /internal/documents/{id}/status.
type StatusUpdate struct {
	Status Status `json:"status"`
	Error  string `json:"error,omitempty"`
}

// StatusError is an unexpected HTTP status from the API.
type StatusError struct {
	Method     string
	Path       string
	StatusCode int
	Body       string
	RetryAfter time.Duration
}

// Error implements error.
func (e *StatusError) Error() string {
	return fmt.Sprintf("api: %s %s returned %d: %s", e.Method, e.Path, e.StatusCode, e.Body)
}

// IsPermanent reports whether err is an API rejection that will not change
// on retry (malformed or unknown document, conflicting state).
func IsPermanent(err error) bool {
	var se *StatusError
	if !errors.As(err, &se) {
		return false
	}
	switch se.StatusCode {
	case http.StatusBadRequest, http.StatusNotFound, http.StatusConflict,
		http.StatusRequestEntityTooLarge, http.StatusUnprocessableEntity:
		return true
	default:
		return false
	}
}

// IsConflict reports whether the API rejected a state transition.
func IsConflict(err error) bool {
	var se *StatusError
	return errors.As(err, &se) && se.StatusCode == http.StatusConflict
}

// Config configures the client.
type Config struct {
	BaseURL      string // including the /api/v1 prefix
	ServiceToken string
	MaxAttempts  int
	Backoff      backoff.Policy
	HTTPClient   *http.Client // carries timeouts and tracing transport
}

// Client is safe for concurrent use.
type Client struct {
	base *url.URL
	cfg  Config
	http *http.Client
	log  *slog.Logger
}

// New validates cfg and returns a client.
func New(cfg Config, log *slog.Logger) (*Client, error) {
	base, err := url.Parse(cfg.BaseURL)
	if err != nil || base.Scheme == "" || base.Host == "" {
		return nil, fmt.Errorf("api: invalid base url %q", cfg.BaseURL)
	}
	if cfg.ServiceToken == "" {
		return nil, errors.New("api: service token is required")
	}
	if cfg.MaxAttempts < 1 {
		return nil, errors.New("api: max attempts must be at least 1")
	}
	if cfg.HTTPClient == nil {
		return nil, errors.New("api: http client is required")
	}
	return &Client{base: base, cfg: cfg, http: cfg.HTTPClient, log: log}, nil
}

// SubmitExtraction posts an extraction. The API deduplicates on event_id, so
// resubmitting after an ambiguous failure (timeout, lost response) is safe.
func (c *Client) SubmitExtraction(ctx context.Context, r ExtractionResult) error {
	return c.post(ctx, c.base.JoinPath("internal", "extractions"), r)
}

// UpdateStatus sets a document's lifecycle status.
func (c *Client) UpdateStatus(ctx context.Context, documentID string, u StatusUpdate) error {
	return c.post(ctx, c.base.JoinPath("internal", "documents", url.PathEscape(documentID), "status"), u)
}

func (c *Client) post(ctx context.Context, u *url.URL, body any) error {
	payload, err := json.Marshal(body)
	if err != nil {
		return fmt.Errorf("api: encode %s: %w", u.Path, err)
	}

	var lastErr error
	for attempt := 0; attempt < c.cfg.MaxAttempts; attempt++ {
		if attempt > 0 {
			delay := c.cfg.Backoff.Delay(attempt - 1)
			var se *StatusError
			if errors.As(lastErr, &se) && se.RetryAfter > delay {
				delay = se.RetryAfter
			}
			if err := backoff.Sleep(ctx, delay); err != nil {
				return fmt.Errorf("api: waiting to retry: %w (last error: %w)", err, lastErr)
			}
		}

		err := c.do(ctx, u, payload)
		if err == nil {
			return nil
		}
		if !retryable(err) || ctx.Err() != nil {
			return err
		}
		c.log.WarnContext(ctx, "api request failed; retrying", "path", u.Path, "attempt", attempt+1, "error", err)
		lastErr = err
	}
	return fmt.Errorf("api: giving up after %d attempts: %w", c.cfg.MaxAttempts, lastErr)
}

func (c *Client) do(ctx context.Context, u *url.URL, payload []byte) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, u.String(), bytes.NewReader(payload))
	if err != nil {
		return fmt.Errorf("api: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.cfg.ServiceToken)

	resp, err := c.http.Do(req)
	if err != nil {
		return fmt.Errorf("api: %s %s: %w", req.Method, u.Path, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		// Drain so the keep-alive connection can be reused.
		_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 64<<10))
		return nil
	}
	body, _ := io.ReadAll(io.LimitReader(resp.Body, errorBodyBytes))
	se := &StatusError{Method: req.Method, Path: u.Path, StatusCode: resp.StatusCode, Body: string(body)}
	if d, ok := backoff.ParseRetryAfter(resp.Header.Get("Retry-After"), time.Now()); ok {
		se.RetryAfter = d
	}
	return se
}

// retryable treats rate limits, server errors and transport failures as
// transient. 401/403 are retried too: they usually mean a token rotation is
// in progress, and the attempt budget bounds the cost if it is not.
func retryable(err error) bool {
	var se *StatusError
	if errors.As(err, &se) {
		return !IsPermanent(err)
	}
	return true
}
