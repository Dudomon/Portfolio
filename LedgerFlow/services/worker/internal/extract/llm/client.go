// Package llm extracts invoice fields with an OpenAI-compatible chat
// completions API using strict JSON-schema structured output.
package llm

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/Dudomon/ledgerflow/services/worker/internal/backoff"
	"github.com/Dudomon/ledgerflow/services/worker/internal/extract"
)

const (
	extractorName = "llm"
	schemaName    = "invoice_extraction"

	// maxResponseBytes bounds memory if a proxy misbehaves; a full invoice
	// extraction is a few kilobytes.
	maxResponseBytes = 4 << 20
	// maxTextBytes keeps XML sent as text well inside the context window.
	maxTextBytes = 256 << 10
	// errorBodyBytes is how much of a failed response is kept for logs.
	errorBodyBytes = 512
)

var (
	// ErrMalformedOutput means the provider answered but the content does not
	// match the schema (or is truncated). The queue retries it: generation is
	// non-deterministic and a second attempt usually succeeds.
	ErrMalformedOutput = errors.New("llm: malformed model output")

	// ErrRefused means the model declined to answer.
	ErrRefused = errors.New("llm: model refused")
)

// StatusError is a non-200 response from the provider.
type StatusError struct {
	StatusCode int
	Body       string
	RetryAfter time.Duration
}

// Error implements error.
func (e *StatusError) Error() string {
	return fmt.Sprintf("llm: provider returned %d: %s", e.StatusCode, e.Body)
}

// transportError marks network-level failures (including per-attempt
// timeouts), which are always worth retrying.
type transportError struct{ err error }

func (e *transportError) Error() string { return "llm: transport: " + e.err.Error() }
func (e *transportError) Unwrap() error { return e.err }

// UsageRecorder receives token usage and cost for every billed request,
// including ones whose output is later rejected: those are still paid for.
type UsageRecorder interface {
	RecordLLMUsage(model string, promptTokens, completionTokens, costMicros int64)
}

// Config configures the client.
type Config struct {
	BaseURL     string // e.g. https://api.openai.com/v1
	APIKey      string
	Model       string
	Timeout     time.Duration // per HTTP attempt
	MaxAttempts int
	Backoff     backoff.Policy
	Prices      PriceTable
	HTTPClient  *http.Client // optional; timeouts come from Timeout
}

// Client is an extract.Extractor backed by a chat completions endpoint.
type Client struct {
	cfg      Config
	endpoint string
	http     *http.Client
	usage    UsageRecorder
	log      *slog.Logger
}

// New validates cfg and returns a client.
func New(cfg Config, usage UsageRecorder, log *slog.Logger) (*Client, error) {
	u, err := url.Parse(cfg.BaseURL)
	if err != nil || u.Scheme == "" || u.Host == "" {
		return nil, fmt.Errorf("llm: invalid base url %q", cfg.BaseURL)
	}
	switch {
	case cfg.APIKey == "":
		return nil, errors.New("llm: api key is required")
	case cfg.Model == "":
		return nil, errors.New("llm: model is required")
	case cfg.Timeout <= 0:
		return nil, errors.New("llm: timeout must be positive")
	case cfg.MaxAttempts < 1:
		return nil, errors.New("llm: max attempts must be at least 1")
	case usage == nil:
		return nil, errors.New("llm: usage recorder is required")
	}
	if cfg.Prices == nil {
		cfg.Prices = DefaultPrices()
	}
	httpClient := cfg.HTTPClient
	if httpClient == nil {
		httpClient = &http.Client{}
	}
	return &Client{
		cfg:      cfg,
		endpoint: strings.TrimRight(cfg.BaseURL, "/") + "/chat/completions",
		http:     httpClient,
		usage:    usage,
		log:      log,
	}, nil
}

// Extract implements extract.Extractor.
func (c *Client) Extract(ctx context.Context, doc extract.Document) (extract.Result, error) {
	parts, err := userContent(doc)
	if err != nil {
		return extract.Result{}, err
	}
	payload, err := json.Marshal(chatRequest{
		Model: c.cfg.Model,
		Messages: []chatMessage{
			{Role: "system", Content: systemPrompt},
			{Role: "user", Content: parts},
		},
		ResponseFormat: responseFormat{
			Type: "json_schema",
			JSONSchema: jsonSchema{
				Name:   schemaName,
				Strict: true,
				Schema: json.RawMessage(responseSchema),
			},
		},
		Temperature: 0,
	})
	if err != nil {
		return extract.Result{}, fmt.Errorf("llm: encode request: %w", err)
	}

	resp, err := c.complete(ctx, payload)
	if err != nil {
		return extract.Result{}, err
	}

	// Providers answer with the resolved snapshot name; that is what the
	// extraction records and what pricing should match.
	model := resp.Model
	if model == "" {
		model = c.cfg.Model
	}
	cost, priced := c.cfg.Prices.CostMicros(model, resp.Usage.PromptTokens, resp.Usage.CompletionTokens)
	if !priced {
		c.log.WarnContext(ctx, "no price configured for model; cost recorded as zero", "model", model)
	}
	c.usage.RecordLLMUsage(model, resp.Usage.PromptTokens, resp.Usage.CompletionTokens, cost)

	out, err := parseOutput(resp)
	if err != nil {
		return extract.Result{}, err
	}
	fields, docType := normalize(out)
	return extract.Result{
		DocType:       docType,
		Fields:        fields,
		Extractor:     extractorName,
		Model:         model,
		PromptVersion: PromptVersion,
		CostUSDMicros: cost,
	}, nil
}

// complete sends the request, retrying rate limits, server errors and
// transport failures with jittered backoff. Other 4xx responses are returned
// immediately: they will fail the same way on every attempt.
func (c *Client) complete(ctx context.Context, payload []byte) (*chatResponse, error) {
	var lastErr error
	for attempt := 0; attempt < c.cfg.MaxAttempts; attempt++ {
		if attempt > 0 {
			delay := c.cfg.Backoff.Delay(attempt - 1)
			var se *StatusError
			if errors.As(lastErr, &se) && se.RetryAfter > delay {
				delay = se.RetryAfter
			}
			if err := backoff.Sleep(ctx, delay); err != nil {
				return nil, fmt.Errorf("llm: waiting to retry: %w (last error: %w)", err, lastErr)
			}
		}

		resp, err := c.send(ctx, payload)
		if err == nil {
			return resp, nil
		}
		if !retryable(err) || ctx.Err() != nil {
			return nil, err
		}
		c.log.WarnContext(ctx, "llm request failed; retrying", "attempt", attempt+1, "error", err)
		lastErr = err
	}
	return nil, fmt.Errorf("llm: giving up after %d attempts: %w", c.cfg.MaxAttempts, lastErr)
}

func (c *Client) send(ctx context.Context, payload []byte) (*chatResponse, error) {
	ctx, cancel := context.WithTimeout(ctx, c.cfg.Timeout)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpoint, bytes.NewReader(payload))
	if err != nil {
		return nil, fmt.Errorf("llm: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.cfg.APIKey)

	resp, err := c.http.Do(req)
	if err != nil {
		return nil, &transportError{err: err}
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(io.LimitReader(resp.Body, maxResponseBytes))
	if err != nil {
		return nil, &transportError{err: fmt.Errorf("read response: %w", err)}
	}
	if resp.StatusCode != http.StatusOK {
		se := &StatusError{StatusCode: resp.StatusCode, Body: truncate(string(body), errorBodyBytes)}
		if d, ok := backoff.ParseRetryAfter(resp.Header.Get("Retry-After"), time.Now()); ok {
			se.RetryAfter = d
		}
		return nil, se
	}

	var cr chatResponse
	if err := json.Unmarshal(body, &cr); err != nil {
		return nil, fmt.Errorf("%w: response envelope: %w", ErrMalformedOutput, err)
	}
	return &cr, nil
}

func retryable(err error) bool {
	var se *StatusError
	if errors.As(err, &se) {
		return se.StatusCode == http.StatusTooManyRequests || se.StatusCode >= http.StatusInternalServerError
	}
	var te *transportError
	return errors.As(err, &te)
}

func parseOutput(resp *chatResponse) (modelOutput, error) {
	if len(resp.Choices) == 0 {
		return modelOutput{}, fmt.Errorf("%w: no choices in response", ErrMalformedOutput)
	}
	choice := resp.Choices[0]
	if choice.Message.Refusal != "" {
		return modelOutput{}, fmt.Errorf("%w: %s", ErrRefused, truncate(choice.Message.Refusal, errorBodyBytes))
	}
	if choice.FinishReason == "length" {
		return modelOutput{}, fmt.Errorf("%w: output truncated at token limit", ErrMalformedOutput)
	}

	var out modelOutput
	dec := json.NewDecoder(strings.NewReader(choice.Message.Content))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&out); err != nil {
		return modelOutput{}, fmt.Errorf("%w: %w", ErrMalformedOutput, err)
	}
	return out, nil
}

// userContent builds the multimodal user message for the document.
func userContent(doc extract.Document) ([]contentPart, error) {
	instruction := contentPart{Type: "text", Text: "Extract the invoice fields from the attached document."}
	mt := extract.MediaType(doc.ContentType)

	switch {
	case mt == "image/png" || mt == "image/jpeg":
		return []contentPart{instruction, {
			Type:     "image_url",
			ImageURL: &imageURL{URL: dataURL(mt, doc.Content), Detail: "high"},
		}}, nil
	case mt == "application/pdf":
		return []contentPart{instruction, {
			Type: "file",
			File: &filePart{Filename: "invoice.pdf", FileData: dataURL(mt, doc.Content)},
		}}, nil
	case extract.IsXMLMediaType(mt) || extract.LooksLikeXML(doc.Content):
		if !utf8.Valid(doc.Content) {
			return nil, fmt.Errorf("%w: xml is not valid UTF-8", extract.ErrUnsupportedContentType)
		}
		if len(doc.Content) > maxTextBytes {
			return nil, fmt.Errorf("%w: xml of %d bytes exceeds the %d-byte model input budget",
				extract.ErrUnsupportedContentType, len(doc.Content), maxTextBytes)
		}
		return []contentPart{instruction, {Type: "text", Text: string(doc.Content)}}, nil
	default:
		return nil, fmt.Errorf("%w: %q", extract.ErrUnsupportedContentType, doc.ContentType)
	}
}

func dataURL(mediaType string, content []byte) string {
	return "data:" + mediaType + ";base64," + base64.StdEncoding.EncodeToString(content)
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	for n > 0 && !utf8.RuneStart(s[n]) {
		n--
	}
	return s[:n] + "…"
}

// Wire types for the chat completions API.

type chatRequest struct {
	Model          string         `json:"model"`
	Messages       []chatMessage  `json:"messages"`
	ResponseFormat responseFormat `json:"response_format"`
	Temperature    float64        `json:"temperature"`
}

type chatMessage struct {
	Role    string `json:"role"`
	Content any    `json:"content"` // string or []contentPart
}

type contentPart struct {
	Type     string    `json:"type"`
	Text     string    `json:"text,omitempty"`
	ImageURL *imageURL `json:"image_url,omitempty"`
	File     *filePart `json:"file,omitempty"`
}

type imageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`
}

type filePart struct {
	Filename string `json:"filename"`
	FileData string `json:"file_data"`
}

type responseFormat struct {
	Type       string     `json:"type"`
	JSONSchema jsonSchema `json:"json_schema"`
}

type jsonSchema struct {
	Name   string          `json:"name"`
	Strict bool            `json:"strict"`
	Schema json.RawMessage `json:"schema"`
}

type chatResponse struct {
	Model   string `json:"model"`
	Choices []struct {
		Message struct {
			Content string `json:"content"`
			Refusal string `json:"refusal"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int64 `json:"prompt_tokens"`
		CompletionTokens int64 `json:"completion_tokens"`
	} `json:"usage"`
}
