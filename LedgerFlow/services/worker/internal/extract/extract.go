// Package extract defines the extraction domain model shared by the rule-based
// and LLM-backed extractors, plus the router that chooses between them.
package extract

import (
	"bytes"
	"context"
	"errors"
	"mime"
	"strings"
)

var (
	// ErrNotApplicable means the extractor does not understand this document
	// (e.g. well-formed XML that is not an NF-e). Routers may try another one.
	ErrNotApplicable = errors.New("document not applicable to extractor")

	// ErrUnsupportedContentType means no extractor can handle the content
	// type. Retrying cannot fix it.
	ErrUnsupportedContentType = errors.New("unsupported content type")
)

// Document is the raw input to an extractor.
type Document struct {
	Content     []byte
	ContentType string
}

// Extractor turns a document into contract fields.
type Extractor interface {
	Extract(ctx context.Context, doc Document) (Result, error)
}

// Document types defined by the service contract.
const (
	DocTypeInvoice = "invoice"
	DocTypeNFe     = "nfe"
)

// Field keys defined by the service contract.
const (
	KeySupplierName  = "supplier_name"
	KeySupplierTaxID = "supplier_tax_id"
	KeyInvoiceNumber = "invoice_number"
	KeyIssueDate     = "issue_date"
	KeyDueDate       = "due_date"
	KeyCurrency      = "currency"
	KeyTotalAmount   = "total_amount"
	KeyTaxAmount     = "tax_amount"
	KeyNFeAccessKey  = "nfe_access_key"
	KeyLineItems     = "line_items"
)

// Source records who produced a field value.
type Source string

// Sources produced by the worker; "human" is only ever written by the API.
const (
	SourceRule Source = "rule"
	SourceLLM  Source = "llm"
)

// Field is the contract's FieldValue wrapper.
type Field struct {
	Value      any     `json:"value"`
	Confidence float64 `json:"confidence"`
	Source     Source  `json:"source"`
}

// Fields maps contract keys to wrapped values.
type Fields map[string]Field

// CapConfidence lowers the confidence of key to at most limit. It is how
// validators signal "the value is suspicious, send it to a human" without
// discarding what was read.
func (f Fields) CapConfidence(key string, limit float64) {
	if v, ok := f[key]; ok && v.Confidence > limit {
		v.Confidence = limit
		f[key] = v
	}
}

// LineItem amounts are decimal strings, never floats, per the contract.
type LineItem struct {
	Description string `json:"description"`
	Quantity    string `json:"quantity"`
	UnitPrice   string `json:"unit_price"`
	Total       string `json:"total"`
}

// Result is the outcome of a successful extraction.
type Result struct {
	DocType       string
	Fields        Fields
	Extractor     string // metrics label: which strategy produced the result
	Model         string // empty for rule-based results
	PromptVersion string // empty for rule-based results
	CostUSDMicros int64
}

// MediaType returns the lower-cased media type without parameters, so that
// "Application/XML; charset=utf-8" and "application/xml" route identically.
func MediaType(contentType string) string {
	mt, _, err := mime.ParseMediaType(contentType)
	if err != nil {
		return strings.ToLower(strings.TrimSpace(contentType))
	}
	return mt
}

// IsXMLMediaType reports whether the declared media type is XML.
func IsXMLMediaType(mt string) bool {
	return mt == "application/xml" || mt == "text/xml" || strings.HasSuffix(mt, "+xml")
}

// LooksLikeXML sniffs content regardless of its declared type.
func LooksLikeXML(content []byte) bool {
	head := content
	if len(head) > 512 {
		head = head[:512]
	}
	head = bytes.TrimLeft(head, " \t\r\nï»¿")
	return bytes.HasPrefix(head, []byte("<"))
}
