package llm

import (
	"strings"
	"time"

	"github.com/shopspring/decimal"

	"github.com/Dudomon/ledgerflow/services/worker/internal/extract"
	"github.com/Dudomon/ledgerflow/services/worker/internal/extract/brtax"
)

// invalidConfidenceCap is the ceiling applied to a field whose value fails
// validation. It is deliberately far below any sensible review threshold:
// the model's self-reported confidence is not calibrated, but a CNPJ with a
// wrong check digit is wrong regardless of how sure the model claims to be.
const invalidConfidenceCap = 0.2

// modelField mirrors $defs/text_field in the response schema.
type modelField struct {
	Value      *string `json:"value"`
	Confidence float64 `json:"confidence"`
}

type modelLineItem struct {
	Description string `json:"description"`
	Quantity    string `json:"quantity"`
	UnitPrice   string `json:"unit_price"`
	Total       string `json:"total"`
}

// modelOutput mirrors the response schema.
type modelOutput struct {
	SupplierName  modelField `json:"supplier_name"`
	SupplierTaxID modelField `json:"supplier_tax_id"`
	InvoiceNumber modelField `json:"invoice_number"`
	IssueDate     modelField `json:"issue_date"`
	DueDate       modelField `json:"due_date"`
	Currency      modelField `json:"currency"`
	TotalAmount   modelField `json:"total_amount"`
	TaxAmount     modelField `json:"tax_amount"`
	NFeAccessKey  modelField `json:"nfe_access_key"`
	LineItems     struct {
		Value      []modelLineItem `json:"value"`
		Confidence float64         `json:"confidence"`
	} `json:"line_items"`
}

// validator normalises a raw value, reporting false when it is invalid.
type validator func(string) (string, bool)

// normalize converts model output into contract fields, validating every
// value the contract constrains. The doc type is derived from a verified
// access key rather than asked of the model, so it cannot be hallucinated.
func normalize(out modelOutput) (extract.Fields, string) {
	fields := extract.Fields{
		extract.KeySupplierName:  toField(out.SupplierName, nil),
		extract.KeySupplierTaxID: toField(out.SupplierTaxID, validTaxID),
		extract.KeyInvoiceNumber: toField(out.InvoiceNumber, nil),
		extract.KeyIssueDate:     toField(out.IssueDate, validDate),
		extract.KeyDueDate:       toField(out.DueDate, validDate),
		extract.KeyCurrency:      toField(out.Currency, validCurrency),
		extract.KeyTotalAmount:   toField(out.TotalAmount, validMoney),
		extract.KeyTaxAmount:     toField(out.TaxAmount, validMoney),
		extract.KeyNFeAccessKey:  toField(out.NFeAccessKey, validAccessKey),
		extract.KeyLineItems:     lineItemsField(out.LineItems.Value, out.LineItems.Confidence),
	}

	docType := extract.DocTypeInvoice
	if key, ok := fields[extract.KeyNFeAccessKey].Value.(string); ok {
		if _, err := brtax.ParseAccessKey(key); err == nil {
			docType = extract.DocTypeNFe
		}
	}
	return fields, docType
}

func toField(m modelField, validate validator) extract.Field {
	f := extract.Field{Confidence: clamp01(m.Confidence), Source: extract.SourceLLM}
	if m.Value == nil {
		return f
	}
	v := strings.TrimSpace(*m.Value)
	if v == "" {
		return f
	}
	f.Value = v
	if validate == nil {
		return f
	}
	norm, ok := validate(v)
	if !ok {
		// Keep the raw value: the reviewer corrects it faster than retyping.
		f.Confidence = min(f.Confidence, invalidConfidenceCap)
		return f
	}
	f.Value = norm
	return f
}

func lineItemsField(raw []modelLineItem, confidence float64) extract.Field {
	f := extract.Field{Confidence: clamp01(confidence), Source: extract.SourceLLM}
	items := make([]extract.LineItem, 0, len(raw))
	valid := true
	for _, r := range raw {
		item := extract.LineItem{
			Description: strings.TrimSpace(r.Description),
			Quantity:    strings.TrimSpace(r.Quantity),
			UnitPrice:   strings.TrimSpace(r.UnitPrice),
			Total:       strings.TrimSpace(r.Total),
		}
		qty, okQ := validDecimal(item.Quantity)
		unit, okU := validDecimal(item.UnitPrice)
		total, okT := validMoney(item.Total)
		if okQ && okU && okT {
			item.Quantity, item.UnitPrice, item.Total = qty, unit, total
		} else {
			valid = false
		}
		items = append(items, item)
	}
	f.Value = items
	if !valid {
		f.Confidence = min(f.Confidence, invalidConfidenceCap)
	}
	return f
}

func clamp01(v float64) float64 {
	return max(0, min(1, v))
}

func validDate(v string) (string, bool) {
	if _, err := time.Parse("2006-01-02", v); err != nil {
		return v, false
	}
	return v, true
}

func validCurrency(v string) (string, bool) {
	v = strings.ToUpper(v)
	if len(v) != 3 {
		return v, false
	}
	for i := 0; i < len(v); i++ {
		if v[i] < 'A' || v[i] > 'Z' {
			return v, false
		}
	}
	return v, true
}

// validDecimal accepts plain decimal strings only. decimal.NewFromString
// also accepts exponents ("1e3"), which the contract does not allow.
func validDecimal(v string) (string, bool) {
	if strings.ContainsAny(v, "eE") {
		return v, false
	}
	d, err := decimal.NewFromString(v)
	if err != nil || d.IsNegative() {
		return v, false
	}
	return d.String(), true
}

func validMoney(v string) (string, bool) {
	norm, ok := validDecimal(v)
	if !ok {
		return v, false
	}
	return decimal.RequireFromString(norm).StringFixed(2), true
}

// validTaxID checks CNPJs (anything that normalises to 14 digits) and passes
// foreign VAT identifiers through, since their formats are country-specific.
func validTaxID(v string) (string, bool) {
	digits := brtax.OnlyDigits(v)
	if len(digits) != 14 {
		return v, true
	}
	return digits, brtax.ValidCNPJ(digits)
}

func validAccessKey(v string) (string, bool) {
	digits := brtax.OnlyDigits(v)
	if _, err := brtax.ParseAccessKey(digits); err != nil {
		return v, false
	}
	return digits, true
}
