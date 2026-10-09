package llm

import (
	"testing"

	"github.com/Dudomon/ledgerflow/services/worker/internal/extract"
)

func str(s string) *string { return &s }

func TestToField(t *testing.T) {
	tests := []struct {
		name     string
		in       modelField
		validate validator
		want     extract.Field
	}{
		{
			name:     "valid CNPJ is normalised to digits",
			in:       modelField{Value: str("11.222.333/0001-81"), Confidence: 0.9},
			validate: validTaxID,
			want:     extract.Field{Value: "11222333000181", Confidence: 0.9, Source: extract.SourceLLM},
		},
		{
			name:     "CNPJ with wrong check digit keeps raw value at capped confidence",
			in:       modelField{Value: str("11.222.333/0001-82"), Confidence: 0.99},
			validate: validTaxID,
			want:     extract.Field{Value: "11.222.333/0001-82", Confidence: invalidConfidenceCap, Source: extract.SourceLLM},
		},
		{
			name:     "foreign VAT id passes through",
			in:       modelField{Value: str("DE811569869"), Confidence: 0.8},
			validate: validTaxID,
			want:     extract.Field{Value: "DE811569869", Confidence: 0.8, Source: extract.SourceLLM},
		},
		{
			name:     "brazilian date format is invalid",
			in:       modelField{Value: str("30/09/2026"), Confidence: 0.95},
			validate: validDate,
			want:     extract.Field{Value: "30/09/2026", Confidence: invalidConfidenceCap, Source: extract.SourceLLM},
		},
		{
			name:     "impossible date is invalid",
			in:       modelField{Value: str("2026-02-30"), Confidence: 0.95},
			validate: validDate,
			want:     extract.Field{Value: "2026-02-30", Confidence: invalidConfidenceCap, Source: extract.SourceLLM},
		},
		{
			name:     "amount is normalised to two decimals",
			in:       modelField{Value: str("1530.4"), Confidence: 0.9},
			validate: validMoney,
			want:     extract.Field{Value: "1530.40", Confidence: 0.9, Source: extract.SourceLLM},
		},
		{
			name:     "brazilian number format is invalid",
			in:       modelField{Value: str("1.530,40"), Confidence: 0.9},
			validate: validMoney,
			want:     extract.Field{Value: "1.530,40", Confidence: invalidConfidenceCap, Source: extract.SourceLLM},
		},
		{
			name:     "exponent notation is invalid",
			in:       modelField{Value: str("1.5e3"), Confidence: 0.9},
			validate: validMoney,
			want:     extract.Field{Value: "1.5e3", Confidence: invalidConfidenceCap, Source: extract.SourceLLM},
		},
		{
			name:     "negative amount is invalid",
			in:       modelField{Value: str("-10.00"), Confidence: 0.9},
			validate: validMoney,
			want:     extract.Field{Value: "-10.00", Confidence: invalidConfidenceCap, Source: extract.SourceLLM},
		},
		{
			name:     "low confidence is not raised by the cap",
			in:       modelField{Value: str("R$1"), Confidence: 0.1},
			validate: validCurrency,
			want:     extract.Field{Value: "R$1", Confidence: 0.1, Source: extract.SourceLLM},
		},
		{
			name: "out of range confidence is clamped",
			in:   modelField{Value: str("Acme"), Confidence: 1.4},
			want: extract.Field{Value: "Acme", Confidence: 1, Source: extract.SourceLLM},
		},
		{
			name:     "null stays null",
			in:       modelField{Value: nil, Confidence: 0.9},
			validate: validDate,
			want:     extract.Field{Value: nil, Confidence: 0.9, Source: extract.SourceLLM},
		},
		{
			name: "blank becomes null",
			in:   modelField{Value: str("   "), Confidence: 0.6},
			want: extract.Field{Value: nil, Confidence: 0.6, Source: extract.SourceLLM},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := toField(tt.in, tt.validate); got != tt.want {
				t.Fatalf("toField = %#v, want %#v", got, tt.want)
			}
		})
	}
}

func TestNormalizeDocTypeAndLineItems(t *testing.T) {
	var out modelOutput
	out.NFeAccessKey = modelField{Value: str("35260911222333000181550010000123451473829106"), Confidence: 0.9}
	out.LineItems.Confidence = 0.9
	out.LineItems.Value = []modelLineItem{
		{Description: "ok", Quantity: "1", UnitPrice: "10", Total: "10"},
		{Description: "bad", Quantity: "1", UnitPrice: "10,00", Total: "10"},
	}

	fields, docType := normalize(out)

	if docType != extract.DocTypeInvoice {
		t.Errorf("docType = %q; an invalid access key must not make a document an NF-e", docType)
	}
	if got := fields[extract.KeyNFeAccessKey].Confidence; got != invalidConfidenceCap {
		t.Errorf("access key confidence = %v, want %v", got, invalidConfidenceCap)
	}
	li := fields[extract.KeyLineItems]
	items, ok := li.Value.([]extract.LineItem)
	if !ok || len(items) != 2 || items[0].Total != "10.00" || items[1].UnitPrice != "10,00" {
		t.Errorf("line items = %#v", li.Value)
	}
	if li.Confidence != invalidConfidenceCap {
		t.Errorf("line items confidence = %v, want %v", li.Confidence, invalidConfidenceCap)
	}
}

func TestNormalizeEmptyLineItemsIsEmptyList(t *testing.T) {
	fields, _ := normalize(modelOutput{})
	items, ok := fields[extract.KeyLineItems].Value.([]extract.LineItem)
	if !ok || items == nil || len(items) != 0 {
		t.Fatalf("line items = %#v, want empty non-nil slice (serialises as [])", fields[extract.KeyLineItems].Value)
	}
}
