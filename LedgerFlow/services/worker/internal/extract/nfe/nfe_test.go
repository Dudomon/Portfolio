package nfe

import (
	"context"
	"errors"
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/Dudomon/ledgerflow/services/worker/internal/extract"
)

const sampleKey = "35260911222333000181550010000123451473829105"

func loadSample(t *testing.T) string {
	t.Helper()
	b, err := os.ReadFile("testdata/nfe_valid.xml")
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func extractString(t *testing.T, xmlDoc string) extract.Result {
	t.Helper()
	res, err := NewParser().Extract(context.Background(), extract.Document{
		Content:     []byte(xmlDoc),
		ContentType: "application/xml",
	})
	if err != nil {
		t.Fatalf("Extract: %v", err)
	}
	return res
}

func TestParserSample(t *testing.T) {
	res := extractString(t, loadSample(t))

	if res.DocType != extract.DocTypeNFe {
		t.Fatalf("DocType = %q, want %q", res.DocType, extract.DocTypeNFe)
	}
	if res.Model != "" || res.PromptVersion != "" || res.CostUSDMicros != 0 {
		t.Fatalf("rule result must not carry model metadata: %+v", res)
	}

	want := map[string]any{
		extract.KeySupplierName:  "Ferragens Boa Vista Ltda",
		extract.KeySupplierTaxID: "11222333000181",
		extract.KeyInvoiceNumber: "12345",
		extract.KeyIssueDate:     "2026-09-30",
		extract.KeyDueDate:       "2026-10-30",
		extract.KeyCurrency:      "BRL",
		extract.KeyTotalAmount:   "414.60",
		extract.KeyTaxAmount:     "98.27",
		extract.KeyNFeAccessKey:  sampleKey,
		extract.KeyLineItems: []extract.LineItem{
			{Description: "PARAFUSO SEXTAVADO ACO INOX M8X40", Quantity: "200", UnitPrice: "0.85", Total: "170.00"},
			{Description: "PORCA SEXTAVADA ACO INOX M8", Quantity: "200", UnitPrice: "0.32", Total: "64.00"},
			{Description: "CHAVE COMBINADA 13MM CROMO VANADIO", Quantity: "4", UnitPrice: "38.9", Total: "155.60"},
		},
	}
	if len(res.Fields) != len(want) {
		t.Fatalf("got %d fields, want %d", len(res.Fields), len(want))
	}
	for key, wantValue := range want {
		got, ok := res.Fields[key]
		if !ok {
			t.Errorf("missing field %q", key)
			continue
		}
		if !reflect.DeepEqual(got.Value, wantValue) {
			t.Errorf("%s = %#v, want %#v", key, got.Value, wantValue)
		}
		if got.Confidence != 1 || got.Source != extract.SourceRule {
			t.Errorf("%s confidence/source = %v/%s, want 1/rule", key, got.Confidence, got.Source)
		}
	}
}

func TestParserValidation(t *testing.T) {
	sample := loadSample(t)

	tests := []struct {
		name        string
		mutate      func(string) string
		wantFlagged []string // fields expected at confidence 0
		wantValue   map[string]any
	}{
		{
			name: "bare NFe without nfeProc envelope",
			mutate: func(s string) string {
				start := strings.Index(s, "<NFe ")
				end := strings.Index(s, "</NFe>") + len("</NFe>")
				return s[start:end]
			},
		},
		{
			name: "access key check digit wrong",
			mutate: func(s string) string {
				return strings.Replace(s, `Id="NFe`+sampleKey, `Id="NFe`+sampleKey[:43]+"0", 1)
			},
			wantFlagged: []string{extract.KeyNFeAccessKey},
		},
		{
			name: "issuer CNPJ invalid",
			mutate: func(s string) string {
				return strings.Replace(s, "<CNPJ>11222333000181</CNPJ>", "<CNPJ>11222333000182</CNPJ>", 1)
			},
			wantFlagged: []string{extract.KeySupplierTaxID, extract.KeyNFeAccessKey},
		},
		{
			name: "invoice number disagrees with access key",
			mutate: func(s string) string {
				return strings.Replace(s, "<nNF>12345</nNF>", "<nNF>12346</nNF>", 1)
			},
			wantFlagged: []string{extract.KeyInvoiceNumber, extract.KeyNFeAccessKey},
		},
		{
			name: "line items do not add up to vProd",
			mutate: func(s string) string {
				return strings.Replace(s, "<vProd>64.00</vProd>", "<vProd>65.00</vProd>", 1)
			},
			wantFlagged: []string{extract.KeyLineItems},
		},
		{
			name: "unparseable total",
			mutate: func(s string) string {
				return strings.Replace(s, "<vNF>414.60</vNF>", "<vNF>414,60</vNF>", 1)
			},
			wantFlagged: []string{extract.KeyTotalAmount},
			wantValue:   map[string]any{extract.KeyTotalAmount: "414,60"},
		},
		{
			name: "no billing block means no due date",
			mutate: func(s string) string {
				start := strings.Index(s, "<cobr>")
				end := strings.Index(s, "</cobr>") + len("</cobr>")
				return s[:start] + s[end:]
			},
			wantValue: map[string]any{extract.KeyDueDate: nil},
		},
		{
			name: "no approximate tax burden",
			mutate: func(s string) string {
				return strings.Replace(s, "<vTotTrib>98.27</vTotTrib>", "", 1)
			},
			wantValue: map[string]any{extract.KeyTaxAmount: nil},
		},
		{
			name: "late-evening issue keeps local calendar date",
			mutate: func(s string) string {
				return strings.Replace(s, "2026-09-30T10:15:00-03:00", "2026-09-30T22:40:00-03:00", 1)
			},
			wantValue: map[string]any{extract.KeyIssueDate: "2026-09-30"},
		},
		{
			name: "legacy dEmi layout",
			mutate: func(s string) string {
				return strings.Replace(s, "<dhEmi>2026-09-30T10:15:00-03:00</dhEmi>", "<dEmi>2026-09-30</dEmi>", 1)
			},
			wantValue: map[string]any{extract.KeyIssueDate: "2026-09-30"},
		},
		{
			name: "malformed issue timestamp",
			mutate: func(s string) string {
				return strings.Replace(s, "2026-09-30T10:15:00-03:00", "30/09/2026 10:15", 1)
			},
			wantFlagged: []string{extract.KeyIssueDate},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			res := extractString(t, tt.mutate(sample))

			flagged := make(map[string]bool, len(tt.wantFlagged))
			for _, k := range tt.wantFlagged {
				flagged[k] = true
			}
			for key, f := range res.Fields {
				want := 1.0
				if flagged[key] {
					want = 0
				}
				if f.Confidence != want {
					t.Errorf("%s confidence = %v, want %v", key, f.Confidence, want)
				}
			}
			for key, want := range tt.wantValue {
				if got := res.Fields[key].Value; !reflect.DeepEqual(got, want) {
					t.Errorf("%s = %#v, want %#v", key, got, want)
				}
			}
		})
	}
}

func TestParserNotApplicable(t *testing.T) {
	tests := []struct {
		name string
		doc  string
	}{
		{name: "other xml schema", doc: `<?xml version="1.0"?><Invoice xmlns="urn:oasis:names:specification:ubl:schema:xsd:Invoice-2"><ID>1</ID></Invoice>`},
		{name: "truncated", doc: `<nfeProc><NFe><infNFe Id="NFe1"><ide>`},
		{name: "not xml", doc: `%PDF-1.7`},
		{name: "empty", doc: ``},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewParser().Extract(context.Background(), extract.Document{Content: []byte(tt.doc)})
			if !errors.Is(err, extract.ErrNotApplicable) {
				t.Fatalf("error = %v, want ErrNotApplicable", err)
			}
		})
	}
}
