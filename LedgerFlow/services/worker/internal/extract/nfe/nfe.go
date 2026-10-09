// Package nfe extracts contract fields from Brazilian NF-e XML (layout 4.00).
//
// The parser is deterministic: every value is read from a schema-defined
// element, so fields are emitted with confidence 1.0 and source "rule". When
// a value fails validation (bad check digit, unparseable amount, internal
// inconsistency) its confidence drops to 0 instead of the document being
// rejected, which routes it to human review with everything else intact.
package nfe

import (
	"bytes"
	"context"
	"encoding/xml"
	"errors"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/shopspring/decimal"

	"github.com/Dudomon/ledgerflow/services/worker/internal/extract"
	"github.com/Dudomon/ledgerflow/services/worker/internal/extract/brtax"
)

const (
	extractorName   = "nfe-rules"
	accessKeyPrefix = "NFe"
	dateLayout      = "2006-01-02"

	confident = 1.0
	flagged   = 0.0
)

var errNoInfNFe = errors.New("no infNFe element found")

// infNFe maps the subset of the NF-e layout the contract needs. Tags carry no
// namespace so both the bare <NFe> and the authorised <nfeProc> envelope match.
type infNFe struct {
	ID  string `xml:"Id,attr"`
	Ide struct {
		NNF   string `xml:"nNF"`
		DhEmi string `xml:"dhEmi"` // layout 3.10+
		DEmi  string `xml:"dEmi"`  // layout 2.00, still found in archives
	} `xml:"ide"`
	Emit struct {
		CNPJ  string `xml:"CNPJ"`
		XNome string `xml:"xNome"`
	} `xml:"emit"`
	Det []struct {
		Prod struct {
			XProd  string `xml:"xProd"`
			QCom   string `xml:"qCom"`
			VUnCom string `xml:"vUnCom"`
			VProd  string `xml:"vProd"`
			IndTot string `xml:"indTot"`
		} `xml:"prod"`
	} `xml:"det"`
	Total struct {
		ICMSTot struct {
			VProd    string `xml:"vProd"`
			VNF      string `xml:"vNF"`
			VTotTrib string `xml:"vTotTrib"`
		} `xml:"ICMSTot"`
	} `xml:"total"`
	Cobr struct {
		Dup []struct {
			DVenc string `xml:"dVenc"`
		} `xml:"dup"`
	} `xml:"cobr"`
}

// Parser is the rule-based NF-e extractor. It is stateless and safe for
// concurrent use.
type Parser struct{}

// NewParser returns an NF-e parser.
func NewParser() *Parser { return &Parser{} }

// Extract implements extract.Extractor. Input that does not contain an
// infNFe element yields extract.ErrNotApplicable so the router can hand the
// document to another strategy.
func (p *Parser) Extract(_ context.Context, doc extract.Document) (extract.Result, error) {
	inf, err := decode(doc.Content)
	if err != nil {
		return extract.Result{}, fmt.Errorf("%w: %w", extract.ErrNotApplicable, err)
	}
	return extract.Result{
		DocType:   extract.DocTypeNFe,
		Fields:    fieldsFrom(inf),
		Extractor: extractorName,
	}, nil
}

// decode streams tokens until infNFe so the envelope (nfeProc, NFe, SOAP
// wrappers) does not matter. encoding/xml never resolves external entities,
// so untrusted uploads cannot trigger XXE.
func decode(data []byte) (*infNFe, error) {
	dec := xml.NewDecoder(bytes.NewReader(data))
	for {
		tok, err := dec.Token()
		if errors.Is(err, io.EOF) {
			return nil, errNoInfNFe
		}
		if err != nil {
			return nil, fmt.Errorf("parse xml: %w", err)
		}
		start, ok := tok.(xml.StartElement)
		if !ok || start.Name.Local != "infNFe" {
			continue
		}
		var inf infNFe
		if err := dec.DecodeElement(&inf, &start); err != nil {
			return nil, fmt.Errorf("decode infNFe: %w", err)
		}
		return &inf, nil
	}
}

func fieldsFrom(inf *infNFe) extract.Fields {
	cnpj := strings.TrimSpace(inf.Emit.CNPJ)
	number := strings.TrimSpace(inf.Ide.NNF)
	rawKey := strings.TrimPrefix(strings.TrimSpace(inf.ID), accessKeyPrefix)

	f := extract.Fields{
		extract.KeySupplierName:  required(strings.TrimSpace(inf.Emit.XNome)),
		extract.KeySupplierTaxID: validated(cnpj, brtax.ValidCNPJ(cnpj)),
		extract.KeyInvoiceNumber: required(number),
		extract.KeyIssueDate:     issueDate(inf.Ide.DhEmi, inf.Ide.DEmi),
		extract.KeyDueDate:       dueDate(inf),
		// NF-e amounts are always in reais; the layout has no currency field.
		extract.KeyCurrency:    rule("BRL"),
		extract.KeyTotalAmount: money(inf.Total.ICMSTot.VNF, true),
		// vTotTrib is the approximate tax burden required by Lei 12.741/2012;
		// it is optional in the layout.
		extract.KeyTaxAmount: money(inf.Total.ICMSTot.VTotTrib, false),
		extract.KeyLineItems: lineItems(inf),
	}

	key, err := brtax.ParseAccessKey(rawKey)
	if err != nil {
		f[extract.KeyNFeAccessKey] = flag(nonEmpty(rawKey))
		return f
	}
	f[extract.KeyNFeAccessKey] = rule(key.Raw)

	// The access key embeds the issuer CNPJ and invoice number. A mismatch
	// means the XML was edited after issue or assembled by hand; we cannot
	// tell which side is wrong, so both go to review.
	if key.IssuerCNPJ != cnpj {
		f.CapConfidence(extract.KeySupplierTaxID, flagged)
		f.CapConfidence(extract.KeyNFeAccessKey, flagged)
	}
	if strings.TrimLeft(key.Number, "0") != strings.TrimLeft(number, "0") {
		f.CapConfidence(extract.KeyInvoiceNumber, flagged)
		f.CapConfidence(extract.KeyNFeAccessKey, flagged)
	}
	return f
}

func rule(v any) extract.Field {
	return extract.Field{Value: v, Confidence: confident, Source: extract.SourceRule}
}

func flag(v any) extract.Field {
	return extract.Field{Value: v, Confidence: flagged, Source: extract.SourceRule}
}

func validated(v string, ok bool) extract.Field {
	if !ok {
		return flag(nonEmpty(v))
	}
	return rule(v)
}

func required(v string) extract.Field {
	return validated(v, v != "")
}

// nonEmpty maps "" to JSON null so reviewers see "missing", not "blank".
func nonEmpty(v string) any {
	if v == "" {
		return nil
	}
	return v
}

func issueDate(dhEmi, dEmi string) extract.Field {
	if dhEmi = strings.TrimSpace(dhEmi); dhEmi != "" {
		t, err := time.Parse(time.RFC3339, dhEmi)
		if err != nil {
			return flag(dhEmi)
		}
		// Keep the issuer's offset: converting to UTC would move invoices
		// issued after 21:00 in Brasília to the next calendar day.
		return rule(t.Format(dateLayout))
	}
	dEmi = strings.TrimSpace(dEmi)
	if _, err := time.Parse(dateLayout, dEmi); err != nil {
		return flag(nonEmpty(dEmi))
	}
	return rule(dEmi)
}

// dueDate is the earliest installment due date. Invoices paid on issue have
// no <cobr> block, in which case "no due date" is itself a certain answer.
func dueDate(inf *infNFe) extract.Field {
	var earliest time.Time
	for _, d := range inf.Cobr.Dup {
		raw := strings.TrimSpace(d.DVenc)
		t, err := time.Parse(dateLayout, raw)
		if err != nil {
			return flag(nonEmpty(raw))
		}
		if earliest.IsZero() || t.Before(earliest) {
			earliest = t
		}
	}
	if earliest.IsZero() {
		return rule(nil)
	}
	return rule(earliest.Format(dateLayout))
}

func money(raw string, isRequired bool) extract.Field {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		if isRequired {
			return flag(nil)
		}
		return rule(nil)
	}
	d, err := decimal.NewFromString(raw)
	if err != nil || d.IsNegative() {
		return flag(raw)
	}
	return rule(d.StringFixed(2))
}

func lineItems(inf *infNFe) extract.Field {
	items := make([]extract.LineItem, 0, len(inf.Det))
	sum := decimal.Zero
	ok := true

	for _, det := range inf.Det {
		p := det.Prod
		item := extract.LineItem{
			Description: strings.TrimSpace(p.XProd),
			Quantity:    strings.TrimSpace(p.QCom),
			UnitPrice:   strings.TrimSpace(p.VUnCom),
			Total:       strings.TrimSpace(p.VProd),
		}
		qty, errQ := decimal.NewFromString(item.Quantity)
		unit, errU := decimal.NewFromString(item.UnitPrice)
		total, errT := decimal.NewFromString(item.Total)
		if errQ != nil || errU != nil || errT != nil {
			// Keep the raw strings so the reviewer sees what the XML says.
			ok = false
			items = append(items, item)
			continue
		}
		// qCom/vUnCom carry up to 4 and 10 decimal places; trailing zeros
		// are layout padding, not precision, so they are normalised away.
		item.Quantity = qty.String()
		item.UnitPrice = unit.String()
		item.Total = total.StringFixed(2)
		items = append(items, item)
		if p.IndTot != "0" {
			sum = sum.Add(total)
		}
	}

	// ICMSTot/vProd is by definition the sum of det/prod/vProd over items
	// with indTot=1; a mismatch signals a malformed or tampered document.
	if declared, err := decimal.NewFromString(strings.TrimSpace(inf.Total.ICMSTot.VProd)); err != nil || !declared.Equal(sum) {
		ok = false
	}
	if len(items) == 0 || !ok {
		return flag(items)
	}
	return rule(items)
}
