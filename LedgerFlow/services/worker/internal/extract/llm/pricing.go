package llm

import (
	"fmt"
	"strings"

	"github.com/shopspring/decimal"
)

// Price is the USD cost per one million tokens. Conveniently, that number is
// also the cost in micro-USD per single token, which is the unit the API
// stores (cost_usd_micros).
type Price struct {
	InputPerMTok  decimal.Decimal
	OutputPerMTok decimal.Decimal
}

// PriceTable maps a model name or name prefix to its price. Providers return
// dated snapshots ("gpt-4o-mini-2024-07-18"), so lookups fall back to the
// longest configured prefix.
type PriceTable map[string]Price

// DefaultPrices returns list prices for the models the worker is configured
// with by default. Override with LLM_PRICES when using other models or
// negotiated rates.
func DefaultPrices() PriceTable {
	return PriceTable{
		"gpt-4o-mini": {InputPerMTok: decimal.RequireFromString("0.15"), OutputPerMTok: decimal.RequireFromString("0.60")},
		"gpt-4o":      {InputPerMTok: decimal.RequireFromString("2.50"), OutputPerMTok: decimal.RequireFromString("10.00")},
	}
}

// ParsePrices parses "model=input/output,model=input/output", prices in USD
// per million tokens, e.g. "gpt-4o-mini=0.15/0.60".
func ParsePrices(s string) (PriceTable, error) {
	table := PriceTable{}
	for _, entry := range strings.Split(s, ",") {
		entry = strings.TrimSpace(entry)
		if entry == "" {
			continue
		}
		model, prices, ok := strings.Cut(entry, "=")
		in, out, ok2 := strings.Cut(prices, "/")
		if !ok || !ok2 || strings.TrimSpace(model) == "" {
			return nil, fmt.Errorf("price entry %q: want model=input/output", entry)
		}
		inPrice, err := decimal.NewFromString(strings.TrimSpace(in))
		if err != nil {
			return nil, fmt.Errorf("price entry %q: input price: %w", entry, err)
		}
		outPrice, err := decimal.NewFromString(strings.TrimSpace(out))
		if err != nil {
			return nil, fmt.Errorf("price entry %q: output price: %w", entry, err)
		}
		if inPrice.IsNegative() || outPrice.IsNegative() {
			return nil, fmt.Errorf("price entry %q: prices must not be negative", entry)
		}
		table[strings.TrimSpace(model)] = Price{InputPerMTok: inPrice, OutputPerMTok: outPrice}
	}
	if len(table) == 0 {
		return nil, fmt.Errorf("price table %q has no entries", s)
	}
	return table, nil
}

// Lookup finds the price for model by exact name, then by longest prefix.
func (t PriceTable) Lookup(model string) (Price, bool) {
	if p, ok := t[model]; ok {
		return p, true
	}
	var (
		best    Price
		bestLen int
	)
	for prefix, p := range t {
		if len(prefix) > bestLen && strings.HasPrefix(model, prefix) {
			best, bestLen = p, len(prefix)
		}
	}
	return best, bestLen > 0
}

// CostMicros returns the request cost in micro-USD, rounded up so that
// accumulated spend is never under-reported.
func (t PriceTable) CostMicros(model string, promptTokens, completionTokens int64) (int64, bool) {
	p, ok := t.Lookup(model)
	if !ok {
		return 0, false
	}
	cost := p.InputPerMTok.Mul(decimal.NewFromInt(promptTokens)).
		Add(p.OutputPerMTok.Mul(decimal.NewFromInt(completionTokens)))
	return cost.Ceil().IntPart(), true
}
