package llm

import "testing"

func TestCostMicros(t *testing.T) {
	prices := DefaultPrices()

	tests := []struct {
		name               string
		model              string
		prompt, completion int64
		want               int64
		wantOK             bool
	}{
		{name: "dated snapshot matches longest prefix", model: "gpt-4o-mini-2024-07-18", prompt: 1000, completion: 200, want: 270, wantOK: true},
		{name: "shorter prefix model", model: "gpt-4o-2024-08-06", prompt: 1000, completion: 200, want: 4500, wantOK: true},
		{name: "fractional micros round up", model: "gpt-4o-mini", prompt: 1, completion: 0, want: 1, wantOK: true},
		{name: "zero usage", model: "gpt-4o-mini", want: 0, wantOK: true},
		{name: "unknown model", model: "unlisted-model-x", prompt: 1000, completion: 200, want: 0, wantOK: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := prices.CostMicros(tt.model, tt.prompt, tt.completion)
			if got != tt.want || ok != tt.wantOK {
				t.Fatalf("CostMicros = (%d, %v), want (%d, %v)", got, ok, tt.want, tt.wantOK)
			}
		})
	}
}

func TestParsePrices(t *testing.T) {
	table, err := ParsePrices(" gpt-4o-mini=0.15/0.60 , local-model=0/0")
	if err != nil {
		t.Fatal(err)
	}
	if got, _ := table.CostMicros("gpt-4o-mini-2024-07-18", 1000, 200); got != 270 {
		t.Fatalf("parsed price cost = %d, want 270", got)
	}
	if got, ok := table.CostMicros("local-model", 1000, 1000); !ok || got != 0 {
		t.Fatalf("free model cost = (%d, %v)", got, ok)
	}

	for _, bad := range []string{"", "gpt-4o", "gpt-4o=1", "=1/2", "gpt-4o=x/2", "gpt-4o=1/-2"} {
		if _, err := ParsePrices(bad); err == nil {
			t.Errorf("ParsePrices(%q) succeeded, want error", bad)
		}
	}
}
