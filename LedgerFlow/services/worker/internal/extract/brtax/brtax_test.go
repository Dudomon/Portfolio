package brtax

import (
	"errors"
	"testing"
)

func TestValidCNPJ(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want bool
	}{
		{name: "valid", in: "11222333000181", want: true},
		{name: "valid second", in: "27865757000102", want: true},
		{name: "wrong first check digit", in: "11222333000191", want: false},
		{name: "wrong second check digit", in: "11222333000182", want: false},
		{name: "formatted input must be normalised first", in: "11.222.333/0001-81", want: false},
		{name: "too short", in: "1122233300018", want: false},
		{name: "too long", in: "112223330001810", want: false},
		{name: "letters", in: "1122233300018A", want: false},
		{name: "repeated digits", in: "00000000000000", want: false},
		{name: "empty", in: "", want: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := ValidCNPJ(tt.in); got != tt.want {
				t.Fatalf("ValidCNPJ(%q) = %v, want %v", tt.in, got, tt.want)
			}
		})
	}
}

func TestOnlyDigits(t *testing.T) {
	if got := OnlyDigits("11.222.333/0001-81"); got != "11222333000181" {
		t.Fatalf("OnlyDigits = %q", got)
	}
}

func TestParseAccessKey(t *testing.T) {
	const valid = "35260911222333000181550010000123451473829105"

	tests := []struct {
		name    string
		in      string
		wantErr bool
	}{
		{name: "valid", in: valid},
		{name: "wrong check digit", in: valid[:43] + "6", wantErr: true},
		{name: "transposed digits", in: "35260911222333000181550010000123415473829105", wantErr: true},
		{name: "43 digits", in: valid[:43], wantErr: true},
		{name: "non digit", in: valid[:10] + "X" + valid[11:], wantErr: true},
		{name: "with NFe prefix", in: "NFe" + valid, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ParseAccessKey(tt.in)
			if tt.wantErr != (err != nil) {
				t.Fatalf("ParseAccessKey(%q) error = %v, wantErr %v", tt.in, err, tt.wantErr)
			}
			if err != nil && !errors.Is(err, ErrInvalidAccessKey) {
				t.Fatalf("error %v does not wrap ErrInvalidAccessKey", err)
			}
		})
	}
}

func TestParseAccessKeyLayout(t *testing.T) {
	k, err := ParseAccessKey("35260911222333000181550010000123451473829105")
	if err != nil {
		t.Fatal(err)
	}
	want := AccessKey{
		Raw:          "35260911222333000181550010000123451473829105",
		StateCode:    "35",
		YearMonth:    "2609",
		IssuerCNPJ:   "11222333000181",
		Model:        "55",
		Series:       "001",
		Number:       "000012345",
		EmissionType: "1",
		RandomCode:   "47382910",
	}
	if k != want {
		t.Fatalf("ParseAccessKey layout:\n got %+v\nwant %+v", k, want)
	}
}
