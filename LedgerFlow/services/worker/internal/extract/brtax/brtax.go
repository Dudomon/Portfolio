// Package brtax validates Brazilian fiscal identifiers (CNPJ and NF-e access
// keys). Both use the same modulo-11 check-digit scheme, so a single
// implementation backs them.
//
// Only numeric CNPJs are accepted: the alphanumeric format introduced by
// IN RFB 2.229/2024 is out of scope because the service contract defines
// supplier_tax_id and nfe_access_key as digit strings.
package brtax

import (
	"errors"
	"strings"
)

const (
	cnpjLen      = 14
	accessKeyLen = 44
)

// ErrInvalidAccessKey is returned when a string is not a well-formed NF-e
// access key with a correct check digit.
var ErrInvalidAccessKey = errors.New("invalid NF-e access key")

// OnlyDigits strips everything but ASCII digits, so formatted input such as
// "11.222.333/0001-81" can be validated.
func OnlyDigits(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	for i := 0; i < len(s); i++ {
		if c := s[i]; c >= '0' && c <= '9' {
			b.WriteByte(c)
		}
	}
	return b.String()
}

// ValidCNPJ reports whether s is a 14-digit CNPJ with correct check digits.
// Callers normalise formatting with OnlyDigits first.
func ValidCNPJ(s string) bool {
	if len(s) != cnpjLen || !allDigits(s) || allSame(s) {
		return false
	}
	return mod11(s[:12]) == int(s[12]-'0') && mod11(s[:13]) == int(s[13]-'0')
}

// AccessKey is a decoded NF-e chave de acesso. Field boundaries follow the
// layout in the NF-e Manual de Orientação do Contribuinte.
type AccessKey struct {
	Raw          string
	StateCode    string // cUF, IBGE state code
	YearMonth    string // AAMM of issue
	IssuerCNPJ   string
	Model        string // 55 = NF-e, 65 = NFC-e
	Series       string
	Number       string // nNF, zero-padded to 9 digits
	EmissionType string // tpEmis
	RandomCode   string // cNF
}

// ParseAccessKey validates and decodes a 44-digit NF-e access key.
func ParseAccessKey(s string) (AccessKey, error) {
	if len(s) != accessKeyLen || !allDigits(s) {
		return AccessKey{}, ErrInvalidAccessKey
	}
	if mod11(s[:43]) != int(s[43]-'0') {
		return AccessKey{}, ErrInvalidAccessKey
	}
	return AccessKey{
		Raw:          s,
		StateCode:    s[0:2],
		YearMonth:    s[2:6],
		IssuerCNPJ:   s[6:20],
		Model:        s[20:22],
		Series:       s[22:25],
		Number:       s[25:34],
		EmissionType: s[34:35],
		RandomCode:   s[35:43],
	}, nil
}

// mod11 computes the check digit used by both CNPJ and NF-e access keys:
// weights 2..9 cycle from the rightmost digit, and remainders 0 and 1 map
// to a check digit of 0.
func mod11(digits string) int {
	sum, weight := 0, 2
	for i := len(digits) - 1; i >= 0; i-- {
		sum += int(digits[i]-'0') * weight
		weight++
		if weight > 9 {
			weight = 2
		}
	}
	r := sum % 11
	if r < 2 {
		return 0
	}
	return 11 - r
}

func allDigits(s string) bool {
	for i := 0; i < len(s); i++ {
		if s[i] < '0' || s[i] > '9' {
			return false
		}
	}
	return true
}

// allSame rejects sequences such as 00000000000000 whose check digits happen
// to be valid but which the Receita Federal never issues.
func allSame(s string) bool {
	return strings.Count(s, s[:1]) == len(s)
}
