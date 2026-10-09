package backoff

import (
	"context"
	"errors"
	"testing"
	"time"
)

func TestPolicyDelayBounds(t *testing.T) {
	p := Policy{Base: 100 * time.Millisecond, Max: 2 * time.Second}

	tests := []struct {
		retry    int
		min, max time.Duration
	}{
		{retry: 0, min: 50 * time.Millisecond, max: 100 * time.Millisecond},
		{retry: 1, min: 100 * time.Millisecond, max: 200 * time.Millisecond},
		{retry: 3, min: 400 * time.Millisecond, max: 800 * time.Millisecond},
		{retry: 10, min: time.Second, max: 2 * time.Second},
		{retry: 1000, min: time.Second, max: 2 * time.Second},
	}
	for _, tt := range tests {
		for i := 0; i < 200; i++ {
			got := p.Delay(tt.retry)
			if got < tt.min || got > tt.max {
				t.Fatalf("Delay(%d) = %v, want within [%v, %v]", tt.retry, got, tt.min, tt.max)
			}
		}
	}
}

func TestPolicyDelayZeroBase(t *testing.T) {
	if got := (Policy{}).Delay(3); got != 0 {
		t.Fatalf("zero policy Delay = %v, want 0", got)
	}
}

func TestSleepHonoursContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := Sleep(ctx, time.Hour); !errors.Is(err, context.Canceled) {
		t.Fatalf("Sleep on canceled ctx = %v, want context.Canceled", err)
	}
}

func TestParseRetryAfter(t *testing.T) {
	now := time.Date(2026, 10, 9, 12, 0, 0, 0, time.UTC)

	tests := []struct {
		name   string
		in     string
		want   time.Duration
		wantOK bool
	}{
		{name: "seconds", in: "7", want: 7 * time.Second, wantOK: true},
		{name: "zero", in: "0", want: 0, wantOK: true},
		{name: "http date", in: "Fri, 09 Oct 2026 12:00:30 GMT", want: 30 * time.Second, wantOK: true},
		{name: "date in past", in: "Fri, 09 Oct 2026 11:00:00 GMT", want: 0, wantOK: true},
		{name: "negative", in: "-1", wantOK: false},
		{name: "garbage", in: "soon", wantOK: false},
		{name: "empty", in: "", wantOK: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := ParseRetryAfter(tt.in, now)
			if ok != tt.wantOK || got != tt.want {
				t.Fatalf("ParseRetryAfter(%q) = (%v, %v), want (%v, %v)", tt.in, got, ok, tt.want, tt.wantOK)
			}
		})
	}
}
