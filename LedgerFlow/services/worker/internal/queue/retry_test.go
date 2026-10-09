package queue

import (
	"testing"
	"time"

	amqp "github.com/rabbitmq/amqp091-go"
)

func TestRetryPolicyNext(t *testing.T) {
	contract := RetryPolicy{MaxRetries: 3, Delays: []time.Duration{5 * time.Second, 30 * time.Second, 2 * time.Minute}}

	tests := []struct {
		name      string
		policy    RetryPolicy
		attempt   int
		wantDelay time.Duration
		wantRetry bool
	}{
		{name: "first failure", policy: contract, attempt: 1, wantDelay: 5 * time.Second, wantRetry: true},
		{name: "second failure", policy: contract, attempt: 2, wantDelay: 30 * time.Second, wantRetry: true},
		{name: "third failure", policy: contract, attempt: 3, wantDelay: 2 * time.Minute, wantRetry: true},
		{name: "budget exhausted", policy: contract, attempt: 4, wantRetry: false},
		{name: "far past budget", policy: contract, attempt: 99, wantRetry: false},
		{name: "missing attempt treated as first", policy: contract, attempt: 0, wantDelay: 5 * time.Second, wantRetry: true},
		{
			name:      "last delay repeats",
			policy:    RetryPolicy{MaxRetries: 5, Delays: []time.Duration{time.Second, time.Minute}},
			attempt:   4,
			wantDelay: time.Minute,
			wantRetry: true,
		},
		{name: "retries disabled", policy: RetryPolicy{MaxRetries: 0, Delays: contract.Delays}, attempt: 1, wantRetry: false},
		{name: "no delays configured", policy: RetryPolicy{MaxRetries: 3}, attempt: 1, wantRetry: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			delay, retry := tt.policy.Next(tt.attempt)
			if delay != tt.wantDelay || retry != tt.wantRetry {
				t.Fatalf("Next(%d) = (%v, %v), want (%v, %v)", tt.attempt, delay, retry, tt.wantDelay, tt.wantRetry)
			}
		})
	}
}

func TestAttemptFromHeaders(t *testing.T) {
	tests := []struct {
		name    string
		headers amqp.Table
		want    int
	}{
		{name: "nil headers", headers: nil, want: 1},
		{name: "absent", headers: amqp.Table{"other": "x"}, want: 1},
		{name: "int64 as published by this worker", headers: amqp.Table{attemptHeader: int64(3)}, want: 3},
		{name: "int32", headers: amqp.Table{attemptHeader: int32(2)}, want: 2},
		{name: "uint8", headers: amqp.Table{attemptHeader: uint8(4)}, want: 4},
		{name: "string is ignored", headers: amqp.Table{attemptHeader: "3"}, want: 1},
		{name: "zero clamps to first", headers: amqp.Table{attemptHeader: int64(0)}, want: 1},
		{name: "negative clamps to first", headers: amqp.Table{attemptHeader: int32(-5)}, want: 1},
		{name: "huge is clamped", headers: amqp.Table{attemptHeader: int64(1 << 40)}, want: 1 << 16},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := attemptFromHeaders(tt.headers); got != tt.want {
				t.Fatalf("attemptFromHeaders = %d, want %d", got, tt.want)
			}
		})
	}
}
