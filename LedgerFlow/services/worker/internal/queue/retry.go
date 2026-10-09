package queue

import (
	"time"

	amqp "github.com/rabbitmq/amqp091-go"
)

const (
	// attemptHeader carries the 1-based delivery attempt across retries.
	// It is authoritative over the body's "attempt" field because the body
	// is republished untouched.
	attemptHeader = "x-attempt"
	// lastErrorHeader records why a message ended up in the DLQ.
	lastErrorHeader = "x-last-error"
)

// RetryPolicy decides what happens to a message whose handling failed.
type RetryPolicy struct {
	// MaxRetries is the number of redeliveries after the first attempt.
	MaxRetries int
	// Delays is the wait before each retry; the last entry repeats if
	// MaxRetries exceeds len(Delays).
	Delays []time.Duration
}

// Next reports whether a message that failed on attempt (1-based) should be
// retried, and after how long. With the contract's schedule (3 retries at
// 5s, 30s, 2m), attempts 1-3 are retried and attempt 4 is dead-lettered.
func (p RetryPolicy) Next(attempt int) (time.Duration, bool) {
	retriesSoFar := max(attempt, 1) - 1
	if retriesSoFar >= p.MaxRetries || len(p.Delays) == 0 {
		return 0, false
	}
	return p.Delays[min(retriesSoFar, len(p.Delays)-1)], true
}

// attemptFromHeaders reads x-attempt, accepting any integer width because
// AMQP clients in other languages encode small integers differently.
func attemptFromHeaders(h amqp.Table) int {
	var n int64
	switch v := h[attemptHeader].(type) {
	case int64:
		n = v
	case int32:
		n = int64(v)
	case int16:
		n = int64(v)
	case int8:
		n = int64(v)
	case int:
		n = int64(v)
	case uint8:
		n = int64(v)
	case uint16:
		n = int64(v)
	case uint32:
		n = int64(v)
	default:
		return 1
	}
	// Clamp nonsense rather than trusting it: a negative or huge value must
	// not bypass the retry budget or overflow int on 32-bit platforms.
	const maxAttempt = 1 << 16
	switch {
	case n < 1:
		return 1
	case n > maxAttempt:
		return maxAttempt
	default:
		return int(n)
	}
}
