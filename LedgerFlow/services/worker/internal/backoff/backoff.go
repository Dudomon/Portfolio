// Package backoff provides the jittered exponential delays shared by every
// outbound retry loop in the worker (HTTP clients and AMQP reconnects).
package backoff

import (
	"context"
	"math/rand/v2"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// Policy describes an exponential backoff: Base doubles per retry until Max.
type Policy struct {
	Base time.Duration
	Max  time.Duration
}

// Delay returns the wait before retry n (0-based) using "equal jitter": half
// of the exponential step is fixed and half is random. Pure full jitter can
// return ~0, which turns a flapping dependency into a hot loop; equal jitter
// keeps a floor while still de-synchronizing replicas.
func (p Policy) Delay(n int) time.Duration {
	if p.Base <= 0 {
		return 0
	}
	d := p.Base
	for i := 0; i < n; i++ {
		if p.Max > 0 && d >= p.Max {
			break
		}
		d *= 2
	}
	if p.Max > 0 && d > p.Max {
		d = p.Max
	}
	half := d / 2
	return half + time.Duration(rand.Int64N(int64(d-half)+1))
}

// Sleep waits for d or until ctx is done, whichever comes first.
func Sleep(ctx context.Context, d time.Duration) error {
	if d <= 0 {
		return ctx.Err()
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}

// ParseRetryAfter interprets an HTTP Retry-After header value, which may be
// either delta-seconds or an HTTP-date (RFC 9110 §10.2.3).
func ParseRetryAfter(v string, now time.Time) (time.Duration, bool) {
	v = strings.TrimSpace(v)
	if v == "" {
		return 0, false
	}
	if secs, err := strconv.Atoi(v); err == nil {
		if secs < 0 {
			return 0, false
		}
		return time.Duration(secs) * time.Second, true
	}
	t, err := http.ParseTime(v)
	if err != nil {
		return 0, false
	}
	if d := t.Sub(now); d > 0 {
		return d, true
	}
	return 0, true
}
