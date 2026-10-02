// Package retry provides opt-in HTTP retries for providers.
//
// Providers never retry. Pass a client from NewHTTPClient as a provider
// Config.HTTPClient to opt in. Enabling retries authorizes repeated requests
// and possible duplicate charges: a transient HTTP status does not establish
// that an operation was not processed.
package retry

import (
	"bytes"
	"context"
	"errors"
	"io"
	"math/rand/v2"
	"net/http"
	"time"

	"github.com/voocel/litellm/internal/wire"
)

// Policy controls retries. Zero durations and multiplier take the
// DefaultPolicy values.
type Policy struct {
	// MaxAttempts counts the first attempt; 1 or less disables retries.
	MaxAttempts int
	// Delays grow from InitialDelay by Multiplier up to MaxDelay.
	InitialDelay time.Duration
	MaxDelay     time.Duration
	Multiplier   float64
	// Jitter varies each delay by up to ±25%.
	Jitter bool
	// RespectRetryAfter uses the wait the server suggests instead, in a
	// Retry-After header or a Google API error body, even beyond MaxDelay.
	// One beyond MaxRetryAfter, 60s by default, ends retrying with the
	// response, whose error reports the wait in RetryAfter.
	RespectRetryAfter bool
	MaxRetryAfter     time.Duration
}

// DefaultPolicy returns a conservative retry policy for complete retryable HTTP
// responses. It does not retry network write/read errors because a POST may
// already have been processed by the provider.
func DefaultPolicy() *Policy {
	return &Policy{
		MaxAttempts:       3,
		InitialDelay:      200 * time.Millisecond,
		MaxDelay:          2 * time.Second,
		Multiplier:        2,
		Jitter:            true,
		RespectRetryAfter: true,
		MaxRetryAfter:     time.Minute,
	}
}

// NewHTTPClient returns a shallow copy of base, http.DefaultClient when nil,
// whose Transport retries, according to policy, complete responses that
// providers report as temporary errors: 408, 429, 500, 502, 503, 504 and 529
// statuses, except those the body shows to be exhausted quota,
// authentication, content filter or context overflow failures. A nil policy
// or MaxAttempts <= 1 disables retries. Only replayable request bodies are
// resent; replayable means the bytes can be resent, not that the operation is
// idempotent. Transport failures and response-body errors, including
// interrupted successful streams, are never retried.
func NewHTTPClient(base *http.Client, policy *Policy) *http.Client {
	if base == nil {
		base = http.DefaultClient
	}
	out := *base
	out.Transport = newTransport(base.Transport, policy)
	return &out
}

func newTransport(base http.RoundTripper, policy *Policy) http.RoundTripper {
	if base == nil {
		base = http.DefaultTransport
	}
	if policy == nil || policy.MaxAttempts <= 1 {
		return base
	}
	return &transport{base: base, policy: normalizePolicy(*policy)}
}

type transport struct {
	base   http.RoundTripper
	policy Policy
}

func (t *transport) RoundTrip(req *http.Request) (*http.Response, error) {
	policy, base := t.policy, t.base
	for attempt := 1; attempt <= policy.MaxAttempts; attempt++ {
		attemptReq, err := requestForAttempt(req, attempt)
		if err != nil {
			return nil, err
		}
		resp, err := base.RoundTrip(attemptReq)
		if err != nil {
			return nil, err
		}
		// A body that cannot be resent ends retrying with the response as is.
		if attempt == policy.MaxAttempts || (req.Body != nil && req.GetBody == nil) {
			return resp, nil
		}
		temp, retryAfter := temporary(resp)
		if !temp {
			return resp, nil
		}
		delay, ok := policy.delay(attempt, retryAfter)
		if !ok {
			return resp, nil
		}
		discard(resp)
		if err := sleep(req.Context(), delay); err != nil {
			return nil, err
		}
	}
	return nil, errors.New("retry: exhausted attempts without response")
}

func requestForAttempt(req *http.Request, attempt int) (*http.Request, error) {
	if attempt == 1 {
		return req, nil
	}
	cloned := req.Clone(req.Context())
	if req.GetBody != nil {
		body, err := req.GetBody()
		if err != nil {
			return nil, err
		}
		cloned.Body = body
	}
	return cloned, nil
}

func normalizePolicy(policy Policy) Policy {
	if policy.MaxAttempts <= 0 {
		policy.MaxAttempts = 1
	}
	if policy.InitialDelay <= 0 {
		policy.InitialDelay = 200 * time.Millisecond
	}
	if policy.MaxDelay <= 0 {
		policy.MaxDelay = 2 * time.Second
	}
	if policy.Multiplier <= 0 {
		policy.Multiplier = 2
	}
	if policy.MaxRetryAfter <= 0 {
		policy.MaxRetryAfter = time.Minute
	}
	return policy
}

// delay returns the wait before the next attempt, or false when the server
// asks, with retryAfter, for a longer one than MaxRetryAfter.
func (p Policy) delay(attempt int, retryAfter time.Duration) (time.Duration, bool) {
	if p.RespectRetryAfter && retryAfter > 0 {
		return retryAfter, retryAfter <= p.MaxRetryAfter
	}
	delay := p.InitialDelay
	for i := 1; i < attempt; i++ {
		delay = time.Duration(float64(delay) * p.Multiplier)
		if delay >= p.MaxDelay {
			delay = p.MaxDelay
			break
		}
	}
	if delay > p.MaxDelay {
		delay = p.MaxDelay
	}
	if p.Jitter && delay > 0 {
		spread := float64(delay) * 0.25
		delay = time.Duration(float64(delay) + spread*(2*rand.Float64()-1))
		if delay < 0 {
			delay = 0
		}
	}
	return delay, true
}

// temporary classifies a failed response as the provider will report it:
// whether it is temporary, and the wait the server suggests. The status
// decides unless the body can rule a retry out; the body prefix read for
// that is put back.
func temporary(resp *http.Response) (bool, time.Duration) {
	if resp.StatusCode < 400 || !wire.HTTPError("", resp.StatusCode, nil, "").Temporary {
		return false, 0
	}
	var data []byte
	if resp.Body != nil {
		data, _ = io.ReadAll(io.LimitReader(resp.Body, wire.MaxErrorBody))
		resp.Body = struct {
			io.Reader
			io.Closer
		}{io.MultiReader(bytes.NewReader(data), resp.Body), resp.Body}
	}
	e := wire.HTTPError("", resp.StatusCode, resp.Header, string(data))
	return e.Temporary, e.RetryAfter
}

func sleep(ctx context.Context, delay time.Duration) error {
	if delay <= 0 {
		return ctx.Err()
	}
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-timer.C:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// discard closes a response that is about to be retried. It drains a bounded
// prefix so the connection can be reused; a read error does not matter here.
func discard(resp *http.Response) {
	if resp.Body == nil {
		return
	}
	_, _ = io.CopyN(io.Discard, resp.Body, 64<<10)
	_ = resp.Body.Close()
}
