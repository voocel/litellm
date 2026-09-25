// Package retry provides opt-in HTTP retries for providers.
//
// Providers never retry. Pass a client from NewHTTPClient as a provider
// Config.HTTPClient to opt in. Enabling retries authorizes repeated requests
// and possible duplicate charges: a transient HTTP status does not establish
// that an operation was not processed.
package retry

import (
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
	// RespectRetryAfter uses the server's Retry-After instead, even beyond
	// MaxDelay.
	RespectRetryAfter bool
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
	}
}

// NewHTTPClient returns a shallow copy of base, http.DefaultClient when nil,
// whose Transport retries complete 429/5xx/529 responses according to policy.
// A nil policy or MaxAttempts <= 1 disables retries. Only replayable request
// bodies are resent; replayable means the bytes can be resent, not that the
// operation is idempotent. Transport failures and response-body errors,
// including interrupted successful streams, are never retried.
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
		if !isRetryableStatus(resp.StatusCode) || attempt == policy.MaxAttempts || (req.Body != nil && req.GetBody == nil) {
			return resp, nil
		}

		delay := policy.delay(attempt, resp)
		if err := drainAndCloseResponse(resp); err != nil {
			return nil, err
		}
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
	return policy
}

func (p Policy) delay(attempt int, resp *http.Response) time.Duration {
	if p.RespectRetryAfter {
		if retryAfter := parseRetryAfter(resp); retryAfter > 0 {
			return retryAfter
		}
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
	return delay
}

func isRetryableStatus(statusCode int) bool {
	switch statusCode {
	case http.StatusTooManyRequests,
		http.StatusInternalServerError,
		http.StatusBadGateway,
		http.StatusServiceUnavailable,
		http.StatusGatewayTimeout,
		529:
		return true
	default:
		return false
	}
}

func parseRetryAfter(resp *http.Response) time.Duration {
	if resp == nil {
		return 0
	}
	return wire.ParseRetryAfter(resp.Header.Get("Retry-After"), time.Now())
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

func drainAndCloseResponse(resp *http.Response) error {
	if resp == nil || resp.Body == nil {
		return nil
	}
	_, err := io.Copy(io.Discard, resp.Body)
	closeErr := resp.Body.Close()
	if err != nil {
		return err
	}
	return closeErr
}
