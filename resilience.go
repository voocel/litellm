package litellm

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"net"
	"net/http"
	"strconv"
	"syscall"
	"time"

	"github.com/voocel/litellm/providers"
)

// ResilienceConfig and defaults are sourced from providers; re-exported here to keep the public API small.
type ResilienceConfig = providers.ResilienceConfig

func DefaultResilienceConfig() ResilienceConfig {
	return providers.DefaultResilienceConfig()
}

// ResilientHTTPClient wraps http.Client with retry logic
type ResilientHTTPClient struct {
	client *http.Client
	config ResilienceConfig
}

// NewResilientHTTPClient creates a new resilient HTTP client
func NewResilientHTTPClient(config ResilienceConfig) *ResilientHTTPClient {
	return &ResilientHTTPClient{
		client: &http.Client{
			Timeout: config.RequestTimeout,
			Transport: &http.Transport{
				DialContext: (&net.Dialer{
					Timeout: config.ConnectTimeout,
				}).DialContext,
				MaxIdleConns:        100,
				MaxIdleConnsPerHost: 10,
				IdleConnTimeout:     90 * time.Second,
			},
		},
		config: config,
	}
}

// Do executes HTTP request with retry logic
func (c *ResilientHTTPClient) Do(req *http.Request) (*http.Response, error) {
	var lastErr error
	var originalBody []byte

	// Read and store original body for retries only if GetBody is not available
	if req.Body != nil && req.GetBody == nil {
		var err error
		originalBody, err = io.ReadAll(req.Body)
		if err != nil {
			return nil, fmt.Errorf("failed to read request body: %w", err)
		}
		req.Body.Close()
	}

	for attempt := 0; attempt <= c.config.MaxRetries; attempt++ {
		// Restore request body for each retry attempt
		if req.GetBody != nil {
			body, err := req.GetBody()
			if err != nil {
				return nil, fmt.Errorf("failed to get request body: %w", err)
			}
			req.Body = body
		} else if originalBody != nil {
			req.Body = io.NopCloser(bytes.NewReader(originalBody))
		}

		resp, err := c.client.Do(req)
		if err == nil {
			if !isRetryableStatusCode(resp.StatusCode) {
				return resp, nil
			}
		}

		if err != nil {
			lastErr = err
		} else {
			bodyBytes, readErr := io.ReadAll(resp.Body)
			resp.Body.Close()
			if readErr != nil || len(bodyBytes) == 0 {
				lastErr = fmt.Errorf("HTTP %d", resp.StatusCode)
			} else {
				lastErr = fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(bodyBytes))
			}
		}

		// Don't retry on last attempt or non-retryable errors
		if attempt == c.config.MaxRetries {
			break
		}

		// For HTTP errors, check if status code is retryable
		if err == nil && !isRetryableStatusCode(resp.StatusCode) {
			break
		}

		// For network errors, check if error is retryable
		if err != nil && !isRetryableError(err) {
			break
		}

		// Prefer server-provided Retry-After over exponential backoff
		delay := parseRetryAfter(resp)
		if delay == 0 {
			delay = c.calculateDelay(attempt)
		}

		// Wait with context cancellation support
		timer := time.NewTimer(delay)
		select {
		case <-timer.C:
			// Continue to next attempt
		case <-req.Context().Done():
			timer.Stop()
			return nil, req.Context().Err()
		}
	}

	return nil, lastErr
}

// calculateDelay calculates retry delay with exponential backoff and jitter
func (c *ResilientHTTPClient) calculateDelay(attempt int) time.Duration {
	// Exponential backoff: delay = initial * multiplier^attempt
	delay := float64(c.config.InitialDelay) * math.Pow(c.config.Multiplier, float64(attempt))
	if delay > float64(c.config.MaxDelay) {
		delay = float64(c.config.MaxDelay)
	}

	// Add jitter to avoid thundering herd
	if c.config.Jitter {
		// Add random jitter of +/-25%
		jitter := delay * 0.25 * (2*rand.Float64() - 1)
		delay += jitter
		if delay < 0 {
			delay = float64(c.config.InitialDelay)
		}
	}

	return time.Duration(delay)
}

// isRetryableError determines if an error should trigger a retry
func isRetryableError(err error) bool {
	if err == nil {
		return false
	}

	// Context errors are not retryable (user cancelled or deadline exceeded)
	// Check this FIRST before net.Error, as context.DeadlineExceeded implements net.Error
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return false
	}

	// Network errors are generally retryable
	var netErr net.Error
	if errors.As(err, &netErr) {
		return netErr.Timeout()
	}

	// System call errors
	var opErr *net.OpError
	if errors.As(err, &opErr) {
		var syscallErr *syscall.Errno
		if errors.As(opErr.Err, &syscallErr) {
			switch {
			case errors.Is(*syscallErr, syscall.ECONNREFUSED), errors.Is(*syscallErr, syscall.ECONNRESET), errors.Is(*syscallErr, syscall.ETIMEDOUT):
				return true
			}
		}
	}

	return false
}

// isRetryableStatusCode determines if an HTTP status code should trigger a retry
func isRetryableStatusCode(statusCode int) bool {
	switch statusCode {
	case http.StatusTooManyRequests, // 429
		http.StatusInternalServerError, // 500
		http.StatusBadGateway,          // 502
		http.StatusServiceUnavailable,  // 503
		http.StatusGatewayTimeout:      // 504
		return true
	default:
		return false
	}
}

// parseRetryAfter extracts the retry delay from a Retry-After response header.
// Supports both delta-seconds ("120") and HTTP-date formats (RFC 7231 §7.1.3).
// Returns 0 if the response is nil, the header is absent, or the value is unparseable.
func parseRetryAfter(resp *http.Response) time.Duration {
	if resp == nil {
		return 0
	}
	val := resp.Header.Get("Retry-After")
	if val == "" {
		return 0
	}
	// Delta-seconds (most common for API rate limits)
	if seconds, err := strconv.Atoi(val); err == nil && seconds > 0 {
		return time.Duration(seconds) * time.Second
	}
	// HTTP-date fallback
	if t, err := http.ParseTime(val); err == nil {
		if delay := time.Until(t); delay > 0 {
			return delay
		}
	}
	return 0
}
