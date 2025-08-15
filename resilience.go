package litellm

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"math"
	"math/rand"
	"net"
	"net/http"
	"syscall"
	"time"
)

// ResilienceConfig holds network resilience configuration
type ResilienceConfig struct {
	MaxRetries     int           `json:"max_retries"`     // Maximum retry attempts, default 3
	InitialDelay   time.Duration `json:"initial_delay"`   // Initial delay, default 1 second
	MaxDelay       time.Duration `json:"max_delay"`       // Maximum delay, default 30 seconds
	Multiplier     float64       `json:"multiplier"`      // Backoff multiplier, default 2.0
	Jitter         bool          `json:"jitter"`          // Whether to add jitter, default true
	RequestTimeout time.Duration `json:"request_timeout"` // Single request timeout, default 30 seconds
	ConnectTimeout time.Duration `json:"connect_timeout"` // Connection timeout, default 10 seconds
}

// DefaultResilienceConfig returns default resilience configuration
func DefaultResilienceConfig() ResilienceConfig {
	return ResilienceConfig{
		MaxRetries:     3,
		InitialDelay:   1 * time.Second,
		MaxDelay:       30 * time.Second,
		Multiplier:     2.0,
		Jitter:         true,
		RequestTimeout: 30 * time.Second,
		ConnectTimeout: 10 * time.Second,
	}
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

	// Read and store original body for retries
	if req.Body != nil {
		var err error
		originalBody, err = io.ReadAll(req.Body)
		if err != nil {
			return nil, fmt.Errorf("failed to read request body: %w", err)
		}
		req.Body.Close()
	}

	for attempt := 0; attempt <= c.config.MaxRetries; attempt++ {
		// Restore request body for each attempt
		if originalBody != nil {
			req.Body = io.NopCloser(bytes.NewReader(originalBody))
		}

		resp, err := c.client.Do(req)
		if err == nil {
			if !isRetryableStatusCode(resp.StatusCode) {
				return resp, nil
			}
		}

		// Store error for potential retry
		if err != nil {
			lastErr = err
		} else {
			lastErr = fmt.Errorf("HTTP %d", resp.StatusCode)
			resp.Body.Close() // Close response body before retry
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

		// Calculate delay with exponential backoff
		delay := c.calculateDelay(attempt)

		// Wait with context cancellation support
		select {
		case <-time.After(delay):
			// Continue to next attempt
		case <-req.Context().Done():
			return nil, req.Context().Err()
		}
	}

	return nil, fmt.Errorf("request failed after %d attempts: %w", c.config.MaxRetries+1, lastErr)
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
		// Add random jitter of ±25%
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

	// Network errors are generally retryable
	if netErr, ok := err.(net.Error); ok {
		return netErr.Timeout() || netErr.Temporary()
	}

	// System call errors
	if opErr, ok := err.(*net.OpError); ok {
		if syscallErr, ok := opErr.Err.(*syscall.Errno); ok {
			switch *syscallErr {
			case syscall.ECONNREFUSED, syscall.ECONNRESET, syscall.ETIMEDOUT:
				return true
			}
		}
	}

	// Context errors are not retryable (user cancelled or deadline exceeded)
	if err == context.Canceled || err == context.DeadlineExceeded {
		return false
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
