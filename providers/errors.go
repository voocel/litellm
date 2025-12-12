package providers

import (
	"errors"
	"fmt"
	"net"
	"net/http"
	"strconv"
	"strings"
)

// ErrorType categorizes errors.
type ErrorType string

const (
	ErrorTypeAuth       ErrorType = "auth"       // Auth/authorization errors
	ErrorTypeRateLimit  ErrorType = "rate_limit" // Rate limit errors
	ErrorTypeNetwork    ErrorType = "network"    // Network connectivity errors
	ErrorTypeValidation ErrorType = "validation" // Request validation errors
	ErrorTypeProvider   ErrorType = "provider"   // Upstream provider errors
	ErrorTypeTimeout    ErrorType = "timeout"    // Timeout errors
	ErrorTypeQuota      ErrorType = "quota"      // Quota/billing errors
	ErrorTypeModel      ErrorType = "model"      // Model not found/unsupported errors
	ErrorTypeInternal   ErrorType = "internal"   // Internal library errors
)

// LiteLLMError is a structured error with categorization and retry hints.
type LiteLLMError struct {
	Type     ErrorType `json:"type"`
	Code     string    `json:"code,omitempty"`
	Message  string    `json:"message"`
	Provider string    `json:"provider,omitempty"`
	Model    string    `json:"model,omitempty"`
	Cause    error     `json:"-"` // Original error, not serialized.

	// HTTP details (if applicable).
	StatusCode int               `json:"status_code,omitempty"`
	Headers    map[string]string `json:"headers,omitempty"`

	// Retry hints.
	Retryable  bool `json:"retryable"`
	RetryAfter int  `json:"retry_after,omitempty"` // seconds
}

func (e *LiteLLMError) Error() string {
	if e.Provider != "" {
		return fmt.Sprintf("[%s:%s] %s", e.Provider, e.Type, e.Message)
	}
	return fmt.Sprintf("[%s] %s", e.Type, e.Message)
}

func (e *LiteLLMError) Unwrap() error { return e.Cause }

func (e *LiteLLMError) Is(target error) bool {
	var t *LiteLLMError
	if errors.As(target, &t) {
		return e.Type == t.Type
	}
	return false
}

func (e *LiteLLMError) IsRetryable() bool { return e.Retryable }

func NewError(errorType ErrorType, message string) *LiteLLMError {
	return &LiteLLMError{
		Type:      errorType,
		Message:   message,
		Retryable: isRetryableByType(errorType),
	}
}

func NewErrorWithCause(errorType ErrorType, message string, cause error) *LiteLLMError {
	return &LiteLLMError{
		Type:      errorType,
		Message:   message,
		Cause:     cause,
		Retryable: isRetryableByType(errorType),
	}
}

func NewProviderError(provider string, errorType ErrorType, message string) *LiteLLMError {
	return &LiteLLMError{
		Type:      errorType,
		Provider:  provider,
		Message:   message,
		Retryable: isRetryableByType(errorType),
	}
}

func NewHTTPError(provider string, statusCode int, message string) *LiteLLMError {
	errorType := classifyHTTPError(statusCode)
	return &LiteLLMError{
		Type:       errorType,
		Provider:   provider,
		Message:    message,
		StatusCode: statusCode,
		Retryable:  isRetryableByType(errorType),
	}
}

func NewAuthError(provider, message string) *LiteLLMError {
	return NewProviderError(provider, ErrorTypeAuth, message)
}

func NewValidationError(provider, message string) *LiteLLMError {
	return NewProviderError(provider, ErrorTypeValidation, message)
}

func NewRateLimitError(provider, message string, retryAfter int) *LiteLLMError {
	return &LiteLLMError{
		Type:       ErrorTypeRateLimit,
		Provider:   provider,
		Message:    message,
		Retryable:  true,
		RetryAfter: retryAfter,
	}
}

func NewModelError(provider, model, message string) *LiteLLMError {
	return &LiteLLMError{
		Type:      ErrorTypeModel,
		Provider:  provider,
		Model:     model,
		Message:   message,
		Retryable: false,
	}
}

func NewNetworkError(provider, message string, cause error) *LiteLLMError {
	return &LiteLLMError{
		Type:      ErrorTypeNetwork,
		Provider:  provider,
		Message:   message,
		Cause:     cause,
		Retryable: true,
	}
}

func NewTimeoutError(provider, message string) *LiteLLMError {
	return &LiteLLMError{
		Type:      ErrorTypeTimeout,
		Provider:  provider,
		Message:   message,
		Retryable: true,
	}
}

func classifyHTTPError(statusCode int) ErrorType {
	switch {
	case statusCode == http.StatusUnauthorized, statusCode == http.StatusForbidden:
		return ErrorTypeAuth
	case statusCode == http.StatusTooManyRequests:
		return ErrorTypeRateLimit
	case statusCode == http.StatusPaymentRequired:
		return ErrorTypeQuota
	case statusCode == http.StatusNotFound:
		return ErrorTypeModel
	case statusCode == http.StatusRequestTimeout:
		return ErrorTypeTimeout
	case statusCode == http.StatusBadRequest:
		return ErrorTypeValidation
	case statusCode >= 500:
		return ErrorTypeProvider
	default:
		return ErrorTypeProvider
	}
}

func isRetryableByType(errorType ErrorType) bool {
	switch errorType {
	case ErrorTypeNetwork, ErrorTypeTimeout, ErrorTypeRateLimit:
		return true
	case ErrorTypeProvider:
		return true
	case ErrorTypeAuth, ErrorTypeValidation, ErrorTypeModel, ErrorTypeQuota:
		return false
	default:
		return false
	}
}

func IsAuthError(err error) bool {
	var e *LiteLLMError
	return errors.As(err, &e) && e.Type == ErrorTypeAuth
}

func IsRateLimitError(err error) bool {
	var e *LiteLLMError
	return errors.As(err, &e) && e.Type == ErrorTypeRateLimit
}

func IsNetworkError(err error) bool {
	var e *LiteLLMError
	return errors.As(err, &e) && e.Type == ErrorTypeNetwork
}

func IsValidationError(err error) bool {
	var e *LiteLLMError
	return errors.As(err, &e) && e.Type == ErrorTypeValidation
}

func IsModelError(err error) bool {
	var e *LiteLLMError
	return errors.As(err, &e) && e.Type == ErrorTypeModel
}

func IsRetryableError(err error) bool {
	var e *LiteLLMError
	if errors.As(err, &e) {
		return e.IsRetryable()
	}
	return false
}

func GetRetryAfter(err error) int {
	var e *LiteLLMError
	if errors.As(err, &e) && e.Type == ErrorTypeRateLimit {
		return e.RetryAfter
	}
	return 0
}

// WrapError wraps any error into LiteLLMError (and fills Provider if already wrapped).
func WrapError(err error, provider string) error {
	if err == nil {
		return nil
	}

	var e *LiteLLMError
	if errors.As(err, &e) {
		if e.Provider == "" {
			e.Provider = provider
		}
		return e
	}

	// Network errors / timeouts.
	var netErr net.Error
	if errors.As(err, &netErr) {
		if netErr.Timeout() {
			return NewTimeoutError(provider, err.Error())
		}
		return NewNetworkError(provider, err.Error(), err)
	}

	msgLower := strings.ToLower(err.Error())
	if status := extractHTTPStatus(msgLower); status != 0 {
		return NewHTTPError(provider, status, err.Error())
	}

	// Fallback to provider error.
	return NewProviderError(provider, ErrorTypeProvider, err.Error())
}

// extractHTTPStatus tries to extract a 3-digit HTTP status code from an error string.
func extractHTTPStatus(msgLower string) int {
	// Common patterns: "http 429", "api error 500", "status code 401".
	fields := strings.Fields(msgLower)
	for i, f := range fields {
		if f == "http" || f == "status" || f == "code" || f == "error" {
			if i+1 < len(fields) {
				if n, err := strconv.Atoi(strings.Trim(fields[i+1], ":")); err == nil && n >= 100 && n <= 599 {
					return n
				}
			}
		}
		// Also catch bare codes like "429".
		if n, err := strconv.Atoi(strings.Trim(f, ":")); err == nil && n >= 100 && n <= 599 {
			return n
		}
	}
	return 0
}
