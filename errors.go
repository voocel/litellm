package litellm

import (
	"fmt"
	"net/http"
)

// ErrorType represents different categories of errors
type ErrorType string

const (
	ErrorTypeAuth       ErrorType = "auth"       // Authentication/authorization errors
	ErrorTypeRateLimit  ErrorType = "rate_limit" // Rate limiting errors
	ErrorTypeNetwork    ErrorType = "network"    // Network connectivity errors
	ErrorTypeValidation ErrorType = "validation" // Request validation errors
	ErrorTypeProvider   ErrorType = "provider"   // Provider-specific errors
	ErrorTypeTimeout    ErrorType = "timeout"    // Timeout errors
	ErrorTypeQuota      ErrorType = "quota"      // Quota/billing errors
	ErrorTypeModel      ErrorType = "model"      // Model not found/supported errors
	ErrorTypeInternal   ErrorType = "internal"   // Internal library errors
)

// LiteLLMError represents a structured error with categorization
type LiteLLMError struct {
	Type     ErrorType `json:"type"`
	Code     string    `json:"code,omitempty"`
	Message  string    `json:"message"`
	Provider string    `json:"provider,omitempty"`
	Model    string    `json:"model,omitempty"`
	Cause    error     `json:"-"` // Original error, not serialized

	// HTTP details if applicable
	StatusCode int               `json:"status_code,omitempty"`
	Headers    map[string]string `json:"headers,omitempty"`

	// Retry information
	Retryable  bool `json:"retryable"`
	RetryAfter int  `json:"retry_after,omitempty"` // seconds
}

// Error implements the error interface
func (e *LiteLLMError) Error() string {
	if e.Provider != "" {
		return fmt.Sprintf("[%s:%s] %s", e.Provider, e.Type, e.Message)
	}
	return fmt.Sprintf("[%s] %s", e.Type, e.Message)
}

// Unwrap returns the underlying error
func (e *LiteLLMError) Unwrap() error {
	return e.Cause
}

// Is checks if the error matches a specific type
func (e *LiteLLMError) Is(target error) bool {
	if t, ok := target.(*LiteLLMError); ok {
		return e.Type == t.Type
	}
	return false
}

// IsRetryable returns whether this error can be retried
func (e *LiteLLMError) IsRetryable() bool {
	return e.Retryable
}

// NewError creates a new LiteLLMError
func NewError(errorType ErrorType, message string) *LiteLLMError {
	return &LiteLLMError{
		Type:      errorType,
		Message:   message,
		Retryable: isRetryableByType(errorType),
	}
}

// NewErrorWithCause creates a new LiteLLMError with an underlying cause
func NewErrorWithCause(errorType ErrorType, message string, cause error) *LiteLLMError {
	return &LiteLLMError{
		Type:      errorType,
		Message:   message,
		Cause:     cause,
		Retryable: isRetryableByType(errorType),
	}
}

// NewProviderError creates a provider-specific error
func NewProviderError(provider string, errorType ErrorType, message string) *LiteLLMError {
	return &LiteLLMError{
		Type:      errorType,
		Provider:  provider,
		Message:   message,
		Retryable: isRetryableByType(errorType),
	}
}

// NewHTTPError creates an error from HTTP response
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

// NewAuthError creates an authentication error
func NewAuthError(provider, message string) *LiteLLMError {
	return NewProviderError(provider, ErrorTypeAuth, message)
}

// NewValidationError creates a validation error
func NewValidationError(provider, message string) *LiteLLMError {
	return NewProviderError(provider, ErrorTypeValidation, message)
}

// NewRateLimitError creates a rate limit error
func NewRateLimitError(provider, message string, retryAfter int) *LiteLLMError {
	return &LiteLLMError{
		Type:       ErrorTypeRateLimit,
		Provider:   provider,
		Message:    message,
		Retryable:  true,
		RetryAfter: retryAfter,
	}
}

// NewModelError creates a model-related error
func NewModelError(provider, model, message string) *LiteLLMError {
	return &LiteLLMError{
		Type:      ErrorTypeModel,
		Provider:  provider,
		Model:     model,
		Message:   message,
		Retryable: false,
	}
}

// NewNetworkError creates a network error
func NewNetworkError(provider, message string, cause error) *LiteLLMError {
	return &LiteLLMError{
		Type:      ErrorTypeNetwork,
		Provider:  provider,
		Message:   message,
		Cause:     cause,
		Retryable: true,
	}
}

// NewTimeoutError creates a timeout error
func NewTimeoutError(provider, message string) *LiteLLMError {
	return &LiteLLMError{
		Type:      ErrorTypeTimeout,
		Provider:  provider,
		Message:   message,
		Retryable: true,
	}
}

// classifyHTTPError maps HTTP status codes to error types
func classifyHTTPError(statusCode int) ErrorType {
	switch {
	case statusCode == http.StatusUnauthorized:
		return ErrorTypeAuth
	case statusCode == http.StatusForbidden:
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

// isRetryableByType determines if an error type is generally retryable
func isRetryableByType(errorType ErrorType) bool {
	switch errorType {
	case ErrorTypeNetwork, ErrorTypeTimeout, ErrorTypeRateLimit:
		return true
	case ErrorTypeProvider: // 5xx errors are typically retryable
		return true
	case ErrorTypeAuth, ErrorTypeValidation, ErrorTypeModel, ErrorTypeQuota:
		return false
	default:
		return false
	}
}

// IsAuthError checks if error is authentication related
func IsAuthError(err error) bool {
	if e, ok := err.(*LiteLLMError); ok {
		return e.Type == ErrorTypeAuth
	}
	return false
}

// IsRateLimitError checks if error is rate limit related
func IsRateLimitError(err error) bool {
	if e, ok := err.(*LiteLLMError); ok {
		return e.Type == ErrorTypeRateLimit
	}
	return false
}

// IsNetworkError checks if error is network related
func IsNetworkError(err error) bool {
	if e, ok := err.(*LiteLLMError); ok {
		return e.Type == ErrorTypeNetwork
	}
	return false
}

// IsValidationError checks if error is validation related
func IsValidationError(err error) bool {
	if e, ok := err.(*LiteLLMError); ok {
		return e.Type == ErrorTypeValidation
	}
	return false
}

// IsModelError checks if error is model related
func IsModelError(err error) bool {
	if e, ok := err.(*LiteLLMError); ok {
		return e.Type == ErrorTypeModel
	}
	return false
}

// IsRetryableError checks if an error is retryable
func IsRetryableError(err error) bool {
	if e, ok := err.(*LiteLLMError); ok {
		return e.IsRetryable()
	}
	// Unknown errors are not retryable by default
	return false
}

// GetRetryAfter extracts retry-after duration from rate limit errors
func GetRetryAfter(err error) int {
	if e, ok := err.(*LiteLLMError); ok && e.Type == ErrorTypeRateLimit {
		return e.RetryAfter
	}
	return 0
}

// WrapError wraps an existing error as a LiteLLMError if it isn't already
func WrapError(err error, provider string) error {
	if err == nil {
		return nil
	}

	// Already a LiteLLMError
	if e, ok := err.(*LiteLLMError); ok {
		if e.Provider == "" {
			e.Provider = provider
		}
		return e
	}

	// Wrap as internal error
	return NewErrorWithCause(ErrorTypeInternal, err.Error(), err)
}
