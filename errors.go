package litellm

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"strings"
	"time"
)

// ErrorType classifies an Error.
type ErrorType string

const (
	ErrorTypeAuth            ErrorType = "auth"
	ErrorTypeRateLimit       ErrorType = "rate_limit"
	ErrorTypeNetwork         ErrorType = "network"
	ErrorTypeValidation      ErrorType = "validation"
	ErrorTypeProvider        ErrorType = "provider"
	ErrorTypeTimeout         ErrorType = "timeout"
	ErrorTypeQuota           ErrorType = "quota"
	ErrorTypeModel           ErrorType = "model"
	ErrorTypeInternal        ErrorType = "internal"
	ErrorTypeContextOverflow ErrorType = "context_overflow"
	ErrorTypeOverloaded      ErrorType = "overloaded"
	ErrorTypeContentFilter   ErrorType = "content_filter"
)

// Error is the error type returned by the client and all providers.
type Error struct {
	Type       ErrorType
	Code       string
	Message    string
	Provider   string
	StatusCode int
	// Temporary describes a potentially transient failure, not permission to replay
	// a request. The provider may already have processed or billed the operation.
	Temporary bool
	// RetryAfter is the server-suggested delay; zero means unspecified.
	RetryAfter time.Duration
	Cause      error
}

// Error renders "provider: code: message". Messages carry no provider prefix.
func (e *Error) Error() string {
	msg := strings.TrimSpace(e.Message)
	switch {
	case e.Code != "" && msg != "":
		msg = e.Code + ": " + msg
	case msg == "" && e.Code != "":
		msg = e.Code
	case msg == "" && e.StatusCode != 0:
		msg = fmt.Sprintf("HTTP %d", e.StatusCode)
		if e.Type != "" {
			msg += " (" + string(e.Type) + ")"
		}
	case msg == "":
		msg = cmp.Or(string(e.Type), "litellm error")
	}
	if shouldShowCause(e) {
		if cause := strings.TrimSpace(e.Cause.Error()); cause != "" && !strings.Contains(msg, cause) {
			msg += ": " + cause
		}
	}
	if e.Provider != "" {
		msg = e.Provider + ": " + msg
	}
	return msg
}

func shouldShowCause(e *Error) bool {
	if e == nil || e.Cause == nil {
		return false
	}
	return e.Type == ErrorTypeNetwork || e.Type == ErrorTypeTimeout
}

// Unwrap returns Cause.
func (e *Error) Unwrap() error {
	return e.Cause
}

// NewError builds an error of the given type. provider may be empty and cause
// nil; Temporary follows the type.
func NewError(provider string, errorType ErrorType, message string, cause error) *Error {
	return &Error{Type: errorType, Provider: provider, Message: message, Cause: cause, Temporary: isTemporaryByType(errorType)}
}

// NewNetworkError builds a transport error. Context cancellation is a
// non-temporary network error and a deadline a timeout; other causes are
// temporary.
func NewNetworkError(provider, message string, cause error) *Error {
	if errors.Is(cause, context.Canceled) {
		return &Error{Type: ErrorTypeNetwork, Provider: provider, Message: message, Cause: cause, Temporary: false}
	}
	if errors.Is(cause, context.DeadlineExceeded) {
		return &Error{Type: ErrorTypeTimeout, Provider: provider, Message: message, Cause: cause, Temporary: false}
	}
	return &Error{Type: ErrorTypeNetwork, Provider: provider, Message: message, Cause: cause, Temporary: true}
}

// IsAuthError reports whether err wraps an *Error of type ErrorTypeAuth.
func IsAuthError(err error) bool { return isErrorType(err, ErrorTypeAuth) }

// IsRateLimitError reports whether err wraps an *Error of type ErrorTypeRateLimit.
func IsRateLimitError(err error) bool { return isErrorType(err, ErrorTypeRateLimit) }

// IsNetworkError reports whether err wraps an *Error of type ErrorTypeNetwork.
func IsNetworkError(err error) bool { return isErrorType(err, ErrorTypeNetwork) }

// IsValidationError reports whether err wraps an *Error of type ErrorTypeValidation.
func IsValidationError(err error) bool { return isErrorType(err, ErrorTypeValidation) }

// IsProviderError reports whether err wraps an *Error of type ErrorTypeProvider.
func IsProviderError(err error) bool { return isErrorType(err, ErrorTypeProvider) }

// IsTimeoutError reports whether err wraps an *Error of type ErrorTypeTimeout.
func IsTimeoutError(err error) bool { return isErrorType(err, ErrorTypeTimeout) }

// IsModelError reports whether err wraps an *Error of type ErrorTypeModel.
func IsModelError(err error) bool { return isErrorType(err, ErrorTypeModel) }

// IsContextOverflowError reports whether err wraps an *Error of type ErrorTypeContextOverflow.
func IsContextOverflowError(err error) bool { return isErrorType(err, ErrorTypeContextOverflow) }

// IsOverloadedError reports whether err wraps an *Error of type ErrorTypeOverloaded.
func IsOverloadedError(err error) bool { return isErrorType(err, ErrorTypeOverloaded) }

// IsContentFilterError reports whether err wraps an *Error of type ErrorTypeContentFilter.
func IsContentFilterError(err error) bool { return isErrorType(err, ErrorTypeContentFilter) }

// IsQuotaError reports whether err wraps an *Error of type ErrorTypeQuota.
func IsQuotaError(err error) bool { return isErrorType(err, ErrorTypeQuota) }

// IsInternalError reports whether err wraps an *Error of type ErrorTypeInternal.
func IsInternalError(err error) bool { return isErrorType(err, ErrorTypeInternal) }

// IsTemporaryError reports whether the failure may resolve over time. It does
// not establish that repeating the operation is safe, even before any output.
func IsTemporaryError(err error) bool {
	var e *Error
	return errors.As(err, &e) && e.Temporary
}

// RetryAfter returns the server-suggested delay carried by err, or zero.
func RetryAfter(err error) time.Duration {
	var e *Error
	if errors.As(err, &e) {
		return e.RetryAfter
	}
	return 0
}

// WrapError attributes err to provider. Existing *Error values keep their type
// and gain the provider when unset; context cancellation and deadlines become
// network and timeout errors; anything else becomes a fallback-typed error.
func WrapError(provider string, fallback ErrorType, err error) error {
	if err == nil {
		return nil
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return NewNetworkError(provider, err.Error(), err)
	}
	var e *Error
	if errors.As(err, &e) {
		if e.Provider == "" {
			copy := *e
			copy.Provider = provider
			return &copy
		}
		return err
	}
	return NewError(provider, fallback, err.Error(), err)
}

func isErrorType(err error, errorType ErrorType) bool {
	var e *Error
	return errors.As(err, &e) && e.Type == errorType
}

func isTemporaryByType(errorType ErrorType) bool {
	switch errorType {
	case ErrorTypeNetwork, ErrorTypeTimeout, ErrorTypeRateLimit, ErrorTypeOverloaded:
		return true
	default:
		return false
	}
}
