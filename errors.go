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
	ErrorTypeAuth       ErrorType = "auth"
	ErrorTypeRateLimit  ErrorType = "rate_limit"
	ErrorTypeNetwork    ErrorType = "network"
	ErrorTypeValidation ErrorType = "validation"
	ErrorTypeProvider   ErrorType = "provider"
	// ErrorTypeTimeout is a call that ran out the time its caller gave it:
	// the context's deadline or the HTTP client's timeout. A server that
	// timed out is ErrorTypeProvider.
	ErrorTypeTimeout         ErrorType = "timeout"
	ErrorTypeQuota           ErrorType = "quota"
	ErrorTypeModel           ErrorType = "model"
	ErrorTypeInternal        ErrorType = "internal"
	ErrorTypeContextOverflow ErrorType = "context_overflow"
	ErrorTypeOverloaded      ErrorType = "overloaded"
	ErrorTypeContentFilter   ErrorType = "content_filter"
	// ErrorTypeCanceled is a call the caller's context cancelled.
	ErrorTypeCanceled ErrorType = "canceled"
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

// Error renders "provider: code: message (HTTP status)". Messages carry no
// provider prefix. The status tells a caller what a vague vendor message,
// such as "Provider returned error", does not.
func (e *Error) Error() string {
	msg := strings.TrimSpace(e.Message)
	status := ""
	if e.StatusCode != 0 {
		status = fmt.Sprintf("HTTP %d", e.StatusCode)
	}
	switch {
	case e.Code != "" && msg != "":
		msg = e.Code + ": " + msg
	case msg == "" && e.Code != "":
		msg = e.Code
	case msg == "" && status != "":
		msg = status
		if e.Type != "" {
			msg += " (" + string(e.Type) + ")"
		}
	case msg == "":
		msg = cmp.Or(string(e.Type), "litellm error")
	}
	if status != "" && !strings.Contains(msg, status) {
		msg += " (" + status + ")"
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
	return e.Cause != nil && (e.Type == ErrorTypeNetwork || e.Type == ErrorTypeTimeout)
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

// NewNetworkError builds a transport error. Context cancellation is
// ErrorTypeCanceled and a deadline a timeout, neither temporary; other causes
// are temporary.
func NewNetworkError(provider, message string, cause error) *Error {
	if errors.Is(cause, context.Canceled) {
		return &Error{Type: ErrorTypeCanceled, Provider: provider, Message: message, Cause: cause, Temporary: false}
	}
	if errors.Is(cause, context.DeadlineExceeded) {
		return &Error{Type: ErrorTypeTimeout, Provider: provider, Message: message, Cause: cause, Temporary: false}
	}
	return &Error{Type: ErrorTypeNetwork, Provider: provider, Message: message, Cause: cause, Temporary: true}
}

// ErrorTypeOf returns the type of the *Error err wraps, or "" when it wraps
// none.
func ErrorTypeOf(err error) ErrorType {
	var e *Error
	if errors.As(err, &e) {
		return e.Type
	}
	return ""
}

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
// canceled and timeout errors; anything else becomes a fallback-typed error.
func WrapError(provider string, fallback ErrorType, err error) error {
	if err == nil {
		return nil
	}
	var e *Error
	if isContextError(err) {
		// Rebuild an *Error from its parts: its rendering already carries the
		// provider prefix.
		message, cause := err.Error(), err
		if errors.As(err, &e) && isContextError(e.Cause) {
			provider, message, cause = cmp.Or(e.Provider, provider), e.Message, e.Cause
		}
		return NewNetworkError(provider, message, cause)
	}
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

func isContextError(err error) bool {
	return errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded)
}

func isTemporaryByType(errorType ErrorType) bool {
	switch errorType {
	case ErrorTypeNetwork, ErrorTypeRateLimit, ErrorTypeOverloaded:
		return true
	default:
		return false
	}
}
