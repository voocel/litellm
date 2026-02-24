package litellm

import "github.com/voocel/litellm/providers"

// Error types and constructors are sourced from providers; this file is a thin re-export.
type ErrorType = providers.ErrorType
type LiteLLMError = providers.LiteLLMError

const (
	ErrorTypeAuth            ErrorType = providers.ErrorTypeAuth
	ErrorTypeRateLimit       ErrorType = providers.ErrorTypeRateLimit
	ErrorTypeNetwork         ErrorType = providers.ErrorTypeNetwork
	ErrorTypeValidation      ErrorType = providers.ErrorTypeValidation
	ErrorTypeProvider        ErrorType = providers.ErrorTypeProvider
	ErrorTypeTimeout         ErrorType = providers.ErrorTypeTimeout
	ErrorTypeQuota           ErrorType = providers.ErrorTypeQuota
	ErrorTypeModel           ErrorType = providers.ErrorTypeModel
	ErrorTypeInternal        ErrorType = providers.ErrorTypeInternal
	ErrorTypeContextOverflow ErrorType = providers.ErrorTypeContextOverflow
)

func NewError(errorType ErrorType, message string) *LiteLLMError {
	return providers.NewError(errorType, message)
}

func NewErrorWithCause(errorType ErrorType, message string, cause error) *LiteLLMError {
	return providers.NewErrorWithCause(errorType, message, cause)
}

func NewProviderError(provider string, errorType ErrorType, message string) *LiteLLMError {
	return providers.NewProviderError(provider, errorType, message)
}

func NewHTTPError(provider string, statusCode int, message string) *LiteLLMError {
	return providers.NewHTTPError(provider, statusCode, message)
}

func NewAuthError(provider, message string) *LiteLLMError {
	return providers.NewAuthError(provider, message)
}

func NewValidationError(provider, message string) *LiteLLMError {
	return providers.NewValidationError(provider, message)
}

func NewRateLimitError(provider, message string, retryAfter int) *LiteLLMError {
	return providers.NewRateLimitError(provider, message, retryAfter)
}

func NewModelError(provider, model, message string) *LiteLLMError {
	return providers.NewModelError(provider, model, message)
}

func NewNetworkError(provider, message string, cause error) *LiteLLMError {
	return providers.NewNetworkError(provider, message, cause)
}

func NewTimeoutError(provider, message string) *LiteLLMError {
	return providers.NewTimeoutError(provider, message)
}

func IsAuthError(err error) bool            { return providers.IsAuthError(err) }
func IsRateLimitError(err error) bool       { return providers.IsRateLimitError(err) }
func IsNetworkError(err error) bool         { return providers.IsNetworkError(err) }
func IsValidationError(err error) bool      { return providers.IsValidationError(err) }
func IsModelError(err error) bool           { return providers.IsModelError(err) }
func IsContextOverflowError(err error) bool { return providers.IsContextOverflowError(err) }
func IsRetryableError(err error) bool       { return providers.IsRetryableError(err) }
func GetRetryAfter(err error) int           { return providers.GetRetryAfter(err) }

func WrapError(err error, provider string) error {
	return providers.WrapError(err, provider)
}
