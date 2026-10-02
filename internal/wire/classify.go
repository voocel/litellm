package wire

import (
	"encoding/json"
	"net/http"
	"strings"
	"time"

	"github.com/voocel/litellm"
)

// HTTPError classifies a non-2xx response from its status, headers and body.
// header may be nil. RetryAfter is the Retry-After header's wait or, without
// one, the retryDelay a Google API error body suggests, as Gemini sends.
func HTTPError(provider string, statusCode int, header http.Header, body string) *litellm.Error {
	code, message := parseHTTPErrorMessage(body)
	errorType := classifyHTTPError(statusCode)
	if vendorType, ok := vendorErrorType(code, message); ok {
		errorType = vendorType
	}
	err := litellm.NewError(provider, errorType, message, nil)
	err.Code, err.StatusCode = code, statusCode
	err.Temporary = isTemporaryHTTPError(statusCode, errorType)
	err.RetryAfter = ParseRetryAfter(header.Get("Retry-After"), time.Now())
	if err.RetryAfter == 0 {
		err.RetryAfter = googleRetryDelay(body)
	}
	return err
}

// googleRetryDelay returns the wait in the google.rpc.RetryInfo detail of a
// Google API error body, or zero.
func googleRetryDelay(body string) time.Duration {
	var payload struct {
		Error struct {
			Details []struct {
				Type       string `json:"@type"`
				RetryDelay string `json:"retryDelay"`
			} `json:"details"`
		} `json:"error"`
	}
	if json.Unmarshal([]byte(body), &payload) != nil {
		return 0
	}
	for _, d := range payload.Error.Details {
		if d.Type != "type.googleapis.com/google.rpc.RetryInfo" {
			continue
		}
		if delay, err := time.ParseDuration(d.RetryDelay); err == nil && delay > 0 {
			return delay
		}
	}
	return 0
}

func parseHTTPErrorMessage(body string) (string, string) {
	body = strings.TrimSpace(body)
	if body == "" {
		return "", ""
	}

	var payload struct {
		Error any `json:"error"`
	}
	if err := json.Unmarshal([]byte(body), &payload); err != nil || payload.Error == nil {
		return "", body
	}

	switch e := payload.Error.(type) {
	case string:
		return "", strings.TrimSpace(e)
	case map[string]any:
		code := stringField(e, "code")
		msg := stringField(e, "message")
		meta, _ := e["metadata"].(map[string]any)
		if code == "" && meta != nil {
			// OpenRouter uses a numeric code and names the failure here.
			code = stringField(meta, "error_type")
		}
		// Aggregator gateways (OpenRouter) wrap the upstream provider's real
		// error under error.metadata: message is a generic "Provider returned
		// error" while metadata.raw carries the actual reason (unsupported
		// response_format, context overflow, ...). Dropping raw makes such
		// failures undiagnosable, so surface it with the serving provider name.
		if meta != nil {
			if raw := stringField(meta, "raw"); raw != "" {
				if pn := stringField(meta, "provider_name"); pn != "" {
					raw = pn + ": " + raw
				}
				if msg == "" {
					msg = raw
				} else {
					msg += " — " + raw
				}
			}
		}
		if msg == "" {
			msg = body
		}
		return code, msg
	default:
		return "", body
	}
}

func stringField(m map[string]any, key string) string {
	v, ok := m[key].(string)
	if !ok {
		return ""
	}
	return strings.TrimSpace(v)
}

// StreamError classifies an error reported inside a response stream, where no
// HTTP status is available. code is the vendor error code or type.
func StreamError(provider, code, message string) *litellm.Error {
	errorType, ok := vendorErrorType(code, message)
	if !ok {
		if errorType = streamErrorTypes[strings.ToLower(code)]; errorType == "" {
			errorType = litellm.ErrorTypeProvider
		}
	}
	err := litellm.NewError(provider, errorType, message, nil)
	err.Code = code
	// A server fault reported mid-stream is as transient as its HTTP 5xx.
	err.Temporary = err.Temporary || streamServerFaults[strings.ToLower(code)]
	return err
}

// streamServerFaults are the in-stream codes of server faults, which an HTTP
// response would report with a retryable 5xx status.
var streamServerFaults = map[string]bool{
	"api_error":                 true,
	"internalserverexception":   true,
	"modelstreamerrorexception": true,
	"server_error":              true,
}

// streamErrorTypes maps documented in-stream codes: Anthropic error types,
// Bedrock ConverseStream exception names and OpenAI Responses error codes.
var streamErrorTypes = map[string]litellm.ErrorType{
	"overloaded_error":            litellm.ErrorTypeOverloaded,
	"rate_limit_error":            litellm.ErrorTypeRateLimit,
	"api_error":                   litellm.ErrorTypeProvider,
	"invalid_request_error":       litellm.ErrorTypeValidation,
	"throttlingexception":         litellm.ErrorTypeRateLimit,
	"validationexception":         litellm.ErrorTypeValidation,
	"serviceunavailableexception": litellm.ErrorTypeOverloaded,
	"internalserverexception":     litellm.ErrorTypeProvider,
	"modelstreamerrorexception":   litellm.ErrorTypeProvider,
	"too_many_requests":           litellm.ErrorTypeRateLimit,
	"rate_limit_exceeded":         litellm.ErrorTypeRateLimit,
	"server_is_overloaded":        litellm.ErrorTypeOverloaded,
	"service_unavailable_error":   litellm.ErrorTypeOverloaded,
	"server_error":                litellm.ErrorTypeProvider,
}

// vendorErrorType detects deterministic rejections that providers signal with
// vendor codes or fixed messages rather than a common status; proxies often
// rewrite the status to a retryable 429/5xx. Context overflow is checked first
// because OpenRouter reports it with invalid_prompt, a content-filter code.
func vendorErrorType(code, message string) (litellm.ErrorType, bool) {
	haystack := strings.ToLower(code + " " + message)
	// Zhipu's numeric code must match exactly to avoid token counts or IDs.
	if code == "1261" || containsAny(haystack, contextOverflowTokens) {
		return litellm.ErrorTypeContextOverflow, true
	}
	if containsAny(haystack, contentFilterTokens) {
		return litellm.ErrorTypeContentFilter, true
	}
	if containsAny(haystack, quotaTokens) {
		return litellm.ErrorTypeQuota, true
	}
	if containsAny(haystack, authTokens) {
		return litellm.ErrorTypeAuth, true
	}
	return "", false
}

// contextOverflowTokens are vendor codes and fixed messages for inputs that
// exceed the context window: OpenAI and OpenRouter (context_length_exceeded,
// "maximum context length", "exceeds the context window", the configured
// input limit), DeepSeek, Anthropic and Claude on Bedrock ("prompt is too
// long", "exceed context limit"), Bedrock, Gemini ("input token count"), xAI,
// DashScope, MiniMax, Zhipu and Ollama.
var contextOverflowTokens = []string{
	"context_length_exceeded",
	"maximum context length",
	"exceeds the context window",
	"input tokens exceed the configured limit",
	"prompt is too long",
	"exceed context limit",
	"input is too long for requested model",
	"too many total text bytes",
	"input token count",
	"maximum prompt length",
	"range of input length",
	"context window exceeds limit",
	"prompt too long",
	"prompt exceeds max length",
	"prompt 超长",
	"longer than the context length",
	"exceeds the context length",
}

// contentFilterTokens are stable vendor error codes for content moderation
// rejections: Azure (content_filter), OpenAI (content_policy_violation;
// invalid_prompt on reasoning models), Zhipu-style gateways
// (sensitive_words_detected), DashScope/Qwen (data_inspection_failed in
// OpenAI-compat mode, InternalError.Algo.DataInspectionFailed natively).
// Anthropic has no dedicated code; its fixed message "Output blocked by
// content filtering policy" is matched instead.
var contentFilterTokens = []string{
	"content_filter",
	"content_policy",
	"sensitive_words",
	"data_inspection_failed",
	"datainspectionfailed",
	"invalid_prompt",
	"content filtering policy",
}

// quotaTokens are OpenAI billing codes, sent with 429 but not cleared by
// retrying: insufficient_quota and its specific successors.
var quotaTokens = []string{
	"insufficient_quota",
	"credit_balance_exhausted",
	"spend_limit_exceeded",
	"usage_limit_exceeded",
}

// authTokens are invalid-key rejections sent without 401/403: Gemini answers
// 400 with a fixed message and reason API_KEY_INVALID.
var authTokens = []string{
	"api key not valid",
	"api_key_invalid",
}

func containsAny(haystack string, tokens []string) bool {
	for _, token := range tokens {
		if strings.Contains(haystack, token) {
			return true
		}
	}
	return false
}

func classifyHTTPError(statusCode int) litellm.ErrorType {
	switch {
	case statusCode == http.StatusUnauthorized, statusCode == http.StatusForbidden:
		return litellm.ErrorTypeAuth
	case statusCode == http.StatusTooManyRequests:
		return litellm.ErrorTypeRateLimit
	case statusCode == http.StatusPaymentRequired:
		return litellm.ErrorTypeQuota
	case statusCode == http.StatusNotFound:
		return litellm.ErrorTypeModel
	case statusCode == http.StatusRequestTimeout:
		return litellm.ErrorTypeTimeout
	case statusCode == http.StatusBadRequest:
		return litellm.ErrorTypeValidation
	case statusCode == 529:
		return litellm.ErrorTypeOverloaded
	case statusCode >= 500:
		return litellm.ErrorTypeProvider
	default:
		return litellm.ErrorTypeProvider
	}
}

// A provider error without an HTTP status has no known recovery semantics.
func isTemporaryHTTPError(status int, kind litellm.ErrorType) bool {
	switch kind {
	case litellm.ErrorTypeContentFilter, litellm.ErrorTypeContextOverflow, litellm.ErrorTypeQuota, litellm.ErrorTypeAuth:
		return false
	}
	switch status {
	case http.StatusRequestTimeout, http.StatusTooManyRequests,
		http.StatusInternalServerError, http.StatusBadGateway,
		http.StatusServiceUnavailable, http.StatusGatewayTimeout, 529:
		return true
	default:
		return false
	}
}

// FinishReason maps a vendor stop reason. An unrecognized value is
// FinishReasonOther; the caller keeps the original as FinishReasonRaw.
func FinishReason(raw string) litellm.FinishReason {
	switch raw {
	case "stop", "end_turn", "STOP", "stop_sequence":
		return litellm.FinishReasonStop
	case "length", "max_tokens", "max_output_tokens", "MAX_TOKENS", "model_context_window_exceeded":
		return litellm.FinishReasonLength
	case "tool_calls", "tool_use":
		return litellm.FinishReasonToolCall
	case "completed":
		return litellm.FinishReasonStop
	case "incomplete":
		return litellm.FinishReasonLength
	case "safety", "SAFETY", "content_filter", "content_filtered", "guardrail_intervened", "refusal", "RECITATION", "sensitive", "BLOCKLIST",
		"PROHIBITED_CONTENT", "SPII", "LANGUAGE", "IMAGE_SAFETY", "IMAGE_PROHIBITED_CONTENT",
		"IMAGE_RECITATION":
		return litellm.FinishReasonSafety
	case "failed", "error", "cancelled", "canceled", "insufficient_system_resource", "network_error",
		"MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL", "TOO_MANY_TOOL_CALLS", "malformed_model_output", "malformed_tool_use", "OTHER",
		"IMAGE_OTHER", "NO_IMAGE", "MISSING_THOUGHT_SIGNATURE":
		return litellm.FinishReasonError
	case "":
		return ""
	default:
		return litellm.FinishReasonOther
	}
}
