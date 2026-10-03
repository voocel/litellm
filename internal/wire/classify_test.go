package wire

import (
	"net/http"
	"testing"

	"github.com/voocel/litellm"
)

func TestFinishReason(t *testing.T) {
	tests := []struct {
		raw  string
		want litellm.FinishReason
	}{
		{raw: "stop", want: litellm.FinishReasonStop},
		{raw: "end_turn", want: litellm.FinishReasonStop},
		{raw: "pause_turn", want: litellm.FinishReasonOther},
		{raw: "STOP", want: litellm.FinishReasonStop},
		{raw: "length", want: litellm.FinishReasonLength},
		{raw: "max_tokens", want: litellm.FinishReasonLength},
		{raw: "max_output_tokens", want: litellm.FinishReasonLength},
		{raw: "model_context_window_exceeded", want: litellm.FinishReasonLength},
		{raw: "tool_use", want: litellm.FinishReasonToolCall},
		{raw: "content_filter", want: litellm.FinishReasonSafety},
		{raw: "content_filtered", want: litellm.FinishReasonSafety},
		{raw: "guardrail_intervened", want: litellm.FinishReasonSafety},
		{raw: "refusal", want: litellm.FinishReasonSafety},
		{raw: "RECITATION", want: litellm.FinishReasonSafety},
		{raw: "PROHIBITED_CONTENT", want: litellm.FinishReasonSafety},
		{raw: "IMAGE_SAFETY", want: litellm.FinishReasonSafety},
		{raw: "MALFORMED_FUNCTION_CALL", want: litellm.FinishReasonError},
		{raw: "malformed_model_output", want: litellm.FinishReasonError},
		{raw: "malformed_tool_use", want: litellm.FinishReasonError},
		{raw: "UNEXPECTED_TOOL_CALL", want: litellm.FinishReasonError},
		{raw: "MISSING_THOUGHT_SIGNATURE", want: litellm.FinishReasonError},
		{raw: "insufficient_system_resource", want: litellm.FinishReasonError},
		{raw: "", want: ""},
		{raw: "provider_specific", want: litellm.FinishReasonOther},
	}

	for _, tt := range tests {
		if got := FinishReason(tt.raw); got != tt.want {
			t.Fatalf("FinishReason(%q) = %q, want %q", tt.raw, got, tt.want)
		}
	}
}

func TestHTTPErrorTemporary(t *testing.T) {
	for _, tc := range []struct {
		status    int
		body      string
		temporary bool
	}{
		{429, "busy", true},
		{503, "busy", true},
		{408, "timeout", true},
		{529, "busy", true},
		{501, "unsupported", false},
		{400, "bad input", false},
		{503, `{"error":{"code":"content_filter","message":"blocked"}}`, false},
	} {
		if err := HTTPError("test", tc.status, nil, tc.body); err.Temporary != tc.temporary {
			t.Errorf("%d %s: temporary = %v, want %v", tc.status, tc.body, err.Temporary, tc.temporary)
		}
	}
	if got := HTTPError("p", 400, nil, `{"error":{"code":"bad","message":"no"}}`).Error(); got != "p: bad: no (HTTP 400)" {
		t.Errorf("Error() = %q", got)
	}
}

func TestHTTPErrorStatus(t *testing.T) {
	for _, tc := range []struct {
		status    int
		want      litellm.ErrorType
		temporary bool
	}{
		{401, litellm.ErrorTypeAuth, false},
		{402, litellm.ErrorTypeQuota, false},
		{404, litellm.ErrorTypeModel, false},
		{408, litellm.ErrorTypeProvider, true},
		{413, litellm.ErrorTypeValidation, false},
		{422, litellm.ErrorTypeValidation, false},
		{503, litellm.ErrorTypeOverloaded, true},
		{504, litellm.ErrorTypeProvider, true},
		{529, litellm.ErrorTypeOverloaded, true},
	} {
		if err := HTTPError("test", tc.status, nil, ""); err.Type != tc.want || err.Temporary != tc.temporary {
			t.Errorf("%d: type = %q, temporary = %v; want %q, %v", tc.status, err.Type, err.Temporary, tc.want, tc.temporary)
		}
	}
}

// The code names the failure; the status still decides its type, since
// OpenAI sends invalid_request_error with 401s and 404s too.
func TestHTTPErrorCode(t *testing.T) {
	aws := func(name string) http.Header { return http.Header{"X-Amzn-Errortype": {name}} }
	for _, tc := range []struct {
		name   string
		status int
		header http.Header
		body   string
		want   string
		typ    litellm.ErrorType
	}{
		{"anthropic", 529, nil, `{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"},"request_id":"req_1"}`,
			"test: overloaded_error: Overloaded (HTTP 529)", litellm.ErrorTypeOverloaded},
		{"openai without a key", 401, nil, `{"error":{"message":"You didn't provide an API key.","type":"invalid_request_error","param":null,"code":null}}`,
			"test: invalid_request_error: You didn't provide an API key. (HTTP 401)", litellm.ErrorTypeAuth},
		{"bedrock", 403, aws("AccessDeniedException:http://internal.amazon.com/coral/com.amazon.bedrock/"), `{"Message":"Not authorized"}`,
			"test: AccessDeniedException: Not authorized (HTTP 403)", litellm.ErrorTypeAuth},
		{"bedrock quota", 400, aws("aws.bedrock#ServiceQuotaExceededException"), `{"message":"Too many tokens per day"}`,
			"test: ServiceQuotaExceededException: Too many tokens per day (HTTP 400)", litellm.ErrorTypeQuota},
		{"top-level code", 401, nil, `{"code":"InvalidApiKey","message":"Invalid API-key provided."}`,
			"test: InvalidApiKey: Invalid API-key provided. (HTTP 401)", litellm.ErrorTypeAuth},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := HTTPError("test", tc.status, tc.header, tc.body)
			if err.Error() != tc.want || err.Type != tc.typ {
				t.Fatalf("got %q (%s), want %q (%s)", err.Error(), err.Type, tc.want, tc.typ)
			}
		})
	}
}

func TestHTTPErrorClassifiesVendorRejections(t *testing.T) {
	for _, tc := range []struct {
		name string
		body string
		want litellm.ErrorType
	}{
		{"openai", `{"error":{"message":"This model's maximum context length is 128000 tokens. However, your messages resulted in 130000 tokens.","type":"invalid_request_error","code":"context_length_exceeded"}}`, litellm.ErrorTypeContextOverflow},
		{"anthropic", `{"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long: 210000 tokens > 200000 maximum"}}`, litellm.ErrorTypeContextOverflow},
		{"gemini", `{"error":{"code":400,"message":"The input token count (1200000) exceeds the maximum number of tokens allowed (1048576).","status":"INVALID_ARGUMENT"}}`, litellm.ErrorTypeContextOverflow},
		{"bedrock", `{"message":"Input is too long for requested model."}`, litellm.ErrorTypeContextOverflow},
		{"openrouter", `{"error":{"code":400,"message":"Provider returned error","metadata":{"error_type":"context_length_exceeded"}}}`, litellm.ErrorTypeContextOverflow},
		{"openrouter responses", `{"error":{"code":"invalid_prompt","message":"This endpoint's maximum context length is 8192 tokens."}}`, litellm.ErrorTypeContextOverflow},
		{"glm", `{"error":{"code":"1261","message":"Prompt 超长"}}`, litellm.ErrorTypeContextOverflow},
		{"xai", `{"code":"Client specified an invalid argument","error":"This model's maximum prompt length is 131072 but the request contains 140000 tokens."}`, litellm.ErrorTypeContextOverflow},
		{"content filter", `{"error":{"code":"content_filter","message":"blocked"}}`, litellm.ErrorTypeContentFilter},
		{"gemini invalid key", `{"error":{"code":400,"message":"API key not valid. Please pass a valid API key.","status":"INVALID_ARGUMENT"}}`, litellm.ErrorTypeAuth},
		{"glm code substring", `{"error":{"code":"1214","message":"request 1261 invalid"}}`, litellm.ErrorTypeValidation},
		{"max tokens", `{"error":{"type":"invalid_request_error","message":"max_tokens is too large"}}`, litellm.ErrorTypeValidation},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := HTTPError("test", 400, nil, tc.body)
			if err.Type != tc.want || err.Temporary {
				t.Fatalf("type = %q, temporary = %v; want %q", err.Type, err.Temporary, tc.want)
			}
		})
	}
	if err := HTTPError("test", 503, nil, `{"error":{"message":"prompt is too long"}}`); err.Temporary {
		t.Fatal("context overflow must not be temporary behind a rewritten 5xx")
	}
	// Billing failures arrive as 429 but retrying does not clear them.
	for _, body := range []string{
		`{"error":{"message":"You exceeded your current quota","type":"insufficient_quota","param":null,"code":"insufficient_quota"}}`,
		`{"error":{"message":"Your credit balance is exhausted","type":"insufficient_quota","code":"credit_balance_exhausted"}}`,
	} {
		if err := HTTPError("test", 429, nil, body); err.Type != litellm.ErrorTypeQuota || err.Temporary {
			t.Fatalf("%s: type = %q, temporary = %v", body, err.Type, err.Temporary)
		}
	}
}

func TestStreamErrorClassifiesVendorCodes(t *testing.T) {
	for _, tc := range []struct {
		code, message string
		want          litellm.ErrorType
		temporary     bool
	}{
		{"overloaded_error", "Overloaded", litellm.ErrorTypeOverloaded, true},
		{"throttlingException", "Too many requests", litellm.ErrorTypeRateLimit, true},
		{"validationException", "Input is too long for requested model", litellm.ErrorTypeContextOverflow, false},
		{"server_is_overloaded", "Our servers are currently overloaded.", litellm.ErrorTypeOverloaded, true},
		{"context_length_exceeded", "Your input exceeds the context window of this model.", litellm.ErrorTypeContextOverflow, false},
		{"unknown", "boom", litellm.ErrorTypeProvider, false},
		// Server faults are as retryable in a stream as their HTTP 5xx.
		{"api_error", "Internal server error", litellm.ErrorTypeProvider, true},
		{"server_error", "The server had an error", litellm.ErrorTypeProvider, true},
		{"internalServerException", "Internal failure", litellm.ErrorTypeProvider, true},
		{"modelStreamErrorException", "Model stream failed", litellm.ErrorTypeProvider, true},
		{"timeout_error", "Request timeout", litellm.ErrorTypeProvider, true},
		// Anthropic types stand in for the status they are sent with.
		{"authentication_error", "invalid x-api-key", litellm.ErrorTypeAuth, false},
		{"permission_error", "not allowed", litellm.ErrorTypeAuth, false},
		{"billing_error", "credit balance too low", litellm.ErrorTypeQuota, false},
		{"not_found_error", "model: nope", litellm.ErrorTypeModel, false},
		{"request_too_large", "Request exceeds the maximum size", litellm.ErrorTypeValidation, false},
	} {
		err := StreamError("test", tc.code, tc.message)
		if err.Type != tc.want || err.Temporary != tc.temporary || err.Code != tc.code {
			t.Errorf("%s: type = %q, temporary = %v", tc.code, err.Type, err.Temporary)
		}
	}
}
