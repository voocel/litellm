package wire

import (
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
	if got := HTTPError("p", 400, nil, `{"error":{"code":"bad","message":"no"}}`).Error(); got != "p: bad: no" {
		t.Errorf("Error() = %q", got)
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
	} {
		err := StreamError("test", tc.code, tc.message)
		if err.Type != tc.want || err.Temporary != tc.temporary || err.Code != tc.code {
			t.Errorf("%s: type = %q, temporary = %v", tc.code, err.Type, err.Temporary)
		}
	}
}
