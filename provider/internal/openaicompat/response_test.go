package openaicompat_test

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/internal/testgolden"
	"github.com/voocel/litellm/provider/internal/openaicompat"
	"github.com/voocel/litellm/provider/internal/openaicompat/compattest"
)

func TestChatResponse(t *testing.T) {
	details := compattest.Spec(openaicompat.Spec{Name: "test", ReasoningFields: []string{"reasoning_details", "reasoning_content"}})
	tests := []struct {
		name  string
		newFn compattest.NewFunc
		body  string
		want  *litellm.Response
	}{
		{
			name:  "fixture",
			newFn: details,
			body:  testgolden.ReadFixtureString(t, "../../../testdata/compat/chat_response.json"),
			want: &litellm.Response{
				Blocks: []litellm.Block{
					litellm.ReasoningBlock{Text: "think", State: &litellm.ProviderState{Provider: "test", Model: "m", Data: json.RawMessage(`[
          {
            "text": "think"
          }
        ]`)}},
					litellm.TextBlock{Text: "hello"},
					litellm.ToolUseBlock{ID: "call_1", Name: "lookup", Arguments: `{"q":"x"}`},
				},
				Usage:           litellm.Usage{InputTokens: 3, OutputTokens: 4, ReasoningTokens: 2},
				Model:           "provider-model",
				Provider:        "test",
				FinishReason:    litellm.FinishReasonToolCall,
				FinishReasonRaw: "tool_calls",
			},
		},
		{
			name:  "reasoning field priority and model fallback",
			newFn: compattest.Spec(openaicompat.Spec{Name: "test", ReasoningFields: []string{"reasoning", "reasoning_content"}}),
			body:  `{"choices":[{"message":{"reasoning":"","reasoning_content":{"summary":"sum"},"content":"ok"},"finish_reason":"stop"}]}`,
			want: &litellm.Response{
				Blocks:   []litellm.Block{litellm.ReasoningBlock{Text: "sum"}, litellm.TextBlock{Text: "ok"}},
				Model:    "m",
				Provider: "test", FinishReason: litellm.FinishReasonStop, FinishReasonRaw: "stop",
			},
		},
		{
			name:  "content parts keep annotations and logprobs",
			newFn: plain,
			body: `{"choices":[{"message":{"content":[
				{"type":"text","text":"a","annotations":[{"type":"url_citation","url":"https://x.test"}],"logprobs":[{"token":"a"}]},
				{"type":"text","text":""}]},"finish_reason":"stop"}]}`,
			want: &litellm.Response{
				Blocks: []litellm.Block{litellm.TextBlock{
					Text:        "a",
					Annotations: []litellm.Annotation{{Type: "url_citation", URL: "https://x.test", Extra: json.RawMessage(`{"type":"url_citation","url":"https://x.test"}`)}},
					Logprobs:    json.RawMessage(`[{"token":"a"}]`),
				}},
				Model: "m", Provider: "test", FinishReason: litellm.FinishReasonStop, FinishReasonRaw: "stop",
			},
		},
		{
			name:  "refusal is text with a safety finish",
			newFn: plain,
			body:  `{"choices":[{"message":{"content":null,"refusal":"no"},"finish_reason":"stop"}]}`,
			want: &litellm.Response{
				Blocks: []litellm.Block{litellm.TextBlock{Text: "no"}},
				Model:  "m", Provider: "test", FinishReason: litellm.FinishReasonSafety, FinishReasonRaw: "stop",
			},
		},
		{
			name:  "argument-less tool call is an empty object, as streamed",
			newFn: plain,
			body:  `{"choices":[{"message":{"tool_calls":[{"id":"c","type":"function","function":{"name":"f","arguments":""}}]},"finish_reason":"tool_calls"}]}`,
			want: &litellm.Response{
				Blocks: []litellm.Block{litellm.ToolUseBlock{ID: "c", Name: "f", Arguments: `{}`}},
				Model:  "m", Provider: "test", FinishReason: litellm.FinishReasonToolCall, FinishReasonRaw: "tool_calls",
			},
		},
		{
			name:  "cache usage",
			newFn: plain,
			body:  `{"choices":[],"usage":{"prompt_tokens":9,"completion_tokens":1,"total_tokens":10,"prompt_tokens_details":{"cached_tokens":4,"cache_write_tokens":2}}}`,
			want: &litellm.Response{
				Usage: litellm.Usage{InputTokens: 9, OutputTokens: 1, CacheReadTokens: 4, CacheWriteTokens: 2},
				Model: "m", Provider: "test",
			},
		},
		{
			name:  "prompt cache hit tokens fallback",
			newFn: plain,
			body:  `{"choices":[],"usage":{"prompt_tokens":9,"completion_tokens":1,"prompt_cache_hit_tokens":5}}`,
			want: &litellm.Response{
				Usage: litellm.Usage{InputTokens: 9, OutputTokens: 1, CacheReadTokens: 5},
				Model: "m", Provider: "test",
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := compattest.Chat(t, tt.newFn, tt.body)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("got  %#v\nwant %#v", got, tt.want)
			}
		})
	}
}

func TestChatErrors(t *testing.T) {
	if _, err := compattest.Chat(t, plain, `{"choices":[{"message":{"content":[{"type":"audio"}]}}]}`); litellm.ErrorTypeOf(err) != litellm.ErrorTypeProvider || !strings.Contains(err.Error(), `unsupported content part type "audio"`) {
		t.Fatalf("unknown part: %v", err)
	}
	if _, err := compattest.Chat(t, plain, `{`); litellm.ErrorTypeOf(err) != litellm.ErrorTypeProvider {
		t.Fatalf("malformed body: %v", err)
	}
	// Gateways report upstream failures with HTTP 200 and an error body.
	if _, err := compattest.Chat(t, plain, `{"error":{"code":429,"message":"Rate limited"}}`); litellm.ErrorTypeOf(err) != litellm.ErrorTypeRateLimit {
		t.Fatalf("error body: %v", err)
	}
}
