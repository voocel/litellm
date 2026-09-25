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
					litellm.ReasoningBlock{Text: "think", Extra: json.RawMessage(`[
          {
            "text": "think"
          }
        ]`)},
					litellm.TextBlock{Text: "hello"},
					litellm.ToolUseBlock{ID: "call_1", Name: "lookup", Arguments: json.RawMessage(`{"q":"x"}`)},
				},
				Usage:           litellm.Usage{InputTokens: new(3), OutputTokens: new(4), TotalTokens: new(7), ReasoningTokens: new(2)},
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
			name:  "cache usage",
			newFn: plain,
			body:  `{"choices":[],"usage":{"prompt_tokens":9,"completion_tokens":1,"total_tokens":10,"prompt_tokens_details":{"cached_tokens":4,"cache_write_tokens":2}}}`,
			want: &litellm.Response{
				Usage: litellm.Usage{InputTokens: new(9), OutputTokens: new(1), TotalTokens: new(10), CacheReadTokens: new(4), CacheWriteTokens: new(2)},
				Model: "m", Provider: "test",
			},
		},
		{
			name:  "prompt cache hit tokens fallback",
			newFn: plain,
			body:  `{"choices":[],"usage":{"prompt_tokens":9,"completion_tokens":1,"prompt_cache_hit_tokens":5}}`,
			want: &litellm.Response{
				Usage: litellm.Usage{InputTokens: new(9), OutputTokens: new(1), CacheReadTokens: new(5)},
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
	if _, err := compattest.Chat(t, plain, `{"choices":[{"message":{"content":[{"type":"audio"}]}}]}`); !litellm.IsProviderError(err) || !strings.Contains(err.Error(), `unsupported content part type "audio"`) {
		t.Fatalf("unknown part: %v", err)
	}
	if _, err := compattest.Chat(t, plain, `{`); !litellm.IsProviderError(err) {
		t.Fatalf("malformed body: %v", err)
	}
}
