package openai

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/voocel/litellm"
)

func TestBuildResponsesRequest(t *testing.T) {
	lookup := litellm.Tool{Name: "lookup", Description: "Lookup.", Parameters: litellm.Schema(`{"type":"object","properties":{"q":{"type":"string"}}}`), Strict: litellm.StrictEnabled}
	schema := &litellm.ResponseFormat{Type: litellm.ResponseFormatJSONSchema, JSONSchema: &litellm.JSONSchema{Name: "answer", Description: "d", Schema: litellm.Schema(`{"type":"object"}`), Strict: litellm.StrictEnabled}}
	cached := &litellm.CacheControl{}
	tests := []struct {
		name   string
		req    litellm.Request
		stream bool
		want   string
	}{{
		name: "core fields",
		req: litellm.Request{
			Model: "gpt-5.1", MaxTokens: new(100), Temperature: new(0.5), TopP: new(0.9),
			Messages: []litellm.Message{
				litellm.System("be brief"),
				litellm.System("be kind"),
				litellm.User(litellm.Text("look"), litellm.ImageBlock{URL: "https://x.test/a.png", Detail: "low"}, litellm.ImageBlock{Data: []byte("png"), MIME: "image/png"}, litellm.ImageBlock{FileURI: "file-1"}),
				litellm.Assistant(
					// Reasoning from other providers has no Responses item.
					litellm.ReasoningBlock{Text: "claude", Signature: "sig"},
					litellm.ReasoningBlock{Text: "router", Extra: json.RawMessage(`[{"type":"reasoning.text","text":"router"}]`)},
					litellm.ReasoningBlock{Text: "summary", Summary: true, Extra: json.RawMessage(`{"type":"reasoning","id":"rs_1","summary":[],"encrypted_content":"enc"}`)},
					litellm.Text("calling"),
					litellm.ToolUseBlock{ID: "call_1", Name: "lookup", Arguments: json.RawMessage(`{"q":"x"}`)},
				),
				litellm.ToolResultText("call_1", "result"),
			},
			Tools:          []litellm.Tool{lookup, {Name: "ping"}},
			ToolChoice:     &litellm.ToolChoice{Name: "lookup"},
			ResponseFormat: schema,
			Thinking:       &litellm.Thinking{Effort: "high", IncludeOutput: true},
		},
		want: `{"model":"gpt-5.1","max_output_tokens":100,"temperature":0.5,"top_p":0.9,
			"instructions":"be brief\nbe kind",
			"input":[
				{"type":"message","role":"user","content":[
					{"type":"input_text","text":"look"},
					{"type":"input_image","image_url":"https://x.test/a.png","detail":"low"},
					{"type":"input_image","image_url":"data:image/png;base64,cG5n"},
					{"type":"input_image","file_id":"file-1"}]},
				{"type":"reasoning","id":"rs_1","summary":[],"encrypted_content":"enc"},
				{"type":"message","role":"assistant","content":[{"type":"output_text","text":"calling"}]},
				{"type":"function_call","call_id":"call_1","name":"lookup","arguments":"{\"q\":\"x\"}"},
				{"type":"function_call_output","call_id":"call_1","output":"result"}],
			"tools":[
				{"type":"function","name":"lookup","description":"Lookup.","parameters":{"type":"object","properties":{"q":{"type":"string"}}},"strict":true},
				{"type":"function","name":"ping","parameters":{"type":"object"}}],
			"tool_choice":{"type":"function","name":"lookup"},
			"text":{"format":{"type":"json_schema","name":"answer","description":"d","schema":{"type":"object"},"strict":true}},
			"reasoning":{"effort":"high","summary":"auto"}}`,
	}, {
		// Instructions cannot carry a breakpoint, and only input content has one.
		name: "cache breakpoints",
		req: litellm.Request{Model: "m", Messages: []litellm.Message{
			{Role: litellm.RoleSystem, Blocks: []litellm.Block{litellm.TextBlock{Text: "sys", Cache: cached}}},
			litellm.User(litellm.ImageBlock{URL: "https://x.test/a.png", Cache: cached}),
			litellm.Assistant(litellm.TextBlock{Text: "a", Cache: cached}),
		}},
		want: `{"model":"m","input":[
			{"type":"message","role":"developer","content":[{"type":"input_text","text":"sys","prompt_cache_breakpoint":{"mode":"explicit"}}]},
			{"type":"message","role":"user","content":[{"type":"input_image","image_url":"https://x.test/a.png","prompt_cache_breakpoint":{"mode":"explicit"}}]},
			{"type":"message","role":"assistant","content":[{"type":"output_text","text":"a"}]}]}`,
	}, {
		name:   "disabled thinking and tool choice mode",
		req:    litellm.Request{Model: "m", Thinking: &litellm.Thinking{Mode: litellm.ThinkingDisabled}, ToolChoice: &litellm.ToolChoice{Mode: litellm.ToolChoiceRequired}},
		stream: true,
		want:   `{"model":"m","stream":true,"tool_choice":"required","reasoning":{"effort":"none"}}`,
	}, {
		// Options merge into generated objects and append to generated arrays.
		name: "provider options",
		req: litellm.Request{
			Model: "m", Tools: []litellm.Tool{{Name: "ping"}}, ResponseFormat: schema, Thinking: &litellm.Thinking{Effort: "low"},
			ProviderOptions: providerOptions(t, map[string]any{
				ProviderOptionText:      map[string]any{"verbosity": "low"},
				ProviderOptionReasoning: map[string]any{"summary": "detailed"},
				ProviderOptionTools:     []any{map[string]any{"type": "web_search"}},
				ProviderOptionInclude:   []any{"reasoning.encrypted_content"},
				ProviderOptionStore:     false,
			}),
		},
		want: `{"model":"m","store":false,"include":["reasoning.encrypted_content"],
			"tools":[{"type":"function","name":"ping","parameters":{"type":"object"}},{"type":"web_search"}],
			"text":{"format":{"type":"json_schema","name":"answer","description":"d","schema":{"type":"object"},"strict":true},"verbosity":"low"},
			"reasoning":{"effort":"low","summary":"detailed"}}`,
	}}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			body, err := buildResponsesRequest(&test.req, test.stream)
			if err != nil {
				t.Fatal(err)
			}
			assertJSON(t, body, test.want)
		})
	}
}

func TestBuildResponsesRequestErrors(t *testing.T) {
	tests := []struct {
		name, wantErr string
		req           litellm.Request
	}{
		{"stop", "stop is not supported", litellm.Request{Stop: []string{"x"}}},
		{"budget", "budget_tokens is not supported", litellm.Request{Thinking: &litellm.Thinking{BudgetTokens: new(1024)}}},
		{"cache TTL", "prompt_cache_options", litellm.Request{Messages: []litellm.Message{litellm.User(litellm.TextBlock{Text: "x", Cache: &litellm.CacheControl{TTL: litellm.CacheTTL5m}})}}},
		{"image tool result", "only supports text content", litellm.Request{Messages: []litellm.Message{litellm.ToolResult("call_1", litellm.ImageURL("https://x.test/a.png"))}}},
		{"nested option conflict", `provider option "reasoning.summary" conflicts`, litellm.Request{Thinking: &litellm.Thinking{IncludeOutput: true}, ProviderOptions: providerOptions(t, map[string]any{ProviderOptionReasoning: map[string]any{"summary": "detailed"}})}},
		{"option conflict", `provider option "tools" conflicts`, litellm.Request{Tools: []litellm.Tool{{Name: "ping"}}, ProviderOptions: providerOptions(t, map[string]any{ProviderOptionTools: map[string]any{"type": "web_search"}})}},
	}
	for _, test := range tests {
		if _, err := buildResponsesRequest(&test.req, false); err == nil || !strings.Contains(err.Error(), test.wantErr) {
			t.Errorf("%s: err = %v, want %q", test.name, err, test.wantErr)
		}
	}
}

func TestConvertResponsesResponse(t *testing.T) {
	const (
		reasoning = `{"id":"rs_1","type":"reasoning","summary":[{"type":"summary_text","text":"a"},{"type":"summary_text","text":"b"}],"encrypted_content":"enc"}`
		rawText   = `{"id":"rs_2","type":"reasoning","summary":[],"content":[{"type":"reasoning_text","text":"raw"}]}`
		citation  = `{"type":"url_citation","url":"https://example.com","title":"t"}`
	)
	tests := []struct {
		name, body string
		want       litellm.Response
	}{{
		name: "blocks in output order",
		body: `{"model":"gpt-5.1","status":"completed","output":[` + reasoning + `,
			{"type":"web_search_call","id":"ws_1","status":"completed"},
			{"type":"message","content":[{"type":"output_text","text":"hi","annotations":[` + citation + `],"logprobs":[{"token":"hi"}]}]},
			{"type":"function_call","call_id":"call_1","name":"lookup","arguments":"{\"q\":\"x\"}"}],
			"usage":{"input_tokens":5,"output_tokens":3,"total_tokens":8,"input_tokens_details":{"cached_tokens":2},"output_tokens_details":{"reasoning_tokens":1}}}`,
		want: litellm.Response{
			Blocks: []litellm.Block{
				litellm.ReasoningBlock{Text: "a\nb", Summary: true, Extra: json.RawMessage(reasoning)},
				litellm.TextBlock{Text: "hi", Annotations: []litellm.Annotation{{Type: "url_citation", URL: "https://example.com", Extra: json.RawMessage(citation)}}, Logprobs: json.RawMessage(`[{"token":"hi"}]`)},
				litellm.ToolUseBlock{ID: "call_1", Name: "lookup", Arguments: json.RawMessage(`{"q":"x"}`)},
			},
			Usage:        litellm.Usage{InputTokens: new(5), OutputTokens: new(3), TotalTokens: new(8), CacheReadTokens: new(2), ReasoningTokens: new(1)},
			Model:        "gpt-5.1",
			FinishReason: litellm.FinishReasonToolCall, FinishReasonRaw: "completed",
		},
	}, {
		name: "raw reasoning text and refusal",
		body: `{"status":"completed","output":[` + rawText + `,{"type":"message","content":[{"type":"refusal","refusal":"no"}]}]}`,
		want: litellm.Response{
			Blocks:       []litellm.Block{litellm.ReasoningBlock{Text: "raw", Extra: json.RawMessage(rawText)}, litellm.TextBlock{Text: "no"}},
			Model:        "req-model",
			FinishReason: litellm.FinishReasonSafety, FinishReasonRaw: "completed",
		},
	}, {
		name: "argument-less call is an empty object, as streamed",
		body: `{"status":"completed","output":[{"type":"function_call","call_id":"c","name":"f","arguments":""}]}`,
		want: litellm.Response{
			Blocks:       []litellm.Block{litellm.ToolUseBlock{ID: "c", Name: "f", Arguments: json.RawMessage(`{}`)}},
			Model:        "req-model",
			FinishReason: litellm.FinishReasonToolCall, FinishReasonRaw: "completed",
		},
	}, {
		name: "incomplete",
		body: `{"status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"output":[]}`,
		want: litellm.Response{Model: "req-model", FinishReason: litellm.FinishReasonLength, FinishReasonRaw: "max_output_tokens"},
	}, {
		name: "content filter",
		body: `{"status":"incomplete","incomplete_details":{"reason":"content_filter"},"output":[]}`,
		want: litellm.Response{Model: "req-model", FinishReason: litellm.FinishReasonSafety, FinishReasonRaw: "content_filter"},
	}}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var parsed responsesResponse
			if err := json.Unmarshal([]byte(test.body), &parsed); err != nil {
				t.Fatal(err)
			}
			test.want.Provider = "openai"
			if got := convertResponsesResponse(&parsed, "req-model"); !reflect.DeepEqual(*got, test.want) {
				t.Fatalf("response = %#v\nwant %#v", *got, test.want)
			}
		})
	}
}

func TestResponsesChat(t *testing.T) {
	const reply = `{"model":"gpt-5.1","status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"ok"}]}]}`
	p, got := testProvider(t, Config{API: APIResponses}, reply)
	client, err := litellm.New(p, litellm.WithCaptureRawResponse(true))
	if err != nil {
		t.Fatal(err)
	}
	resp, err := client.Chat(context.Background(), litellm.Request{Model: "gpt-5.1", Messages: []litellm.Message{litellm.UserText("hi")}})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text() != "ok" || string(resp.Raw) != reply || got.req.Header.Get("Accept") != "application/json" {
		t.Fatalf("response = %+v, Accept = %q", resp, got.req.Header.Get("Accept"))
	}
	assertJSON(t, got.body, `{"model":"gpt-5.1","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}]}`)

	p, _ = testProvider(t, Config{API: APIResponses}, `{"status":"failed","error":{"code":"server_error","message":"boom"}}`)
	_, err = p.Chat(context.Background(), &litellm.Request{Model: "m"})
	var e *litellm.Error
	if !errors.As(err, &e) || e.Code != "server_error" || e.Message != "boom" || e.Provider != "openai" {
		t.Fatalf("failed response: err = %#v", err)
	}
}
