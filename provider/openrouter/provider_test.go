package openrouter

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/internal/testgolden"
	"github.com/voocel/litellm/provider/compat"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) Do(req *http.Request) (*http.Response, error) {
	return f(req)
}

func TestHeadersReasoningAndCache(t *testing.T) {
	var referer, title string
	body := captureBody(t, &referer, &title, &litellm.Request{
		Model: "anthropic/claude-sonnet-4",
		Messages: []litellm.Message{litellm.User(litellm.TextBlock{
			Text:  "hi",
			Cache: &litellm.CacheControl{Type: litellm.CacheTypeEphemeral, TTL: litellm.CacheTTL1h},
		})},
		Thinking:        &litellm.Thinking{Mode: litellm.ThinkingEnabled, Effort: "high"},
		ProviderOptions: mustProviderOptions(t, map[string]any{"cache_retention": "1h"}),
	})
	if referer == "" || title != "litellm" {
		t.Fatalf("headers referer=%q title=%q", referer, title)
	}
	reasoning := body["reasoning"].(map[string]any)
	if reasoning["effort"] != "high" {
		t.Fatalf("reasoning = %#v", reasoning)
	}
	cache := body["cache_control"].(map[string]any)
	if cache["ttl"] != "1h" {
		t.Fatalf("cache_control = %#v", cache)
	}
	content := body["messages"].([]any)[0].(map[string]any)["content"].([]any)
	blockCache := content[0].(map[string]any)["cache_control"].(map[string]any)
	if blockCache["ttl"] != "1h" {
		t.Fatalf("block cache = %#v", blockCache)
	}
	testgolden.AssertJSON(t, "../../testdata/compat/openrouter_request.golden.json", body)
}

func TestThinkingRequiresBudgetOrEffort(t *testing.T) {
	body := captureBody(t, nil, nil, &litellm.Request{
		Model:    "anthropic/claude-sonnet-4",
		Messages: []litellm.Message{litellm.UserText("hi")},
		Thinking: &litellm.Thinking{Mode: litellm.ThinkingEnabled},
	})
	reasoning := body["reasoning"].(map[string]any)
	if reasoning["enabled"] != true {
		t.Fatalf("reasoning = %#v", reasoning)
	}
}

func TestThinkingDisabledAndEffortValidation(t *testing.T) {
	body := captureBody(t, nil, nil, &litellm.Request{
		Model:    "anthropic/claude-sonnet-4",
		Messages: []litellm.Message{litellm.UserText("hi")},
		Thinking: &litellm.Thinking{Mode: litellm.ThinkingDisabled},
	})
	reasoning := body["reasoning"].(map[string]any)
	if reasoning["effort"] != "none" {
		t.Fatalf("reasoning = %#v", reasoning)
	}

	p, err := New(compat.Config{APIKey: "key", BaseURL: "https://openrouter.test", HTTPClient: roundTripFunc(nil)})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, err = p.Chat(context.Background(), &litellm.Request{
		Model:    "anthropic/claude-sonnet-4",
		Messages: []litellm.Message{litellm.UserText("hi")},
		Thinking: &litellm.Thinking{Mode: litellm.ThinkingEnabled, Effort: "extreme"},
	})
	if err == nil || !strings.Contains(err.Error(), "unsupported reasoning effort") {
		t.Fatalf("expected effort error, got %v", err)
	}
}

func TestCacheRetentionValidation(t *testing.T) {
	p, err := New(compat.Config{APIKey: "key", BaseURL: "https://openrouter.test", HTTPClient: roundTripFunc(nil)})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, err = p.Chat(context.Background(), &litellm.Request{
		Model:           "anthropic/claude-sonnet-4",
		Messages:        []litellm.Message{litellm.UserText("hi")},
		ProviderOptions: mustProviderOptions(t, map[string]any{"cache_retention": "forever"}),
	})
	if err == nil || !strings.Contains(err.Error(), "unsupported cache_retention") {
		t.Fatalf("expected cache retention error, got %v", err)
	}

	_, err = p.Chat(context.Background(), &litellm.Request{
		Model:           "openai/gpt-4o-mini",
		Messages:        []litellm.Message{litellm.UserText("hi")},
		ProviderOptions: mustProviderOptions(t, map[string]any{"cache_retention": "1h"}),
	})
	if err == nil || !strings.Contains(err.Error(), "only supported for anthropic models") {
		t.Fatalf("expected non-anthropic cache error, got %v", err)
	}
}

func TestSessionIDProviderOption(t *testing.T) {
	body := captureBody(t, nil, nil, &litellm.Request{
		Model:           "anthropic/claude-sonnet-4",
		Messages:        []litellm.Message{litellm.UserText("hi")},
		ProviderOptions: mustProviderOptions(t, map[string]any{ProviderOptionSessionID: "agent-session"}),
	})
	if body["session_id"] != "agent-session" {
		t.Fatalf("body = %#v", body)
	}

	p, err := New(compat.Config{APIKey: "key", BaseURL: "https://openrouter.test", HTTPClient: roundTripFunc(nil)})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, err = p.Chat(context.Background(), &litellm.Request{
		Model:           "anthropic/claude-sonnet-4",
		Messages:        []litellm.Message{litellm.UserText("hi")},
		ProviderOptions: mustProviderOptions(t, map[string]any{ProviderOptionSessionID: strings.Repeat("x", 257)}),
	})
	if err == nil || !strings.Contains(err.Error(), "at most 256") {
		t.Fatalf("expected session_id length error, got %v", err)
	}
}

func TestBlockCacheValidation(t *testing.T) {
	_, _, _, err := mapBlocks([]litellm.Block{
		litellm.TextBlock{
			Text:  "hi",
			Cache: &litellm.CacheControl{Type: litellm.CacheTypeEphemeral, TTL: "24h"},
		},
	})
	if err == nil || !strings.Contains(err.Error(), "unsupported cache ttl") {
		t.Fatalf("expected cache ttl error, got %v", err)
	}

	_, _, _, err = mapBlocks([]litellm.Block{
		litellm.TextBlock{
			Text:  "hi",
			Cache: &litellm.CacheControl{Type: "persistent"},
		},
	})
	if err == nil || !strings.Contains(err.Error(), "unsupported cache type") {
		t.Fatalf("expected cache type error, got %v", err)
	}
}

func TestResponseReasoningBlocksRoundTrip(t *testing.T) {
	var capturedBody map[string]any
	p, err := New(compat.Config{
		APIKey:  "key",
		BaseURL: "https://openrouter.test",
		HTTPClient: roundTripFunc(func(httpReq *http.Request) (*http.Response, error) {
			if err := json.NewDecoder(httpReq.Body).Decode(&capturedBody); err != nil {
				t.Fatalf("decode request body: %v", err)
			}
			return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader(`{"model":"anthropic/claude-sonnet-4","choices":[{"message":{"content":"ok","reasoning":"think"}}]}`)), Header: make(http.Header)}, nil
		}),
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	resp, err := p.Chat(context.Background(), &litellm.Request{
		Model:    "anthropic/claude-sonnet-4",
		Messages: []litellm.Message{litellm.UserText("hi")},
	})
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if resp.Reasoning() != "think" {
		t.Fatalf("reasoning = %q", resp.Reasoning())
	}

	_, err = p.Chat(context.Background(), &litellm.Request{
		Model:    "anthropic/claude-sonnet-4",
		Messages: []litellm.Message{litellm.Assistant(resp.Blocks...)},
	})
	if err != nil {
		t.Fatalf("round-trip Chat: %v", err)
	}
	message := capturedBody["messages"].([]any)[0].(map[string]any)
	if message["reasoning"] != "think" {
		t.Fatalf("message reasoning = %#v", message)
	}
}

func TestReasoningDetailsRoundTrip(t *testing.T) {
	var capturedBody map[string]any
	p, err := New(compat.Config{
		APIKey:  "key",
		BaseURL: "https://openrouter.test",
		HTTPClient: roundTripFunc(func(httpReq *http.Request) (*http.Response, error) {
			if err := json.NewDecoder(httpReq.Body).Decode(&capturedBody); err != nil {
				t.Fatalf("decode request body: %v", err)
			}
			return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader(`{
				"model":"anthropic/claude-sonnet-4",
				"choices":[{
					"message":{
						"content":"ok",
						"reasoning_details":[
							{"type":"reasoning.summary","summary":"sum"},
							{"type":"reasoning.encrypted","data":"cipher","format":"anthropic-claude-v1"}
						]
					}
				}]
			}`)), Header: make(http.Header)}, nil
		}),
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	resp, err := p.Chat(context.Background(), &litellm.Request{
		Model:    "anthropic/claude-sonnet-4",
		Messages: []litellm.Message{litellm.UserText("hi")},
	})
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if resp.Reasoning() != "sum" {
		t.Fatalf("reasoning = %q", resp.Reasoning())
	}
	_, err = p.Chat(context.Background(), &litellm.Request{
		Model:    "anthropic/claude-sonnet-4",
		Messages: []litellm.Message{litellm.Assistant(resp.Blocks...)},
	})
	if err != nil {
		t.Fatalf("round-trip Chat: %v", err)
	}
	message := capturedBody["messages"].([]any)[0].(map[string]any)
	if _, hasPlain := message["reasoning"]; hasPlain {
		t.Fatalf("message should preserve reasoning_details, got %#v", message)
	}
	details := message["reasoning_details"].([]any)
	if len(details) != 2 || details[1].(map[string]any)["data"] != "cipher" {
		t.Fatalf("reasoning_details = %#v", details)
	}
}

func TestRejectsSignedOrRedactedReasoningBlockHistory(t *testing.T) {
	_, _, _, err := mapBlocks([]litellm.Block{
		litellm.ReasoningBlock{Text: "think", Signature: "sig"},
	})
	if err == nil || !strings.Contains(err.Error(), "signed or redacted") {
		t.Fatalf("expected signed reasoning error, got %v", err)
	}

	_, _, _, err = mapBlocks([]litellm.Block{
		litellm.ReasoningBlock{Text: "think", Extra: litellm.MustJSONRaw(map[string]any{"provider": "state"})},
	})
	if err == nil || !strings.Contains(err.Error(), "valid JSON array") {
		t.Fatalf("expected reasoning_details shape error, got %v", err)
	}
}

func TestUsageIncludesCacheWriteTokens(t *testing.T) {
	p, err := New(compat.Config{
		APIKey:  "key",
		BaseURL: "https://openrouter.test",
		HTTPClient: roundTripFunc(func(httpReq *http.Request) (*http.Response, error) {
			return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader(`{
				"choices":[{"message":{"content":"ok"}}],
				"usage":{
					"prompt_tokens":10,
					"completion_tokens":1,
					"total_tokens":11,
					"prompt_tokens_details":{"cached_tokens":6,"cache_write_tokens":4}
				}
			}`)), Header: make(http.Header)}, nil
		}),
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	resp, err := p.Chat(context.Background(), &litellm.Request{
		Model:    "anthropic/claude-sonnet-4",
		Messages: []litellm.Message{litellm.UserText("hi")},
	})
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if *resp.Usage.CacheReadTokens != 6 || *resp.Usage.CacheWriteTokens != 4 {
		t.Fatalf("usage = %+v", resp.Usage)
	}
}

func TestJSONSchemaCleaned(t *testing.T) {
	format, err := litellm.NewResponseFormatJSONSchema("answer", "", map[string]any{
		"type":       "object",
		"properties": map[string]any{"ok": map[string]any{"type": "boolean"}},
	}, litellm.StrictEnabled)
	if err != nil {
		t.Fatalf("NewResponseFormatJSONSchema: %v", err)
	}
	body := captureBody(t, nil, nil, &litellm.Request{
		Model:          "openai/gpt-4o-mini",
		Messages:       []litellm.Message{litellm.UserText("hi")},
		ResponseFormat: format,
		ProviderOptions: mustProviderOptions(t, map[string]any{ProviderOptionRouting: map[string]any{
			"order":              []any{"OpenAI"},
			"require_parameters": false,
		}}),
	})
	schema := body["response_format"].(map[string]any)["json_schema"].(map[string]any)["schema"].(map[string]any)
	if schema["additionalProperties"] != false {
		t.Fatalf("schema should be cleaned: %#v", schema)
	}
	provider := body["provider"].(map[string]any)
	if provider["require_parameters"] != true {
		t.Fatalf("structured output must require compatible upstream parameters: %#v", provider)
	}
	if order := provider["order"].([]any); len(order) != 1 || order[0] != "OpenAI" {
		t.Fatalf("provider routing preferences were not preserved: %#v", provider)
	}
}

func TestNonStrictJSONSchemaPreservesAdditionalProperties(t *testing.T) {
	format, err := litellm.NewResponseFormatJSONSchema("answer", "", map[string]any{
		"type":                 "object",
		"additionalProperties": true,
		"properties":           map[string]any{"ok": map[string]any{"type": "boolean"}},
	}, litellm.StrictDisabled)
	if err != nil {
		t.Fatalf("NewResponseFormatJSONSchema: %v", err)
	}
	body := captureBody(t, nil, nil, &litellm.Request{
		Model:          "openai/gpt-4o-mini",
		Messages:       []litellm.Message{litellm.UserText("hi")},
		ResponseFormat: format,
	})
	schema := body["response_format"].(map[string]any)["json_schema"].(map[string]any)["schema"].(map[string]any)
	if schema["additionalProperties"] != true {
		t.Fatalf("non-strict schema was changed: %#v", schema)
	}
}

func TestStrictToolsAreForwarded(t *testing.T) {
	tool, err := litellm.NewTool("lookup", "Lookup.", map[string]any{"type": "object"})
	if err != nil {
		t.Fatalf("NewTool: %v", err)
	}
	tool.Strict = litellm.StrictEnabled
	req := &litellm.Request{
		Model:    "anthropic/claude-sonnet-4.5",
		Messages: []litellm.Message{litellm.UserText("hi")},
		Tools:    []litellm.Tool{tool},
	}
	body := captureBody(t, nil, nil, req)
	fn := body["tools"].([]any)[0].(map[string]any)["function"].(map[string]any)
	if fn["strict"] != true {
		t.Fatalf("strict = %#v, want true", fn["strict"])
	}
	headers := make(http.Header)
	headers.Set("x-anthropic-beta", "interleaved-thinking-2025-05-14")
	mapHeaders(headers, req)
	if got := headers.Get("x-anthropic-beta"); got != "interleaved-thinking-2025-05-14,"+structuredOutputsBeta {
		t.Fatalf("x-anthropic-beta = %q", got)
	}

	p, err := New(compat.Config{APIKey: "key", BaseURL: "https://openrouter.test", HTTPClient: roundTripFunc(nil)})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if got := p.Capabilities("openai/gpt-4o-mini").Tools.StrictSchema; got != litellm.SupportPartial {
		t.Fatalf("strict tool support = %v, want partial", got)
	}
}

func captureBody(t *testing.T, referer, title *string, req *litellm.Request) map[string]any {
	t.Helper()
	var body map[string]any
	p, err := New(compat.Config{
		APIKey:  "key",
		BaseURL: "https://openrouter.test",
		HTTPClient: roundTripFunc(func(httpReq *http.Request) (*http.Response, error) {
			if referer != nil {
				*referer = httpReq.Header.Get("HTTP-Referer")
			}
			if title != nil {
				*title = httpReq.Header.Get("X-Title")
			}
			if err := json.NewDecoder(httpReq.Body).Decode(&body); err != nil {
				t.Fatalf("decode request body: %v", err)
			}
			return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader(`{"choices":[{"message":{"content":"ok"}}]}`)), Header: make(http.Header)}, nil
		}),
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, err := p.Chat(context.Background(), req); err != nil {
		t.Fatalf("Chat: %v", err)
	}
	return body
}

func mustProviderOptions(t *testing.T, values map[string]any) litellm.ProviderOptions {
	t.Helper()
	o, err := litellm.NewProviderOptions(values)
	if err != nil {
		t.Fatal(err)
	}
	return o
}
