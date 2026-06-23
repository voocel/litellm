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
		Thinking:        &litellm.Thinking{Mode: litellm.ThinkingEnabled, Level: "high"},
		ProviderOptions: litellm.ProviderOptions{"cache_retention": "1h"},
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

func TestThinkingRequiresBudgetEffortOrLevel(t *testing.T) {
	p, err := New(compat.Config{APIKey: "key", BaseURL: "https://openrouter.test", HTTPClient: roundTripFunc(nil)})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, err = p.Chat(context.Background(), &litellm.Request{
		Model:    "anthropic/claude-sonnet-4",
		Messages: []litellm.Message{litellm.UserText("hi")},
		Thinking: &litellm.Thinking{Mode: litellm.ThinkingEnabled},
	})
	if err == nil || !strings.Contains(err.Error(), "budget_tokens, effort, or level is required") {
		t.Fatalf("expected thinking requirement error, got %v", err)
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
		ProviderOptions: litellm.ProviderOptions{"cache_retention": "forever"},
	})
	if err == nil || !strings.Contains(err.Error(), "unsupported cache_retention") {
		t.Fatalf("expected cache retention error, got %v", err)
	}

	_, err = p.Chat(context.Background(), &litellm.Request{
		Model:           "openai/gpt-4o-mini",
		Messages:        []litellm.Message{litellm.UserText("hi")},
		ProviderOptions: litellm.ProviderOptions{"cache_retention": "1h"},
	})
	if err == nil || !strings.Contains(err.Error(), "only supported for anthropic models") {
		t.Fatalf("expected non-anthropic cache error, got %v", err)
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

func TestRejectsOpaqueReasoningBlockHistory(t *testing.T) {
	_, _, _, err := mapBlocks([]litellm.Block{
		litellm.ReasoningBlock{Text: "think", Extra: litellm.MustJSONRaw(map[string]any{"provider": "state"})},
	})
	if err == nil || !strings.Contains(err.Error(), "provider-extra reasoning") {
		t.Fatalf("expected opaque reasoning error, got %v", err)
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
	})
	schema := body["response_format"].(map[string]any)["json_schema"].(map[string]any)["schema"].(map[string]any)
	if schema["additionalProperties"] != false {
		t.Fatalf("schema should be cleaned: %#v", schema)
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
