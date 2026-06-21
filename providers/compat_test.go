package providers

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"testing"
)

func TestCompatFindReasoningSupportsSummaryText(t *testing.T) {
	compat := &Compat{}

	reasoning, field := compat.findReasoning(map[string]any{
		"reasoning_summary": map[string]any{"text": "summary"},
	})
	if reasoning != "summary" {
		t.Fatalf("reasoning = %q, want %q", reasoning, "summary")
	}
	if field != "reasoning_summary" {
		t.Fatalf("field = %q, want %q", field, "reasoning_summary")
	}
}

func TestCompatOmitsThinkingWithoutMapper(t *testing.T) {
	var warnings []string
	// Use an inline compat with no ThinkingMapper rather than a real provider —
	// real providers gain mappers over time (e.g. ollama did), which would
	// silently invalidate this framework-level test.
	p := NewOpenAICompat(ProviderConfig{}, Compat{
		ProviderName:   "no-thinking-mapper",
		DefaultBaseURL: "https://api.example.test/v1",
	})
	body, err := p.buildRequestBody(&Request{
		Model:     "some-model",
		Messages:  []Message{{Role: "user", Content: "hi"}},
		Thinking:  &ThinkingConfig{Type: "enabled"},
		OnWarning: func(provider string, message string) { warnings = append(warnings, provider+": "+message) },
	}, false)
	if err != nil {
		t.Fatalf("buildRequestBody failed: %v", err)
	}

	var got map[string]any
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatalf("unmarshal body: %v", err)
	}
	if _, ok := got["thinking"]; ok {
		t.Fatalf("thinking should be omitted without mapper: %+v", got)
	}
	if len(warnings) != 1 || !strings.Contains(warnings[0], "thinking was omitted") {
		t.Fatalf("expected thinking omission warning, got %+v", warnings)
	}
}

func TestCompatDefaultBaseURLIsApplied(t *testing.T) {
	p := NewOpenAICompat(ProviderConfig{}, Compat{
		ProviderName:   "custom-compatible",
		DefaultBaseURL: "https://api.example.test/v1",
	})

	if got := p.Config().BaseURL; got != "https://api.example.test/v1" {
		t.Fatalf("base URL = %q, want %q", got, "https://api.example.test/v1")
	}
}

func TestCompatSetHeadersUsesConfiguredUserAgentAndExtraHeaders(t *testing.T) {
	p := NewOpenAICompat(ProviderConfig{
		APIKey: "test-key",
		Extra: map[string]any{
			"user_agent": "custom-client/1.0",
			"headers": map[string]string{
				"X-Provider": "from-extra",
				"X-Trace":    "trace-id",
			},
		},
	}, Compat{
		ProviderName:   "custom-compatible",
		DefaultBaseURL: "https://api.example.test/v1",
		ExtraHeaders: map[string]string{
			"X-Provider": "from-compat",
		},
		StreamHeaders: map[string]string{
			"X-Stream": "true",
		},
	})
	httpReq, err := http.NewRequestWithContext(context.Background(), "POST", "https://example.test/chat", nil)
	if err != nil {
		t.Fatal(err)
	}

	if err := p.setHeaders(httpReq, nil, true); err != nil {
		t.Fatalf("setHeaders: %v", err)
	}

	if got := httpReq.Header.Get("User-Agent"); got != "custom-client/1.0" {
		t.Fatalf("User-Agent = %q, want custom-client/1.0", got)
	}
	if got := httpReq.Header.Get("X-Provider"); got != "from-extra" {
		t.Fatalf("X-Provider = %q, want from-extra", got)
	}
	if got := httpReq.Header.Get("X-Trace"); got != "trace-id" {
		t.Fatalf("X-Trace = %q, want trace-id", got)
	}
	if got := httpReq.Header.Get("X-Stream"); got != "true" {
		t.Fatalf("X-Stream = %q, want true", got)
	}
	if got := httpReq.Header.Get("Accept"); got != "text/event-stream" {
		t.Fatalf("Accept = %q, want text/event-stream", got)
	}
}

func TestCompatSetHeadersRejectsInvalidExtraHeaders(t *testing.T) {
	p := NewOpenAICompat(ProviderConfig{
		APIKey: "test-key",
		Extra: map[string]any{
			"headers": map[string]any{"X-Test": 42},
		},
	}, Compat{
		ProviderName:   "custom-compatible",
		DefaultBaseURL: "https://api.example.test/v1",
	})
	httpReq, err := http.NewRequestWithContext(context.Background(), "POST", "https://example.test/chat", nil)
	if err != nil {
		t.Fatal(err)
	}

	err = p.setHeaders(httpReq, nil, false)
	if err == nil {
		t.Fatal("setHeaders error = nil, want invalid extra header error")
	}
	if !strings.Contains(err.Error(), "custom-compatible") {
		t.Fatalf("error = %q, want provider name", err)
	}
}

func TestCompatThinkingTypeIsCaseInsensitive(t *testing.T) {
	p := NewQwen(ProviderConfig{})
	body, err := p.buildRequestBody(&Request{
		Model:    "qwen-plus",
		Messages: []Message{{Role: "user", Content: "hi"}},
		Thinking: &ThinkingConfig{
			Type:  "Enabled",
			Level: "high",
		},
	}, false)
	if err != nil {
		t.Fatalf("buildRequestBody failed: %v", err)
	}

	var got map[string]any
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatalf("unmarshal body: %v", err)
	}
	if got["enable_thinking"] != true {
		t.Fatalf("enable_thinking = %#v, want true; body=%+v", got["enable_thinking"], got)
	}
	if got["thinking_budget"] != float64(16384) {
		t.Fatalf("thinking_budget = %#v, want 16384; body=%+v", got["thinking_budget"], got)
	}
}
