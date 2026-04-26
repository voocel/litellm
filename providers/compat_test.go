package providers

import (
	"encoding/json"
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
	p := NewOllama(ProviderConfig{})
	body, err := p.buildRequestBody(&Request{
		Model:     "llama",
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
