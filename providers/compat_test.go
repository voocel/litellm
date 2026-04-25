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
