package providers

import (
	"encoding/json"
	"testing"
)

func buildOllamaTestBody(t *testing.T, thinking *ThinkingConfig) map[string]any {
	t.Helper()

	p := NewOllama(ProviderConfig{})
	raw, err := p.buildRequestBody(&Request{
		Model:    "qwen3.5",
		Messages: []Message{{Role: "user", Content: "hi"}},
		Thinking: thinking,
	}, false)
	if err != nil {
		t.Fatalf("buildRequestBody failed: %v", err)
	}

	var body map[string]any
	if err := json.Unmarshal(raw, &body); err != nil {
		t.Fatalf("unmarshal body: %v", err)
	}
	return body
}

// Levels clamp to Ollama's low/medium/high: minimal→low, xhigh→high.
func TestOllamaThinkingLevelClampsToSupportedEfforts(t *testing.T) {
	cases := map[string]string{
		"minimal": "low",
		"low":     "low",
		"medium":  "medium",
		"high":    "high",
		"xhigh":   "high",
	}
	for level, want := range cases {
		t.Run(level, func(t *testing.T) {
			body := buildOllamaTestBody(t, &ThinkingConfig{Type: "enabled", Level: level})
			if body["reasoning_effort"] != want {
				t.Fatalf("reasoning_effort = %#v, want %q; body=%+v", body["reasoning_effort"], want, body)
			}
		})
	}
}

// disabled → reasoning_effort:none
func TestOllamaThinkingDisabledMapsToNone(t *testing.T) {
	body := buildOllamaTestBody(t, &ThinkingConfig{Type: "disabled"})
	if body["reasoning_effort"] != "none" {
		t.Fatalf("reasoning_effort = %#v, want none; body=%+v", body["reasoning_effort"], body)
	}
}

func TestOllamaThinkingEnabledNoLevelOmitsEffort(t *testing.T) {
	body := buildOllamaTestBody(t, &ThinkingConfig{Type: "enabled"})
	if _, ok := body["reasoning_effort"]; ok {
		t.Fatalf("reasoning_effort should be omitted when no level; body=%+v", body)
	}
}

func TestOllamaOmitsThinkingWhenUnset(t *testing.T) {
	body := buildOllamaTestBody(t, nil)
	if _, ok := body["reasoning_effort"]; ok {
		t.Fatalf("reasoning_effort should be omitted when unset; body=%+v", body)
	}
}
