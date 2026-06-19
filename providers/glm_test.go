package providers

import (
	"encoding/json"
	"testing"
)

func buildGLMTestBody(t *testing.T, model string, thinking *ThinkingConfig) map[string]any {
	t.Helper()

	p := NewGLM(ProviderConfig{})
	raw, err := p.buildRequestBody(&Request{
		Model:    model,
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

func requireGLMThinking(t *testing.T, body map[string]any) map[string]any {
	t.Helper()
	thinking, ok := body["thinking"].(map[string]any)
	if !ok {
		t.Fatalf("thinking = %#v, want object; body=%+v", body["thinking"], body)
	}
	return thinking
}

// GLM-5.2+ forwards the level verbatim as reasoning_effort.
func TestGLMThinkingLevelMapsToReasoningEffortOnNewModels(t *testing.T) {
	for _, level := range []string{"minimal", "low", "medium", "high", "xhigh"} {
		t.Run(level, func(t *testing.T) {
			body := buildGLMTestBody(t, "glm-5.2", &ThinkingConfig{Type: "enabled", Level: level})
			if got := requireGLMThinking(t, body)["type"]; got != "enabled" {
				t.Fatalf("thinking.type = %#v, want enabled", got)
			}
			if body["reasoning_effort"] != level {
				t.Fatalf("reasoning_effort = %#v, want %q; body=%+v", body["reasoning_effort"], level, body)
			}
		})
	}
}

func TestGLMReasoningEffortGatedBelow52(t *testing.T) {
	for _, model := range []string{"glm-4.5", "glm-4.6", "glm-4.7", "glm-5", "glm-5.1"} {
		t.Run(model, func(t *testing.T) {
			body := buildGLMTestBody(t, model, &ThinkingConfig{Type: "enabled", Level: "high"})
			if got := requireGLMThinking(t, body)["type"]; got != "enabled" {
				t.Fatalf("thinking.type = %#v, want enabled", got)
			}
			if _, ok := body["reasoning_effort"]; ok {
				t.Fatalf("%s must not carry reasoning_effort; body=%+v", model, body)
			}
		})
	}
}

// disabled explicitly turns thinking off; no reasoning_effort.
func TestGLMThinkingDisabled(t *testing.T) {
	body := buildGLMTestBody(t, "glm-5.2", &ThinkingConfig{Type: "disabled", Level: "high"})
	if got := requireGLMThinking(t, body)["type"]; got != "disabled" {
		t.Fatalf("thinking.type = %#v, want disabled; body=%+v", got, body)
	}
	if _, ok := body["reasoning_effort"]; ok {
		t.Fatalf("disabled must not carry reasoning_effort; body=%+v", body)
	}
}

func TestGLMOmitsThinkingWhenUnset(t *testing.T) {
	body := buildGLMTestBody(t, "glm-5.2", nil)
	if _, ok := body["thinking"]; ok {
		t.Fatalf("thinking should be omitted when unset; body=%+v", body)
	}
	if _, ok := body["reasoning_effort"]; ok {
		t.Fatalf("reasoning_effort should be omitted when unset; body=%+v", body)
	}
}

func TestGLMVersionAtLeast(t *testing.T) {
	// reasoning_effort gate (5.2+)
	effort := map[string]bool{
		"glm-5.2": true, "glm-5.2-flash": true, "GLM-5.2": true, "glm-6": true,
		"glm-5": false, "glm-5.1": false, "glm-4.6": false, "glm-4.5v": false,
		"glm-z1-air": false, "custom-model": false,
	}
	for model, want := range effort {
		if got := glmVersionAtLeast(model, 5, 2); got != want {
			t.Errorf("glmVersionAtLeast(%q, 5, 2) = %v, want %v", model, got, want)
		}
	}
	// thinking switch gate (4.5+)
	thinking := map[string]bool{
		"glm-4.5": true, "glm-4.6": true, "glm-4.7": true, "glm-5.2": true,
		"glm-4": false, "glm-4.4": false, "glm-z1-air": false, "custom-model": false,
	}
	for model, want := range thinking {
		if got := glmVersionAtLeast(model, 4, 5); got != want {
			t.Errorf("glmVersionAtLeast(%q, 4, 5) = %v, want %v", model, got, want)
		}
	}
}

func TestGLMOmitsThinkingBelow45(t *testing.T) {
	for _, model := range []string{"glm-4", "glm-4.4"} {
		t.Run(model, func(t *testing.T) {
			body := buildGLMTestBody(t, model, &ThinkingConfig{Type: "enabled", Level: "high"})
			if _, ok := body["thinking"]; ok {
				t.Fatalf("%s must not carry thinking; body=%+v", model, body)
			}
			if _, ok := body["reasoning_effort"]; ok {
				t.Fatalf("%s must not carry reasoning_effort; body=%+v", model, body)
			}
		})
	}
}
