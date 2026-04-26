package providers

import (
	"context"
	"encoding/json"
	"testing"
)

func TestDeepSeekThinkingDisabledMapsToOfficialThinking(t *testing.T) {
	body := buildDeepSeekTestBody(t, &ThinkingConfig{Type: "disabled"})
	thinking := requireDeepSeekThinking(t, body)

	if thinking["type"] != "disabled" {
		t.Fatalf("thinking.type = %#v, want disabled; body=%+v", thinking["type"], body)
	}
	if _, ok := body["reasoning_effort"]; ok {
		t.Fatalf("reasoning_effort should be omitted when disabled; body=%+v", body)
	}
}

func TestDeepSeekThinkingHighMapsToOfficialReasoningEffort(t *testing.T) {
	body := buildDeepSeekTestBody(t, &ThinkingConfig{Type: "enabled", Level: "high"})
	thinking := requireDeepSeekThinking(t, body)

	if thinking["type"] != "enabled" {
		t.Fatalf("thinking.type = %#v, want enabled; body=%+v", thinking["type"], body)
	}
	if body["reasoning_effort"] != "high" {
		t.Fatalf("reasoning_effort = %#v, want high; body=%+v", body["reasoning_effort"], body)
	}
}

func TestDeepSeekThinkingXHighMapsToMaxEffort(t *testing.T) {
	body := buildDeepSeekTestBody(t, &ThinkingConfig{Type: "enabled", Level: "xhigh"})

	if body["reasoning_effort"] != "max" {
		t.Fatalf("reasoning_effort = %#v, want max; body=%+v", body["reasoning_effort"], body)
	}
}

func TestDeepSeekThinkingLowAndMediumMapToHighEffort(t *testing.T) {
	for _, level := range []string{"low", "medium"} {
		t.Run(level, func(t *testing.T) {
			body := buildDeepSeekTestBody(t, &ThinkingConfig{Type: "enabled", Level: level})

			if body["reasoning_effort"] != "high" {
				t.Fatalf("reasoning_effort = %#v, want high; body=%+v", body["reasoning_effort"], body)
			}
		})
	}
}

func TestDeepSeekOmitsThinkingWhenUnset(t *testing.T) {
	body := buildDeepSeekTestBody(t, nil)
	if _, ok := body["thinking"]; ok {
		t.Fatalf("thinking should be omitted when unset; body=%+v", body)
	}
	if _, ok := body["reasoning_effort"]; ok {
		t.Fatalf("reasoning_effort should be omitted when thinking is unset; body=%+v", body)
	}
}

func TestDeepSeekMessagesPreserveReasoningContentForToolTurns(t *testing.T) {
	p := NewDeepSeek(ProviderConfig{})
	raw, err := p.buildRequestBody(&Request{
		Model: "deepseek-v4-pro",
		Messages: []Message{
			{Role: "user", Content: "hi"},
			{
				Role:             "assistant",
				Content:          "",
				ReasoningContent: "need tool",
				ToolCalls: []ToolCall{{
					ID:   "call_1",
					Type: "function",
					Function: FunctionCall{
						Name:      "lookup",
						Arguments: "{}",
					},
				}},
			},
			{Role: "tool", ToolCallID: "call_1", Content: "ok"},
		},
		Thinking: &ThinkingConfig{Type: "enabled", Level: "high"},
	}, false)
	if err != nil {
		t.Fatalf("buildRequestBody failed: %v", err)
	}

	var body map[string]any
	if err := json.Unmarshal(raw, &body); err != nil {
		t.Fatalf("unmarshal body: %v", err)
	}
	messages, ok := body["messages"].([]any)
	if !ok || len(messages) != 3 {
		t.Fatalf("messages = %#v, want 3-item array; body=%+v", body["messages"], body)
	}
	assistant, ok := messages[1].(map[string]any)
	if !ok {
		t.Fatalf("assistant message = %#v, want object", messages[1])
	}
	if assistant["reasoning_content"] != "need tool" {
		t.Fatalf("reasoning_content = %#v, want need tool; assistant=%+v", assistant["reasoning_content"], assistant)
	}
}

func TestDeepSeekAnthropicMapsThinkingToOutputEffort(t *testing.T) {
	var payload []byte
	p := NewDeepSeekAnthropic(ProviderConfig{APIKey: "test"})
	_, err := p.buildHTTPRequest(context.Background(), &Request{
		Model:    "deepseek-v4-pro",
		Messages: []Message{{Role: "user", Content: "hi"}},
		Thinking: &ThinkingConfig{Type: "enabled", Level: "xhigh"},
		OnPayload: func(provider string, raw []byte) {
			if provider != "deepseek-anthropic" {
				t.Fatalf("provider = %q, want deepseek-anthropic", provider)
			}
			payload = raw
		},
	}, false)
	if err != nil {
		t.Fatalf("buildHTTPRequest failed: %v", err)
	}

	var body map[string]any
	if err := json.Unmarshal(payload, &body); err != nil {
		t.Fatalf("unmarshal body: %v", err)
	}
	thinking, ok := body["thinking"].(map[string]any)
	if !ok {
		t.Fatalf("thinking = %#v, want object; body=%+v", body["thinking"], body)
	}
	if thinking["type"] != "enabled" {
		t.Fatalf("thinking.type = %#v, want enabled; body=%+v", thinking["type"], body)
	}
	if _, ok := thinking["budget_tokens"]; ok {
		t.Fatalf("thinking.budget_tokens should be omitted for DeepSeek Anthropic; thinking=%+v", thinking)
	}
	outputConfig, ok := body["output_config"].(map[string]any)
	if !ok {
		t.Fatalf("output_config = %#v, want object; body=%+v", body["output_config"], body)
	}
	if outputConfig["effort"] != "max" {
		t.Fatalf("output_config.effort = %#v, want max; body=%+v", outputConfig["effort"], body)
	}
}

func TestDeepSeekAnthropicDefaultBaseURL(t *testing.T) {
	p := NewDeepSeekAnthropic(ProviderConfig{APIKey: "test"})
	if got := p.Config().BaseURL; got != "https://api.deepseek.com/anthropic" {
		t.Fatalf("base URL = %q, want https://api.deepseek.com/anthropic", got)
	}
}

func buildDeepSeekTestBody(t *testing.T, thinking *ThinkingConfig) map[string]any {
	t.Helper()

	p := NewDeepSeek(ProviderConfig{})
	raw, err := p.buildRequestBody(&Request{
		Model:    "deepseek-v4-pro",
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

func requireDeepSeekThinking(t *testing.T, body map[string]any) map[string]any {
	t.Helper()

	thinking, ok := body["thinking"].(map[string]any)
	if !ok {
		t.Fatalf("thinking = %#v, want object; body=%+v", body["thinking"], body)
	}
	return thinking
}
