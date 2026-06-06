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

func TestDeepSeekPrefersReasoningContentOverSummaryFields(t *testing.T) {
	p := NewDeepSeek(ProviderConfig{})
	reasoning, field := p.compat.findReasoning(map[string]any{
		"reasoning_summary": map[string]any{"text": "summary only"},
		"reasoning_details": []any{
			map[string]any{"text": "detail only"},
		},
		"reasoning_content": "full reasoning content",
	})

	if field != "reasoning_content" {
		t.Fatalf("field = %q, want reasoning_content", field)
	}
	if reasoning != "full reasoning content" {
		t.Fatalf("reasoning = %q, want full reasoning content", reasoning)
	}
}

func TestDeepSeekResponseConversionStoresFullReasoningContent(t *testing.T) {
	p := NewDeepSeek(ProviderConfig{})
	rawMessage := json.RawMessage(`{
		"role": "assistant",
		"content": "",
		"reasoning_summary": {"text": "summary only"},
		"reasoning_details": [{"text": "detail only"}],
		"reasoning_content": "full reasoning content",
		"tool_calls": [{
			"id": "call_1",
			"type": "function",
			"function": {"name": "lookup", "arguments": "{}"}
		}]
	}`)

	resp, err := p.convertResponse(&compatResponse{
		Model: "deepseek-v4-flash",
		Choices: []compatChoice{{
			Message:      rawMessage,
			FinishReason: "tool_calls",
		}},
	}, &Request{
		Model:    "deepseek-v4-flash",
		Thinking: &ThinkingConfig{Type: "enabled", Level: "high"},
	})
	if err != nil {
		t.Fatalf("convertResponse failed: %v", err)
	}

	if resp.ReasoningContent != "full reasoning content" {
		t.Fatalf("reasoning_content = %q, want full reasoning content", resp.ReasoningContent)
	}
	if len(resp.ToolCalls) != 1 {
		t.Fatalf("tool calls len = %d, want 1", len(resp.ToolCalls))
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

func TestDeepSeekAnthropicPreservesReasoningContentForToolTurns(t *testing.T) {
	var payload []byte
	p := NewDeepSeekAnthropic(ProviderConfig{APIKey: "test"})
	_, err := p.buildHTTPRequest(context.Background(), &Request{
		Model: "deepseek-v4-pro",
		Messages: []Message{
			{Role: "user", Content: "hi"},
			{
				Role:             "assistant",
				Content:          "need tool",
				ReasoningContent: "thinking before tool",
				ToolCalls: []ToolCall{{
					ID:   "call_1",
					Type: "function",
					Function: FunctionCall{
						Name:      "lookup",
						Arguments: `{"q":"x"}`,
					},
				}},
			},
			{Role: "tool", ToolCallID: "call_1", Content: "ok"},
		},
		Thinking: &ThinkingConfig{Type: "enabled", Level: "high"},
		OnPayload: func(provider string, raw []byte) {
			payload = raw
		},
	}, false)
	if err != nil {
		t.Fatalf("buildHTTPRequest failed: %v", err)
	}

	var body struct {
		Messages []struct {
			Role    string           `json:"role"`
			Content []map[string]any `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(payload, &body); err != nil {
		t.Fatalf("unmarshal payload: %v", err)
	}
	if len(body.Messages) < 2 {
		t.Fatalf("messages len = %d, want at least 2", len(body.Messages))
	}
	assistant := body.Messages[1]
	if assistant.Role != "assistant" {
		t.Fatalf("message[1].role = %q, want assistant", assistant.Role)
	}
	var foundThinking bool
	for _, block := range assistant.Content {
		if block["type"] == "thinking" && block["thinking"] == "thinking before tool" {
			foundThinking = true
		}
	}
	if !foundThinking {
		t.Fatalf("assistant content missing thinking block: %+v", assistant.Content)
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
