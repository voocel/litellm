package providers

import (
	"strings"
	"testing"
)

const bedrockClaudeModel = "anthropic.claude-sonnet-4-20250514-v1:0"

func buildBedrockReq(t *testing.T, model string, thinking *ThinkingConfig, maxTokens *int, temp *float64, onWarn func(string, string)) *bedrockRequest {
	t.Helper()
	p := &BedrockProvider{}
	built, err := p.buildRequest(&Request{
		Model:       model,
		Messages:    []Message{{Role: "user", Content: "hi"}},
		Thinking:    thinking,
		MaxTokens:   maxTokens,
		Temperature: temp,
		OnWarning:   onWarn,
	})
	if err != nil {
		t.Fatalf("buildRequest failed: %v", err)
	}
	return built
}

func bedrockThinkingField(t *testing.T, built *bedrockRequest) map[string]any {
	t.Helper()
	if built.AdditionalModelRequestFields == nil {
		t.Fatalf("AdditionalModelRequestFields is nil")
	}
	th, ok := built.AdditionalModelRequestFields["thinking"].(map[string]any)
	if !ok {
		t.Fatalf("thinking field missing/wrong type: %+v", built.AdditionalModelRequestFields)
	}
	return th
}

// Claude on Bedrock: level → thinking{type:enabled, budget_tokens} via the same
// extended-thinking path as the direct Anthropic provider; maxTokens defaults to
// 64000 so it stays above the budget.
func TestBedrockClaudeExtendedThinking(t *testing.T) {
	built := buildBedrockReq(t, bedrockClaudeModel, &ThinkingConfig{Type: "enabled", Level: "high"}, nil, nil, nil)
	th := bedrockThinkingField(t, built)
	if th["type"] != "enabled" {
		t.Fatalf("type = %#v, want enabled", th["type"])
	}
	if th["budget_tokens"] != 16384 {
		t.Fatalf("budget_tokens = %#v, want 16384 (high)", th["budget_tokens"])
	}
	if built.InferenceConfig == nil || built.InferenceConfig.MaxTokens != 64000 {
		t.Fatalf("maxTokens should default to 64000 to exceed budget; got %+v", built.InferenceConfig)
	}
}

func TestBedrockClaudeThinkingDisabled(t *testing.T) {
	built := buildBedrockReq(t, bedrockClaudeModel, &ThinkingConfig{Type: "disabled"}, nil, nil, nil)
	th := bedrockThinkingField(t, built)
	if th["type"] != "disabled" {
		t.Fatalf("type = %#v, want disabled", th["type"])
	}
	if _, ok := th["budget_tokens"]; ok {
		t.Fatalf("disabled must not carry budget_tokens: %+v", th)
	}
}

func TestBedrockClaudeThinkingRespectsExplicitMaxTokens(t *testing.T) {
	mt := 20000
	built := buildBedrockReq(t, bedrockClaudeModel, &ThinkingConfig{Type: "enabled", Level: "low"}, &mt, nil, nil)
	if built.InferenceConfig.MaxTokens != 20000 {
		t.Fatalf("explicit maxTokens should stay 20000; got %d", built.InferenceConfig.MaxTokens)
	}
	if th := bedrockThinkingField(t, built); th["budget_tokens"] != 2048 {
		t.Fatalf("budget_tokens = %#v, want 2048 (low)", th["budget_tokens"])
	}
}

// Non-Claude Bedrock families have no documented reasoning schema → drop with warning.
func TestBedrockNonClaudeOmitsThinkingWithWarning(t *testing.T) {
	var warned string
	built := buildBedrockReq(t, "amazon.nova-pro-v1:0", &ThinkingConfig{Type: "enabled", Level: "high"}, nil, nil,
		func(provider, msg string) { warned = provider + ": " + msg })
	if built.AdditionalModelRequestFields != nil {
		if _, ok := built.AdditionalModelRequestFields["thinking"]; ok {
			t.Fatalf("non-Claude must not carry thinking: %+v", built.AdditionalModelRequestFields)
		}
	}
	if !strings.Contains(warned, "bedrock") || !strings.Contains(warned, "thinking") {
		t.Fatalf("expected a thinking-omission warning for non-Claude model, got %q", warned)
	}
}

func TestBedrockOmitsThinkingWhenUnset(t *testing.T) {
	built := buildBedrockReq(t, bedrockClaudeModel, nil, nil, nil, nil)
	if built.AdditionalModelRequestFields != nil {
		if _, ok := built.AdditionalModelRequestFields["thinking"]; ok {
			t.Fatalf("no thinking expected when unset: %+v", built.AdditionalModelRequestFields)
		}
	}
}

// temperature != 1 with thinking enabled must error (Claude constraint, mirrors anthropic.go).
func TestBedrockClaudeThinkingRejectsNonOneTemperature(t *testing.T) {
	p := &BedrockProvider{}
	temp := 0.7
	if _, err := p.buildRequest(&Request{
		Model:       bedrockClaudeModel,
		Messages:    []Message{{Role: "user", Content: "hi"}},
		Thinking:    &ThinkingConfig{Type: "enabled", Level: "high"},
		Temperature: &temp,
	}); err == nil {
		t.Fatal("expected error when temperature != 1 with thinking enabled")
	}
}

// Bedrock Converse API toolResult contract:
//   - IsError=true  → toolResult.status = "error"
//   - success       → status omitted (relies on omitempty)
//   - always wrapped in a role=user message
// https://docs.aws.amazon.com/bedrock/latest/userguide/tool-use.html
func TestBedrockToolResultContract(t *testing.T) {
	p := &BedrockProvider{}
	req := &Request{
		Messages: []Message{
			{
				Role:      "assistant",
				ToolCalls: []ToolCall{{ID: "t1", Type: "function", Function: FunctionCall{Name: "do", Arguments: "{}"}}},
			},
			{Role: "tool", ToolCallID: "t1", Content: "boom", IsError: true},
		},
	}
	built, err := p.buildRequest(req)
	if err != nil {
		t.Fatalf("buildRequest failed: %v", err)
	}

	var tr *bedrockToolResult
	var hostRole string
	for _, m := range built.Messages {
		for _, c := range m.Content {
			if c.ToolResult != nil {
				tr = c.ToolResult
				hostRole = m.Role
			}
		}
	}
	if tr == nil {
		t.Fatalf("no toolResult emitted: %+v", built.Messages)
	}
	if hostRole != "user" {
		t.Fatalf("toolResult must ride in role=user, got %q", hostRole)
	}
	if tr.ToolUseID != "t1" {
		t.Fatalf("toolUseId mismatch, got %q", tr.ToolUseID)
	}
	if tr.Status != "error" {
		t.Fatalf("IsError=true must set status=\"error\", got %q", tr.Status)
	}
}
