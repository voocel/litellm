package providers

import "testing"

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
