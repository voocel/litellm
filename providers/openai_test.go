package providers

import "testing"

func TestPrepareOpenAIMessagesUsesPrepareMessages(t *testing.T) {
	messages := []Message{
		{
			Role: "assistant",
			ToolCalls: []ToolCall{
				{
					ID:   "call.with.invalid/chars",
					Type: "function",
					Function: FunctionCall{
						Name:      "lookup_weather",
						Arguments: `{"city":"Paris"}`,
					},
				},
			},
		},
	}

	prepared, err := prepareOpenAIMessages(messages)
	if err != nil {
		t.Fatalf("prepareOpenAIMessages failed: %v", err)
	}
	if len(prepared) != 2 {
		t.Fatalf("prepared message count = %d, want 2", len(prepared))
	}
	if prepared[0].ToolCalls[0].ID != "call_with_invalid_chars" {
		t.Fatalf("tool call id = %q, want %q", prepared[0].ToolCalls[0].ID, "call_with_invalid_chars")
	}
	if prepared[1].Role != "tool" {
		t.Fatalf("synthetic message role = %q, want %q", prepared[1].Role, "tool")
	}
	if prepared[1].ToolCallID != "call_with_invalid_chars" {
		t.Fatalf("synthetic tool_call_id = %q, want %q", prepared[1].ToolCallID, "call_with_invalid_chars")
	}
}
