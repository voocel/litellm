package providers

import (
	"strings"
	"testing"
)

// PrepareMessages is the single boundary where malformed tool-call inputs are
// either rejected loud (missing name / missing tool_call_id) or repaired
// silently (missing id on assistant tool_call → synthesize, matching the
// existing orphan-compensation pattern).
func TestPrepareMessagesValidation(t *testing.T) {
	t.Run("rejects empty function name with message index", func(t *testing.T) {
		_, err := PrepareMessages([]Message{
			{Role: "user", Content: "hi"},
			{
				Role: "assistant",
				ToolCalls: []ToolCall{
					{ID: "call_1", Type: "function", Function: FunctionCall{Name: ""}},
				},
			},
		})
		if err == nil || !strings.Contains(err.Error(), "messages[1]") {
			t.Fatalf("want error referencing messages[1], got %v", err)
		}
	})

	t.Run("rejects tool message without tool_call_id", func(t *testing.T) {
		_, err := PrepareMessages([]Message{
			{Role: "tool", Content: "result"},
		})
		if err == nil || !strings.Contains(err.Error(), "tool_call_id") {
			t.Fatalf("want error mentioning tool_call_id, got %v", err)
		}
	})

	t.Run("synthesizes missing tool_call ID (fix-forward)", func(t *testing.T) {
		prepared, err := PrepareMessages([]Message{
			{
				Role: "assistant",
				ToolCalls: []ToolCall{
					{ID: "", Type: "function", Function: FunctionCall{Name: "lookup"}},
				},
			},
		})
		if err != nil {
			t.Fatalf("missing ID should be synthesized, got error: %v", err)
		}
		if len(prepared) == 0 || prepared[0].ToolCalls[0].ID == "" {
			t.Fatalf("tool_call ID not synthesized: %+v", prepared)
		}
	})
}
