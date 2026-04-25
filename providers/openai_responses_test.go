package providers

import (
	"testing"
)

// Core OpenAI Responses API contract per
// https://platform.openai.com/docs/api-reference/responses :
//   - user text      → {type: message, role: user, content: [input_text]}
//   - assistant text → {type: message, role: assistant, content: [output_text]}
//   - tool_calls     → {type: function_call, call_id, name, arguments}
//   - tool result    → {type: function_call_output, call_id, output}
//   - role=tool never appears as a message role
//
// A single full conversation exercises every item type and their ordering.
func TestResponsesInputConvertsAllItemTypes(t *testing.T) {
	items := convertMessagesToResponsesInput([]Message{
		{Role: "user", Content: "weather?"},
		{
			Role:    "assistant",
			Content: "let me check",
			ToolCalls: []ToolCall{
				{ID: "call_1", Type: "function", Function: FunctionCall{Name: "get_weather", Arguments: `{"city":"Paris"}`}},
			},
		},
		{Role: "tool", ToolCallID: "call_1", Content: `{"temp":"15C"}`},
	})

	if len(items) != 4 {
		t.Fatalf("want 4 items (user msg, assistant msg, function_call, function_call_output), got %d: %+v", len(items), items)
	}

	// user
	if items[0].Type != "message" || items[0].Role != "user" ||
		len(items[0].Content) != 1 || items[0].Content[0].Type != "input_text" {
		t.Fatalf("user item malformed: %+v", items[0])
	}
	// assistant text (must precede function_call items — mirrors model output order)
	if items[1].Type != "message" || items[1].Role != "assistant" ||
		len(items[1].Content) != 1 || items[1].Content[0].Type != "output_text" {
		t.Fatalf("assistant message item malformed: %+v", items[1])
	}
	// function_call
	fc := items[2]
	if fc.Type != "function_call" || fc.CallID != "call_1" || fc.Name != "get_weather" || fc.Arguments == "" {
		t.Fatalf("function_call item malformed: %+v", fc)
	}
	if fc.Role != "" || len(fc.Content) != 0 {
		t.Fatalf("function_call must not carry role/content: %+v", fc)
	}
	// function_call_output
	fo := items[3]
	if fo.Type != "function_call_output" || fo.CallID != "call_1" || fo.Output == "" {
		t.Fatalf("function_call_output item malformed: %+v", fo)
	}
	if fo.Role == "tool" {
		t.Fatalf("role=tool must never leak into input items")
	}
}

func TestResponsesToolsNilStrictUsesResponsesStrictDefault(t *testing.T) {
	tools, err := convertResponsesAPITools([]Tool{{
		Type: "function",
		Function: FunctionDef{
			Name: "get_weather",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"city": map[string]any{"type": "string"},
				},
				"required": []any{"city"},
			},
		},
	}})
	if err != nil {
		t.Fatalf("convertResponsesAPITools: %v", err)
	}
	if tools[0].Strict != nil {
		t.Fatalf("strict should be omitted so Responses API default applies: %v", tools[0].Strict)
	}
	params := tools[0].Parameters.(map[string]any)
	if params["additionalProperties"] != false {
		t.Fatalf("nil strict should still normalise schema for Responses default strict=true: %v", params)
	}
}
