package litellm

// Pointer helper functions for optional parameters

// IntPtr returns a pointer to an int value
// Example: req.MaxTokens = litellm.IntPtr(2048)
func IntPtr(v int) *int {
	return &v
}

// Float64Ptr returns a pointer to a float64 value
// Example: req.Temperature = litellm.Float64Ptr(0.7)
func Float64Ptr(v float64) *float64 {
	return &v
}

// BoolPtr returns a pointer to a bool value
// Example: enabled := litellm.BoolPtr(true)
func BoolPtr(v bool) *bool {
	return &v
}

// StringPtr returns a pointer to a string value
// Example: req.User = litellm.StringPtr("user-123")
func StringPtr(v string) *string {
	return &v
}

// ResponseFormat helper functions

// NewResponseFormatText creates a text response format
func NewResponseFormatText() *ResponseFormat {
	return &ResponseFormat{Type: ResponseFormatText}
}

// NewResponseFormatJSONObject creates a JSON object response format
// This ensures the model returns valid JSON without enforcing a specific schema
func NewResponseFormatJSONObject() *ResponseFormat {
	return &ResponseFormat{Type: ResponseFormatJSONObject}
}

// NewResponseFormatJSONSchema creates a JSON schema response format
// with strict validation enabled/disabled
//
// Parameters:
//   - name: Schema name (required)
//   - description: Schema description (optional, can be empty)
//   - schema: JSON Schema definition as a map[string]interface{}
//   - strict: Enable strict schema validation (OpenAI only)
//
// Example:
//
//	schema := map[string]interface{}{
//	    "type": "object",
//	    "properties": map[string]interface{}{
//	        "name": map[string]interface{}{"type": "string"},
//	        "age": map[string]interface{}{"type": "integer"},
//	    },
//	    "required": []string{"name", "age"},
//	}
//	format := litellm.NewResponseFormatJSONSchema("person", "A person object", schema, true)
func NewResponseFormatJSONSchema(name, description string, schema any, strict bool) *ResponseFormat {
	return &ResponseFormat{
		Type: ResponseFormatJSONSchema,
		JSONSchema: &JSONSchema{
			Name:        name,
			Description: description,
			Schema:      schema,
			Strict:      BoolPtr(strict),
		},
	}
}

// Message helper functions

// UserMessage creates a user message
// Example: litellm.UserMessage("Hello, AI!")
func UserMessage(content string) Message {
	return Message{Role: "user", Content: content}
}

// AssistantMessage creates an assistant message
// Example: litellm.AssistantMessage("Hello! How can I help you?")
func AssistantMessage(content string) Message {
	return Message{Role: "assistant", Content: content}
}

// SystemMessage creates a system message
// Example: litellm.SystemMessage("You are a helpful assistant.")
func SystemMessage(content string) Message {
	return Message{Role: "system", Content: content}
}

// ToolMessage creates a tool response message
// Example: litellm.ToolMessage("call_abc123", `{"result": "success"}`)
func ToolMessage(toolCallID, content string) Message {
	return Message{Role: "tool", ToolCallID: toolCallID, Content: content}
}
