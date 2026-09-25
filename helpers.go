package litellm

// Text returns a TextBlock.
func Text(text string) TextBlock {
	return TextBlock{Text: text}
}

// ImageURL returns an ImageBlock for a URL, which may be a data URL.
func ImageURL(url string) ImageBlock {
	return ImageBlock{URL: url}
}

// User returns a user message.
func User(blocks ...Block) Message {
	return Message{Role: RoleUser, Blocks: blocks}
}

// UserText returns a user message with one text block.
func UserText(text string) Message {
	return User(Text(text))
}

// System returns a system message.
func System(text string) Message {
	return Message{Role: RoleSystem, Blocks: []Block{Text(text)}}
}

// Assistant returns an assistant message, such as a prior reply to replay.
func Assistant(blocks ...Block) Message {
	return Message{Role: RoleAssistant, Blocks: blocks}
}

// AssistantText returns an assistant message with one text block.
func AssistantText(text string) Message {
	return Assistant(Text(text))
}

// ToolResult returns a tool message answering the tool call toolUseID.
func ToolResult(toolUseID string, blocks ...Block) Message {
	return Message{
		Role: RoleTool,
		Blocks: []Block{
			ToolResultBlock{ToolUseID: toolUseID, Content: blocks},
		},
	}
}

// ToolResultText returns a tool message with a text result.
func ToolResultText(toolUseID, text string) Message {
	return ToolResult(toolUseID, Text(text))
}

// NewResponseFormatText requests plain text output.
func NewResponseFormatText() *ResponseFormat {
	return &ResponseFormat{Type: ResponseFormatText}
}

// NewResponseFormatJSONObject requests a JSON object.
func NewResponseFormatJSONObject() *ResponseFormat {
	return &ResponseFormat{Type: ResponseFormatJSONObject}
}

// NewResponseFormatJSONSchema requests output matching schema, converted with
// SchemaFrom.
func NewResponseFormatJSONSchema(name, description string, schema any, strict StrictMode) (*ResponseFormat, error) {
	s, err := SchemaFrom(schema)
	if err != nil {
		return nil, err
	}
	return &ResponseFormat{
		Type: ResponseFormatJSONSchema,
		JSONSchema: &JSONSchema{
			Name:        name,
			Description: description,
			Schema:      s,
			Strict:      strict,
		},
	}, nil
}
