package litellm

// IntPtr returns a pointer to an int value
func IntPtr(v int) *int {
	return &v
}

// Float64Ptr returns a pointer to a float64 value
func Float64Ptr(v float64) *float64 {
	return &v
}

// BoolPtr returns a pointer to a bool value
func BoolPtr(v bool) *bool {
	return &v
}

// NewResponseFormatText creates a text response format
func NewResponseFormatText() *ResponseFormat {
	return &ResponseFormat{Type: ResponseFormatText}
}

// NewResponseFormatJSONObject creates a JSON object response format
func NewResponseFormatJSONObject() *ResponseFormat {
	return &ResponseFormat{Type: ResponseFormatJSONObject}
}

// NewResponseFormatJSONSchema creates a JSON schema response format
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
