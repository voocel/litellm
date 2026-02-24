package providers

func init() {
	RegisterBuiltin("openrouter", func(cfg ProviderConfig) Provider {
		return NewOpenRouter(cfg)
	}, "https://openrouter.ai/api/v1")
}

// NewOpenRouter creates a new OpenRouter provider.
func NewOpenRouter(config ProviderConfig) *OpenAICompatProvider {
	return NewOpenAICompat(config, Compat{
		ProviderName:              "openrouter",
		DefaultBaseURL:            "https://openrouter.ai/api/v1",
		IncludeStreamUsage:        true,
		ModelFromResponse:         true,
		ContentAsInterface:        true,
		ReasoningField:            "reasoning",
		HasCompletionTokenDetails: true,
		SupportsJSONSchema:        true,
		ExtraHeaders: map[string]string{
			"HTTP-Referer": "https://github.com/voocel/litellm",
			"X-Title":      "litellm",
		},
		CustomMessageConverter: convertOpenRouterMessages,
		CleanSchema:            ensureStrictSchemaGeneric,
	})
}

// convertOpenRouterMessages handles Anthropic-style cache_control on messages.
func convertOpenRouterMessages(messages []Message) any {
	result := make([]openRouterMessage, len(messages))
	for i, msg := range messages {
		m := openRouterMessage{
			Role:       msg.Role,
			ToolCallID: msg.ToolCallID,
		}

		if msg.CacheControl != nil {
			m.Content = []openRouterContent{{
				Type: "text",
				Text: msg.Content,
				CacheControl: &openRouterCacheControl{
					Type: msg.CacheControl.Type,
				},
			}}
		} else if parts := buildOpenAIContentParts(msg); len(parts) > 0 {
			m.Content = parts
		} else {
			m.Content = msg.Content
		}

		if len(msg.ToolCalls) > 0 {
			m.ToolCalls = make([]openaiToolCall, len(msg.ToolCalls))
			for j, tc := range msg.ToolCalls {
				m.ToolCalls[j] = openaiToolCall{
					ID:   tc.ID,
					Type: tc.Type,
					Function: openaiToolCallFunc{
						Name:      tc.Function.Name,
						Arguments: tc.Function.Arguments,
					},
				}
			}
		}

		result[i] = m
	}
	return result
}

// ensureStrictSchemaGeneric recursively adds additionalProperties: false
// to all objects for OpenAI strict JSON schema mode.
func ensureStrictSchemaGeneric(schema any) any {
	switch s := schema.(type) {
	case map[string]any:
		result := make(map[string]any, len(s))
		for k, v := range s {
			result[k] = ensureStrictSchemaGeneric(v)
		}
		if result["type"] == "object" {
			result["additionalProperties"] = false
		}
		return result
	case []any:
		cleaned := make([]any, len(s))
		for i, v := range s {
			cleaned[i] = ensureStrictSchemaGeneric(v)
		}
		return cleaned
	default:
		return schema
	}
}

// ---------------------------------------------------------------------------
// OpenRouter-specific types (minimal, for cache_control support)
// ---------------------------------------------------------------------------

type openRouterMessage struct {
	Role       string           `json:"role"`
	Content    any              `json:"content,omitempty"`
	ToolCalls  []openaiToolCall `json:"tool_calls,omitempty"`
	ToolCallID string           `json:"tool_call_id,omitempty"`
}

type openRouterContent struct {
	Type         string                  `json:"type"`
	Text         string                  `json:"text"`
	CacheControl *openRouterCacheControl `json:"cache_control,omitempty"`
}

type openRouterCacheControl struct {
	Type string `json:"type"`
}
