package providers

import "strings"

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
		ModelFromResponse:         true,
		ContentAsInterface:        true,
		ReasoningField:            "reasoning",
		HasCompletionTokenDetails: true,
		SupportsJSONSchema:        true,
		// OpenRouter uses {"reasoning": {"effort": ...}} OR {"reasoning": {"max_tokens": ...}},
		// but NOT both simultaneously.
		ThinkingMapper: func(thinking *ThinkingConfig, _ string) map[string]any {
			if !isThinkingEnabledConfig(thinking) {
				return nil
			}
			reasoning := map[string]any{}
			if thinking.BudgetTokens != nil && *thinking.BudgetTokens > 0 {
				// Explicit budget → use max_tokens only.
				reasoning["max_tokens"] = *thinking.BudgetTokens
			} else if thinking.Level != "" {
				// Level → use effort only.
				reasoning["effort"] = thinking.Level
			} else {
				reasoning["effort"] = "medium"
			}
			return map[string]any{"reasoning": reasoning}
		},
		ExtraHeaders: map[string]string{
			"HTTP-Referer": "https://github.com/voocel/litellm",
			"X-Title":      "litellm",
		},
		CustomMessageConverter: convertOpenRouterMessages,
		CleanSchema:            ensureStrictSchemaGeneric,
		ExtraTransform:         openRouterExtraTransform,
	})
}

// openRouterExtraTransform translates provider-agnostic Extra options into
// OpenRouter-native request fields, then strips them from Extra so they don't
// leak through verbatim.
//
// Currently handled:
//   - "cache_retention": placed as top-level cache_control with optional ttl
//     ("none" → omit; "" / "short" / "5m" → {"type":"ephemeral"}; "long" / "1h"
//     → {"type":"ephemeral","ttl":"1h"}). Per OpenRouter docs, the top-level
//     form is only honored for Anthropic-routed models, so the field is only
//     emitted when the model id starts with "anthropic/". Other providers
//     (OpenAI, Google, etc.) handle prompt caching automatically without any
//     request-side marker.
func openRouterExtraTransform(extra map[string]any, body map[string]any, req *Request) map[string]any {
	if len(extra) == 0 {
		return extra
	}
	out := make(map[string]any, len(extra))
	for k, v := range extra {
		if k == "cache_retention" {
			retention, _ := v.(string)
			if cc := resolveOpenRouterCacheControl(retention, req.Model); cc != nil {
				body["cache_control"] = cc
			}
			continue
		}
		out[k] = v
	}
	return out
}

// resolveOpenRouterCacheControl maps the cache_retention string to OpenRouter's
// top-level cache_control object. Returns nil when caching should be skipped
// (disabled or routed to a provider that auto-caches without breakpoints).
func resolveOpenRouterCacheControl(retention, model string) map[string]any {
	if !strings.HasPrefix(strings.ToLower(model), "anthropic/") {
		return nil
	}
	switch strings.ToLower(strings.TrimSpace(retention)) {
	case "none":
		return nil
	case "long", "1h":
		return map[string]any{"type": "ephemeral", "ttl": "1h"}
	default:
		return map[string]any{"type": "ephemeral"}
	}
}

// convertOpenRouterMessages handles Anthropic-style cache_control on messages.
func convertOpenRouterMessages(messages []Message) any {
	result := make([]openRouterMessage, len(messages))
	for i, msg := range messages {
		m := openRouterMessage{
			Role:       msg.Role,
			ToolCallID: msg.ToolCallID,
		}

		// Tool result messages require plain string content per OpenAI spec;
		// wrapping in content-parts array causes 400 on some providers.
		if msg.Role == "tool" {
			m.Content = msg.Content
		} else if msg.CacheControl != nil {
			// Forward TTL alongside Type. OpenRouter passes explicit
			// breakpoints through to underlying providers verbatim, and
			// Anthropic-family backends require ttl="1h" for the 1-hour
			// cache tier — dropping it silently downgrades to 5 minutes.
			m.Content = []openRouterContent{{
				Type: "text",
				Text: msg.Content,
				CacheControl: &openRouterCacheControl{
					Type: msg.CacheControl.Type,
					TTL:  msg.CacheControl.TTL,
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
	TTL  string `json:"ttl,omitempty"`
}
