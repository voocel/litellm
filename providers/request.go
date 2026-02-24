package providers

import (
	"fmt"
	"regexp"
	"strings"
)

// ---------------------------------------------------------------------------
// Message types — message, content blocks, and tool calls
// ---------------------------------------------------------------------------

type Message struct {
	Role         string           `json:"role"`
	Content      string           `json:"content"`
	Contents     []MessageContent `json:"contents,omitempty"`
	ToolCalls    []ToolCall       `json:"tool_calls,omitempty"`
	ToolCallID   string           `json:"tool_call_id,omitempty"`
	IsError      bool             `json:"is_error,omitempty"` // tool result error flag (Anthropic)
	CacheControl *CacheControl    `json:"cache_control,omitempty"`
}

type MessageContent struct {
	Type        string           `json:"type"`
	Text        string           `json:"text,omitempty"`
	ImageURL    *MessageImageURL `json:"image_url,omitempty"`
	Annotations []map[string]any `json:"annotations,omitempty"`
	Logprobs    []map[string]any `json:"logprobs,omitempty"`
}

type MessageImageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"` // "auto", "low", or "high"
}

// CacheControl defines prompt caching behavior for providers.
type CacheControl struct {
	Type string `json:"type"`          // "ephemeral" or "persistent"
	TTL  *int   `json:"ttl,omitempty"` // TTL in seconds (optional)
}

type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function FunctionCall `json:"function"`
}

type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// ---------------------------------------------------------------------------
// Request-side types — tools and request configuration
// ---------------------------------------------------------------------------

type Tool struct {
	Type     string      `json:"type"`
	Function FunctionDef `json:"function"`
}

type FunctionDef struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Parameters  any    `json:"parameters"`
}

type ResponseFormat struct {
	Type       string      `json:"type"`
	JSONSchema *JSONSchema `json:"json_schema,omitempty"`
}

type JSONSchema struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Schema      any    `json:"schema"`
	Strict      *bool  `json:"strict,omitempty"`
}

type ThinkingConfig struct {
	Type         string `json:"type"`            // "enabled" or "disabled"
	Level        string `json:"level,omitempty"` // "low", "medium", "high" — provider translates to API-specific param
	BudgetTokens *int   `json:"budget_tokens,omitempty"`
}

type Request struct {
	Model          string          `json:"model"`
	Messages       []Message       `json:"messages"`
	MaxTokens      *int            `json:"max_tokens,omitempty"`
	Temperature    *float64        `json:"temperature,omitempty"`
	TopP           *float64        `json:"top_p,omitempty"`
	Tools          []Tool          `json:"tools,omitempty"`
	ToolChoice     any             `json:"tool_choice,omitempty"`
	ResponseFormat *ResponseFormat `json:"response_format,omitempty"`
	Stop           []string        `json:"stop,omitempty"`
	Thinking       *ThinkingConfig `json:"thinking,omitempty"`

	// APIKey overrides the provider-level API key for this single request.
	// When empty, the provider's default key is used.
	// Enables key rotation, OAuth short-lived tokens, and multi-tenant scenarios.
	APIKey string `json:"-"`

	// Provider-specific extensions
	Extra map[string]any `json:"extra,omitempty"`

	// OnPayload is called with the serialized JSON body before sending
	// the HTTP request. Useful for debugging and logging API calls.
	OnPayload func(provider string, payload []byte) `json:"-"`
}

// ---------------------------------------------------------------------------
// Thinking — normalization, validation, and budget resolution
// ---------------------------------------------------------------------------

// defaultThinkingBudgets maps reasoning levels to default token budgets.
var defaultThinkingBudgets = map[string]int{
	"minimal": 1024,
	"low":     2048,
	"medium":  8192,
	"high":    16384,
}

// LevelToBudget returns the default thinking token budget for a reasoning level.
// Returns 0 if the level is unknown or empty.
func LevelToBudget(level string) int {
	return defaultThinkingBudgets[strings.ToLower(strings.TrimSpace(level))]
}

// ResolveBudgetTokens returns a thinking budget: explicit BudgetTokens if set,
// otherwise derived from Level via LevelToBudget.
// Returns nil if neither is set or the level is unknown.
func ResolveBudgetTokens(thinking *ThinkingConfig) *int {
	if thinking == nil || thinking.Type != "enabled" {
		return nil
	}
	if thinking.BudgetTokens != nil {
		return thinking.BudgetTokens
	}
	if budget := LevelToBudget(thinking.Level); budget > 0 {
		return &budget
	}
	return nil
}

func normalizeThinking(req *Request) *ThinkingConfig {
	if req == nil || req.Thinking == nil {
		return nil
	}

	thinkingType := strings.TrimSpace(req.Thinking.Type)
	if thinkingType == "" {
		thinkingType = "enabled"
	}

	return &ThinkingConfig{
		Type:         thinkingType,
		Level:        req.Thinking.Level,
		BudgetTokens: req.Thinking.BudgetTokens,
	}
}

func isThinkingDisabled(req *Request) bool {
	if req == nil || req.Thinking == nil {
		return false
	}
	return strings.EqualFold(strings.TrimSpace(req.Thinking.Type), "disabled")
}

func validateThinking(thinking *ThinkingConfig) error {
	if thinking == nil {
		return nil
	}
	thinkingType := strings.TrimSpace(thinking.Type)
	if thinkingType == "" {
		return nil
	}
	if strings.EqualFold(thinkingType, "enabled") || strings.EqualFold(thinkingType, "disabled") {
		return nil
	}
	return fmt.Errorf("thinking type must be enabled or disabled")
}

// ---------------------------------------------------------------------------
// Message preprocessing — sanitization, normalization, orphan repair
// ---------------------------------------------------------------------------

// toolCallIDForbidden matches characters NOT allowed in tool call IDs across
// all major providers. Anthropic is the most restrictive: ^[a-zA-Z0-9_-]+$, max 64.
var toolCallIDForbidden = regexp.MustCompile(`[^a-zA-Z0-9_-]`)

const maxToolCallIDLen = 64

// NormalizeToolCallID sanitizes a tool call ID for cross-provider compatibility.
// Keeps only [a-zA-Z0-9_-] and truncates to 64 characters (Anthropic limit).
// Already-compliant IDs pass through unchanged.
func NormalizeToolCallID(id string) string {
	out := toolCallIDForbidden.ReplaceAllString(id, "_")
	if len(out) > maxToolCallIDLen {
		out = out[:maxToolCallIDLen]
	}
	return out
}

// PrepareMessages preprocesses a message slice for API submission:
//   - Sanitizes invalid UTF-8 (including surrogate codepoints) in content
//   - Skips error assistant messages and their associated tool results
//   - Normalizes tool call IDs for cross-provider compatibility
//   - Inserts synthetic error tool results for orphaned tool calls
//     (assistant tool_calls with no matching tool result before next turn)
//
// The original slice is not modified; a new slice is returned.
func PrepareMessages(messages []Message) []Message {
	if len(messages) == 0 {
		return messages
	}

	// idMap tracks original → normalized tool call ID mappings
	idMap := make(map[string]string)

	// skipToolIDs collects tool call IDs from skipped error assistant messages
	// so their tool results are also skipped.
	skipToolIDs := make(map[string]bool)

	result := make([]Message, 0, len(messages)+4)
	var pendingToolCalls []ToolCall
	existingResults := make(map[string]bool)

	for _, msg := range messages {
		// Sanitize invalid UTF-8 sequences (including surrogate codepoints)
		msg = sanitizeMessage(msg)
		switch msg.Role {
		case "assistant":
			// Flush orphaned tool calls from previous assistant turn
			result = flushOrphanedToolCalls(result, pendingToolCalls, existingResults)
			pendingToolCalls = nil
			existingResults = make(map[string]bool)

			// Skip error/aborted assistant messages — they pollute context
			if msg.IsError {
				for _, tc := range msg.ToolCalls {
					skipToolIDs[tc.ID] = true
				}
				continue
			}

			// Normalize tool call IDs
			if len(msg.ToolCalls) > 0 {
				copied := msg
				copied.ToolCalls = make([]ToolCall, len(msg.ToolCalls))
				for i, tc := range msg.ToolCalls {
					normalized := NormalizeToolCallID(tc.ID)
					if normalized != tc.ID {
						idMap[tc.ID] = normalized
					}
					copied.ToolCalls[i] = tc
					copied.ToolCalls[i].ID = normalized
					pendingToolCalls = append(pendingToolCalls, copied.ToolCalls[i])
				}
				result = append(result, copied)
			} else {
				result = append(result, msg)
			}

		case "tool":
			// Skip tool results for skipped error assistant messages
			if skipToolIDs[msg.ToolCallID] {
				continue
			}

			// Normalize the tool_call_id reference to match the normalized ID
			toolCallID := msg.ToolCallID
			if mapped, ok := idMap[toolCallID]; ok {
				toolCallID = mapped
			} else {
				toolCallID = NormalizeToolCallID(toolCallID)
			}

			if toolCallID != msg.ToolCallID {
				msg = Message{
					Role:         msg.Role,
					Content:      msg.Content,
					Contents:     msg.Contents,
					ToolCalls:    msg.ToolCalls,
					ToolCallID:   toolCallID,
					IsError:      msg.IsError,
					CacheControl: msg.CacheControl,
				}
			}
			existingResults[toolCallID] = true
			result = append(result, msg)

		case "user":
			// User message interrupts tool flow — flush orphaned calls
			result = flushOrphanedToolCalls(result, pendingToolCalls, existingResults)
			pendingToolCalls = nil
			existingResults = make(map[string]bool)
			result = append(result, msg)

		default:
			result = append(result, msg)
		}
	}

	// Final flush for any remaining orphaned tool calls at end of conversation
	result = flushOrphanedToolCalls(result, pendingToolCalls, existingResults)

	return result
}

// flushOrphanedToolCalls appends synthetic error tool results for tool calls
// that have no corresponding tool result message.
func flushOrphanedToolCalls(result []Message, pending []ToolCall, existing map[string]bool) []Message {
	for _, tc := range pending {
		if !existing[tc.ID] {
			result = append(result, Message{
				Role:       "tool",
				Content:    "Tool execution was interrupted — no result available.",
				ToolCallID: tc.ID,
				IsError:    true,
			})
		}
	}
	return result
}

// sanitizeMessage removes invalid UTF-8 sequences (including surrogate codepoints
// U+D800-DFFF) from message content fields. Invalid bytes in user input can cause
// json.Marshal to produce malformed JSON that API providers reject.
func sanitizeMessage(msg Message) Message {
	msg.Content = strings.ToValidUTF8(msg.Content, "")
	if len(msg.Contents) > 0 {
		cleaned := make([]MessageContent, len(msg.Contents))
		for i, c := range msg.Contents {
			c.Text = strings.ToValidUTF8(c.Text, "")
			cleaned[i] = c
		}
		msg.Contents = cleaned
	}
	return msg
}
