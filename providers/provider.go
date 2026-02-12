package providers

import (
	"context"
)

type Message struct {
	Role         string           `json:"role"`
	Content      string           `json:"content"`
	Contents     []MessageContent `json:"contents,omitempty"`
	ToolCalls    []ToolCall       `json:"tool_calls,omitempty"`
	ToolCallID   string           `json:"tool_call_id,omitempty"`
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

// CacheControl defines prompt caching behavior for providers
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

	// Provider-specific extensions
	Extra map[string]any `json:"extra,omitempty"`
}

type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
	ReasoningTokens  int `json:"reasoning_tokens,omitempty"`

	// Cache-related token statistics
	CacheCreationInputTokens int `json:"cache_creation_input_tokens,omitempty"` // Tokens written to cache
	CacheReadInputTokens     int `json:"cache_read_input_tokens,omitempty"`     // Tokens read from cache
}

type ReasoningData struct {
	Summary    string `json:"summary,omitempty"`
	Content    string `json:"content,omitempty"`
	TokensUsed int    `json:"tokens_used,omitempty"`
}

type Response struct {
	Content      string           `json:"content"`
	Contents     []MessageContent `json:"contents,omitempty"`
	ToolCalls    []ToolCall       `json:"tool_calls,omitempty"`
	Usage        Usage            `json:"usage"`
	Model        string           `json:"model"`
	Provider     string           `json:"provider"`
	FinishReason string           `json:"finish_reason,omitempty"`
	Reasoning    *ReasoningData   `json:"reasoning,omitempty"`
}

type StreamChunk struct {
	Type          string          `json:"type"`
	Content       string          `json:"content,omitempty"`
	ContentIndex  *int            `json:"content_index,omitempty"`
	OutputIndex   *int            `json:"output_index,omitempty"`
	ItemID        string          `json:"item_id,omitempty"`
	ToolCallDelta *ToolCallDelta  `json:"tool_call_delta,omitempty"`
	FinishReason  string          `json:"finish_reason,omitempty"`
	Model         string          `json:"model,omitempty"`
	Provider      string          `json:"provider"`
	Done          bool            `json:"done"`
	Reasoning     *ReasoningChunk `json:"reasoning,omitempty"`
	Usage         *Usage          `json:"usage,omitempty"`
}

type ToolCallDelta struct {
	Index          int    `json:"index"`
	ID             string `json:"id,omitempty"`
	Type           string `json:"type,omitempty"`
	FunctionName   string `json:"function_name,omitempty"`
	ArgumentsDelta string `json:"arguments_delta,omitempty"`
	OutputIndex    *int   `json:"output_index,omitempty"`
	ItemID         string `json:"item_id,omitempty"`
}

type ReasoningChunk struct {
	Summary string `json:"summary,omitempty"`
	Content string `json:"content,omitempty"`
	Done    bool   `json:"done,omitempty"`
}

// ModelInfo provides a minimal, provider-agnostic model descriptor.
// Fields are best-effort and may be empty if a provider does not supply them.
type ModelInfo struct {
	ID               string `json:"id"`
	Name             string `json:"name,omitempty"`
	Provider         string `json:"provider,omitempty"`
	Description      string `json:"description,omitempty"`
	ContextLength    int    `json:"context_length,omitempty"`
	InputTokenLimit  int    `json:"input_token_limit,omitempty"`
	OutputTokenLimit int    `json:"output_token_limit,omitempty"`
	Created          int64  `json:"created,omitempty"`
	OwnedBy          string `json:"owned_by,omitempty"`

	// Capability flags (best-effort, may be zero-valued for unknown models)
	SupportsTools    bool `json:"supports_tools,omitempty"`
	SupportsVision   bool `json:"supports_vision,omitempty"`
	SupportsThinking bool `json:"supports_thinking,omitempty"`
}

// ---------------------------------------------------------------------------
// FinishReason constants — canonical values returned by all providers.
// ---------------------------------------------------------------------------

const (
	FinishReasonStop     = "stop"
	FinishReasonLength   = "length"
	FinishReasonToolCall = "tool_calls"
	FinishReasonError    = "error"
	FinishReasonSafety   = "safety"
)

// NormalizeFinishReason maps provider-specific stop reasons to canonical constants.
// Unknown values pass through unchanged.
func NormalizeFinishReason(raw string) string {
	switch raw {
	case "stop", "end_turn", "STOP", "stop_sequence":
		return FinishReasonStop
	case "length", "max_tokens", "MAX_TOKENS":
		return FinishReasonLength
	case "tool_calls", "tool_use", "FUNCTION_CALLING":
		return FinishReasonToolCall
	case "content_filter", "SAFETY", "RECITATION":
		return FinishReasonSafety
	case "failed", "error", "cancelled", "canceled":
		return FinishReasonError
	case "completed":
		return FinishReasonStop
	case "incomplete":
		return FinishReasonLength
	case "":
		return ""
	default:
		return raw
	}
}

type StreamReader interface {
	Next() (*StreamChunk, error)
	Close() error
}

// Provider interface
type Provider interface {
	Name() string
	Validate() error
	Chat(ctx context.Context, req *Request) (*Response, error)
	Stream(ctx context.Context, req *Request) (StreamReader, error)
}

// ModelLister is an optional interface implemented by providers that support listing models.
type ModelLister interface {
	ListModels(ctx context.Context) ([]ModelInfo, error)
}
