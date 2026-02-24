package providers

// ---------------------------------------------------------------------------
// Response-side types — completions, streaming chunks, and model metadata
// ---------------------------------------------------------------------------

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

	// Extra captures provider-specific response fields not covered by the
	// standard schema (e.g. logprobs, annotations, system_fingerprint).
	Extra map[string]any `json:"extra,omitempty"`
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
// FinishReason — canonical values and provider-specific normalization
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
	case "content_filter", "SAFETY", "RECITATION", "sensitive":
		return FinishReasonSafety
	case "failed", "error", "cancelled", "canceled", "insufficient_system_resource", "network_error":
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
