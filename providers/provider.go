package providers

import (
	"context"
)

type ModelInfo struct {
	ID              string            `json:"id"`
	Provider        string            `json:"provider"`
	Name            string            `json:"name"`
	ContextWindow   int               `json:"context_window,omitempty"`
	MaxOutputTokens int               `json:"max_output_tokens,omitempty"`
	Capabilities    []ModelCapability `json:"capabilities"`
}

type Message struct {
	Role         string        `json:"role"`
	Content      string        `json:"content"`
	Contents     []MessageContent `json:"contents,omitempty"`
	ToolCalls    []ToolCall    `json:"tool_calls,omitempty"`
	ToolCallID   string        `json:"tool_call_id,omitempty"`
	CacheControl *CacheControl `json:"cache_control,omitempty"`
}

type MessageContent struct {
	Type        string                 `json:"type"`
	Text        string                 `json:"text,omitempty"`
	ImageURL    *MessageImageURL       `json:"image_url,omitempty"`
	Annotations []map[string]any       `json:"annotations,omitempty"`
	Logprobs    []map[string]any       `json:"logprobs,omitempty"`
}

type MessageImageURL struct {
	URL string `json:"url"`
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

type Request struct {
	Model             string           `json:"model"`
	Messages          []Message        `json:"messages"`
	MaxTokens         *int             `json:"max_tokens,omitempty"`
	Temperature       *float64         `json:"temperature,omitempty"`
	TopP              *float64         `json:"top_p,omitempty"`
	TopLogProbs       *int             `json:"top_logprobs,omitempty"`
	Stream            bool             `json:"stream,omitempty"`
	Tools             []Tool           `json:"tools,omitempty"`
	ToolChoice        any              `json:"tool_choice,omitempty"`
	ResponseFormat    *ResponseFormat  `json:"response_format,omitempty"`
	Stop              []string         `json:"stop,omitempty"`
	ReasoningEffort   string           `json:"reasoning_effort,omitempty"`
	ReasoningSummary  string           `json:"reasoning_summary,omitempty"`
	UseResponsesAPI   bool             `json:"use_responses_api,omitempty"`
	ServiceTier       string           `json:"service_tier,omitempty"`
	Store             *bool            `json:"store,omitempty"`
	ParallelToolCalls *bool            `json:"parallel_tool_calls,omitempty"`
	SafetyIdentifier  string           `json:"safety_identifier,omitempty"`

	ResponsesParams *ResponsesParams

	// Provider-specific extensions
	Extra map[string]any `json:"extra,omitempty"`
}

// ResponsesParams describes Response-API-specific or more granular controls
type ResponsesParams struct {
	Instructions         string            `json:"instructions,omitempty"`
	Conversation         string            `json:"conversation,omitempty"`
	PreviousResponseID   string            `json:"previous_response_id,omitempty"`
	Metadata             map[string]string `json:"metadata,omitempty"`
	Store                *bool             `json:"store,omitempty"`
	MaxOutputTokens      *int              `json:"max_output_tokens,omitempty"`
	MaxInputTokens       *int              `json:"max_input_tokens,omitempty"`
	MaxToolCalls         *int              `json:"max_tool_calls,omitempty"`
	ParallelToolCalls    *bool             `json:"parallel_tool_calls,omitempty"`
	Include              []string          `json:"include,omitempty"`
	SafetyIdentifier     string            `json:"safety_identifier,omitempty"`
	ServiceTier          string            `json:"service_tier,omitempty"`
	Temperature          *float64          `json:"temperature,omitempty"`
	TopP                 *float64          `json:"top_p,omitempty"`
	ToolChoice           any               `json:"tool_choice,omitempty"`
	ResponseFormat       *ResponseFormat   `json:"response_format,omitempty"`
	PromptCacheKey       string            `json:"prompt_cache_key,omitempty"`
	PromptCacheRetention string            `json:"prompt_cache_retention,omitempty"`
	Background           *bool             `json:"background,omitempty"`
	Prompt               map[string]any    `json:"prompt,omitempty"`
	ModelOverride        string            `json:"model_override,omitempty"`
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
}

type StreamReader interface {
	Next() (*StreamChunk, error)
	Close() error
}

// Provider interface
type Provider interface {
	Name() string
	Validate() error
	SupportsModel(model string) bool
	Models() []ModelInfo
	Chat(ctx context.Context, req *Request) (*Response, error)
	Stream(ctx context.Context, req *Request) (StreamReader, error)
}

// ModelCapability describes model capabilities (string alias for easy extension).
type ModelCapability = string

// Common capability constants.
const (
	CapabilityChat         ModelCapability = "chat"
	CapabilityFunctionCall ModelCapability = "function_call"
	CapabilityVision       ModelCapability = "vision"
	CapabilityReasoning    ModelCapability = "reasoning"
	CapabilityCode         ModelCapability = "code"
)
