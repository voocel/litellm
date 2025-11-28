package litellm

import (
	"context"
	"time"
)

// Message represents a conversation message
type Message struct {
	Role         string        `json:"role"` // user, assistant, system, tool
	Content      string        `json:"content"`
	ToolCallID   string        `json:"tool_call_id,omitempty"`
	ToolCalls    []ToolCall    `json:"tool_calls,omitempty"`
	CacheControl *CacheControl `json:"cache_control,omitempty"`
}

// Request represents a completion request
type Request struct {
	Model       string    `json:"model"`
	Messages    []Message `json:"messages"`
	MaxTokens   *int      `json:"max_tokens,omitempty"`
	Temperature *float64  `json:"temperature,omitempty"`
	Stream      bool      `json:"stream,omitempty"`
	Tools       []Tool    `json:"tools,omitempty"`
	ToolChoice  any       `json:"tool_choice,omitempty"`

	// Response format for structured output
	ResponseFormat *ResponseFormat `json:"response_format,omitempty"`

	// Stop sequences - custom text sequences that will cause the model to stop generating
	// OpenAI: uses "stop" parameter (string or array, up to 4 sequences)
	// Anthropic: uses "stop_sequences" parameter (array of strings)
	Stop []string `json:"stop,omitempty"`

	// Reasoning parameters for advanced models
	ReasoningEffort  string `json:"reasoning_effort,omitempty"`
	ReasoningSummary string `json:"reasoning_summary,omitempty"`
	UseResponsesAPI  bool   `json:"use_responses_api,omitempty"`

	// Prompt caching control
	CacheControl *CacheControl `json:"cache_control,omitempty"`

	// Provider-specific extensions
	Extra map[string]any `json:"extra,omitempty"`
}

// Response represents a completion response
type Response struct {
	Content      string         `json:"content"`
	ToolCalls    []ToolCall     `json:"tool_calls,omitempty"`
	Usage        Usage          `json:"usage"`
	Model        string         `json:"model"`
	Provider     string         `json:"provider"`
	FinishReason string         `json:"finish_reason,omitempty"`
	Reasoning    *ReasoningData `json:"reasoning,omitempty"`
}

// Usage represents token usage statistics
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
	ReasoningTokens  int `json:"reasoning_tokens,omitempty"`

	// Cache-related token statistics
	CacheCreationInputTokens int `json:"cache_creation_input_tokens,omitempty"` // Tokens written to cache
	CacheReadInputTokens     int `json:"cache_read_input_tokens,omitempty"`     // Tokens read from cache
}

// Tool represents a function tool definition
type Tool struct {
	Type     string      `json:"type"` // "function"
	Function FunctionDef `json:"function"`
}

// FunctionDef represents a function definition
type FunctionDef struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Parameters  any    `json:"parameters"`
}

// ToolCall represents a function call from the model
type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"` // "function"
	Function FunctionCall `json:"function"`
}

// FunctionCall represents the function call details
type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"` // JSON string
}

// ResponseFormat defines the response output format
type ResponseFormat struct {
	Type       string      `json:"type"` // "text", "json_object", "json_schema"
	JSONSchema *JSONSchema `json:"json_schema,omitempty"`
}

// JSONSchema defines structured JSON output schema
type JSONSchema struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Schema      any    `json:"schema"`
	Strict      *bool  `json:"strict,omitempty"`
}

// ReasoningData contains reasoning information for advanced models
type ReasoningData struct {
	Content    string `json:"content,omitempty"` // Full reasoning content
	Summary    string `json:"summary,omitempty"` // Reasoning summary
	TokensUsed int    `json:"tokens_used,omitempty"`
}

// CacheControl defines prompt caching behavior
type CacheControl struct {
	Type string `json:"type"`          // "ephemeral" or "persistent"
	TTL  *int   `json:"ttl,omitempty"` // Time to live in seconds (optional)
}

// Cache control types
const (
	CacheTypeEphemeral  = "ephemeral"
	CacheTypePersistent = "persistent"
)

// NewEphemeralCache creates an ephemeral cache control (default 5 minutes)
func NewEphemeralCache() *CacheControl {
	return &CacheControl{Type: CacheTypeEphemeral}
}

// NewPersistentCache creates a persistent cache control with custom TTL
func NewPersistentCache(ttlSeconds int) *CacheControl {
	return &CacheControl{
		Type: CacheTypePersistent,
		TTL:  &ttlSeconds,
	}
}

// NewCacheControl creates a cache control with specified type and optional TTL
func NewCacheControl(cacheType string, ttlSeconds ...int) *CacheControl {
	cache := &CacheControl{Type: cacheType}
	if len(ttlSeconds) > 0 {
		cache.TTL = &ttlSeconds[0]
	}
	return cache
}

// StreamChunk represents a single chunk in streaming response
type StreamChunk struct {
	Type          string          `json:"type"`
	Content       string          `json:"content,omitempty"`
	ToolCallDelta *ToolCallDelta  `json:"tool_call_delta,omitempty"`
	Reasoning     *ReasoningChunk `json:"reasoning,omitempty"`
	FinishReason  string          `json:"finish_reason,omitempty"`
	Done          bool            `json:"done"`
	Provider      string          `json:"provider"`
	Model         string          `json:"model,omitempty"`
	Usage         *Usage          `json:"usage,omitempty"`
}

// ToolCallDelta represents incremental tool call data
type ToolCallDelta struct {
	Index          int    `json:"index"`
	ID             string `json:"id,omitempty"`
	Type           string `json:"type,omitempty"`
	FunctionName   string `json:"function_name,omitempty"`
	ArgumentsDelta string `json:"arguments_delta,omitempty"`
}

// ReasoningChunk represents incremental reasoning data
type ReasoningChunk struct {
	Content string `json:"content,omitempty"`
	Summary string `json:"summary,omitempty"`
}

// StreamReader provides a unified interface for reading streaming responses
//
// Thread Safety: StreamReader is NOT thread-safe. Do not call Next() or Close()
// concurrently from multiple goroutines. Each StreamReader instance should be
// used by a single goroutine at a time.
//
// IMPORTANT: Always call Close() to prevent resource leaks. Use defer immediately
// after creating the stream:
//
//	stream, err := client.Stream(ctx, req)
//	if err != nil {
//	    return err
//	}
//	defer stream.Close()  // Must call to release resources
type StreamReader interface {
	// Next returns the next chunk or io.EOF when done
	Next() (*StreamChunk, error)
	// Close closes the stream
	Close() error
}

// ModelInfo contains information about a model
type ModelInfo struct {
	ID           string            `json:"id"`
	Name         string            `json:"name"`
	Provider     string            `json:"provider"`
	MaxTokens    int               `json:"max_tokens"`
	Capabilities []ModelCapability `json:"capabilities"`
}

// ModelCapability represents what a model can do
type ModelCapability string

const (
	CapabilityChat         ModelCapability = "chat"
	CapabilityFunctionCall ModelCapability = "function_call"
	CapabilityVision       ModelCapability = "vision"
	CapabilityReasoning    ModelCapability = "reasoning"
	CapabilityCode         ModelCapability = "code"
)

// Chunk types for streaming
const (
	ChunkTypeContent       = "content"
	ChunkTypeToolCallDelta = "tool_call_delta"
	ChunkTypeReasoning     = "reasoning"
)

// Response format types
const (
	ResponseFormatText       = "text"
	ResponseFormatJSONObject = "json_object"
	ResponseFormatJSONSchema = "json_schema"
)

// Config holds client configuration
type Config struct {
	MaxTokens   int            `json:"max_tokens"`
	Temperature float64        `json:"temperature"`
	Timeout     time.Duration  `json:"timeout"`
	Retries     int            `json:"retries"`
	Extra       map[string]any `json:"extra,omitempty"`
}

// Option is a functional option for configuring the client
type Option func(*Client)

// ProviderConfig contains provider-specific configuration
type ProviderConfig struct {
	APIKey  string `json:"api_key"`
	BaseURL string `json:"base_url,omitempty"`

	// Resilience configuration integrated directly
	Resilience ResilienceConfig `json:"resilience,omitempty"`

	// Provider-specific extras
	Extra map[string]any `json:"extra,omitempty"`
}

// Context keys for request metadata
type contextKey string

const (
	ContextKeyRequestID  contextKey = "request_id"
	ContextKeyRetryCount contextKey = "retry_count"
	ContextKeyProvider   contextKey = "provider"
)

// ChatProvider defines the basic chat completion capability
type ChatProvider interface {
	Chat(ctx context.Context, req *Request) (*Response, error)
}

// StreamProvider defines streaming capability
type StreamProvider interface {
	Stream(ctx context.Context, req *Request) (StreamReader, error)
}

// ModelProvider defines model information capability
type ModelProvider interface {
	Models() []ModelInfo
	SupportsModel(model string) bool
}

// Provider combines all capabilities through interface composition
// Implementations can choose which interfaces to support
type Provider interface {
	ChatProvider
	StreamProvider
	ModelProvider

	// Basic provider info
	Name() string
	Validate() error
}

// ProviderFactory is a function that creates a provider instance
type ProviderFactory func(config ProviderConfig) Provider

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

// Response format helpers

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
