package litellm

import (
	"context"
)

// Message represents a conversation message
type Message struct {
	Role       string     `json:"role"`                   // "user", "assistant", "system", "tool"
	Content    string     `json:"content"`                // Message content
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`   // Tool calls made by assistant
	ToolCallID string     `json:"tool_call_id,omitempty"` // ID for tool response
}

// ToolCall represents a function call request
type ToolCall struct {
	ID       string       `json:"id"`       // Unique identifier
	Type     string       `json:"type"`     // Always "function" for now
	Function FunctionCall `json:"function"` // Function details
}

// FunctionCall represents the function to be called
type FunctionCall struct {
	Name      string `json:"name"`      // Function name
	Arguments string `json:"arguments"` // JSON string of arguments
}

// Tool represents a function that can be called
type Tool struct {
	Type     string         `json:"type"`     // Always "function" for now
	Function FunctionSchema `json:"function"` // Function schema
}

// FunctionSchema defines a callable function
type FunctionSchema struct {
	Name        string `json:"name"`        // Function name
	Description string `json:"description"` // Function description
	Parameters  any    `json:"parameters"`  // JSON schema for parameters
}

// Request represents a completion request
type Request struct {
	Model       string    `json:"model"`                 // Model identifier in "provider/model" format
	Messages    []Message `json:"messages"`              // Conversation messages
	MaxTokens   *int      `json:"max_tokens,omitempty"`  // Maximum tokens to generate
	Temperature *float64  `json:"temperature,omitempty"` // Sampling temperature
	Stream      bool      `json:"stream,omitempty"`      // Enable streaming
	Tools       []Tool    `json:"tools,omitempty"`       // Available tools
	ToolChoice  any       `json:"tool_choice,omitempty"` // Tool choice strategy

	// Reasoning model parameters (OpenAI o-series)
	ReasoningEffort  string `json:"reasoning_effort,omitempty"`  // "low", "medium", "high"
	ReasoningSummary string `json:"reasoning_summary,omitempty"` // "concise", "detailed", "auto"
	UseResponsesAPI  bool   `json:"use_responses_api,omitempty"` // Force Responses API usage

	// Extension fields for provider-specific features
	Extra map[string]any `json:"extra,omitempty"` // Provider-specific parameters
}

// Response represents a completion response
type Response struct {
	Content   string     `json:"content"`              // Generated content
	ToolCalls []ToolCall `json:"tool_calls,omitempty"` // Tool calls requested
	Usage     Usage      `json:"usage"`                // Token usage statistics
	Model     string     `json:"model"`                // Actual model used
	Provider  string     `json:"provider"`             // Provider name

	// Reasoning data (for reasoning models)
	Reasoning *ReasoningData `json:"reasoning,omitempty"` // Reasoning process data

	// Metadata
	FinishReason string         `json:"finish_reason,omitempty"` // Why generation stopped
	Extra        map[string]any `json:"extra,omitempty"`         // Provider-specific data
}

// ReasoningData contains reasoning process information
type ReasoningData struct {
	Content    string `json:"content,omitempty"`     // Reasoning process content
	Summary    string `json:"summary,omitempty"`     // Reasoning summary
	TokensUsed int    `json:"tokens_used,omitempty"` // Tokens used for reasoning
}

// Usage represents token usage statistics
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`              // Input tokens
	CompletionTokens int `json:"completion_tokens"`          // Output tokens
	TotalTokens      int `json:"total_tokens"`               // Total tokens
	ReasoningTokens  int `json:"reasoning_tokens,omitempty"` // Reasoning tokens (o-series)
}

// StreamChunk represents a streaming response chunk
type StreamChunk struct {
	Type      ChunkType  `json:"type"`                 // Chunk type
	Content   string     `json:"content,omitempty"`    // Content delta
	ToolCalls []ToolCall `json:"tool_calls,omitempty"` // Complete tool calls
	Usage     *Usage     `json:"usage,omitempty"`      // Final usage stats
	Done      bool       `json:"done"`                 // Stream completion flag
	Error     error      `json:"error,omitempty"`      // Error if any

	// Reasoning data
	Reasoning *ReasoningChunk `json:"reasoning,omitempty"` // Reasoning chunk

	// Tool call delta data
	ToolCallDelta *ToolCallDelta `json:"tool_call_delta,omitempty"` // Tool call incremental data

	// Metadata
	Model        string         `json:"model,omitempty"`         // Model name
	Provider     string         `json:"provider,omitempty"`      // Provider name
	FinishReason string         `json:"finish_reason,omitempty"` // Completion reason
	Extra        map[string]any `json:"extra,omitempty"`         // Provider-specific data
}

// ChunkType defines the type of streaming chunk
type ChunkType string

const (
	ChunkTypeContent       ChunkType = "content"         // Regular content
	ChunkTypeReasoning     ChunkType = "reasoning"       // Reasoning content
	ChunkTypeToolCall      ChunkType = "tool_call"       // Complete tool call
	ChunkTypeToolCallDelta ChunkType = "tool_call_delta" // Tool call arguments delta
	ChunkTypeUsage         ChunkType = "usage"           // Usage statistics
	ChunkTypeDone          ChunkType = "done"            // Completion marker
	ChunkTypeError         ChunkType = "error"           // Error
)

// ReasoningChunk represents a reasoning content chunk
type ReasoningChunk struct {
	Content string `json:"content,omitempty"` // Reasoning content delta
	Summary string `json:"summary,omitempty"` // Reasoning summary delta
}

// ToolCallDelta represents incremental tool call data
type ToolCallDelta struct {
	Index          int    `json:"index,omitempty"`           // Tool call index in the array
	ID             string `json:"id,omitempty"`              // Tool call ID
	Type           string `json:"type,omitempty"`            // Tool type (e.g., "function")
	FunctionName   string `json:"function_name,omitempty"`   // Function name
	ArgumentsDelta string `json:"arguments_delta,omitempty"` // Incremental arguments
}

// StreamReader provides an interface for reading streaming responses
type StreamReader interface {
	Read() (*StreamChunk, error) // Read next chunk
	Close() error                // Close the stream
	Err() error                  // Get any error that occurred
}

// ModelInfo represents information about a model
type ModelInfo struct {
	ID           string            `json:"id"`                     // Model identifier
	Provider     string            `json:"provider"`               // Provider name
	Name         string            `json:"name"`                   // Display name
	Description  string            `json:"description,omitempty"`  // Model description
	MaxTokens    int               `json:"max_tokens,omitempty"`   // Maximum context tokens
	Capabilities []ModelCapability `json:"capabilities,omitempty"` // Model capabilities
	Extra        map[string]any    `json:"extra,omitempty"`        // Provider-specific info
}

// ModelCapability represents what a model can do
type ModelCapability string

const (
	CapabilityChat         ModelCapability = "chat"          // Chat completion
	CapabilityCompletion   ModelCapability = "completion"    // Text completion
	CapabilityEmbedding    ModelCapability = "embedding"     // Text embedding
	CapabilityFunctionCall ModelCapability = "function_call" // Function calling
	CapabilityVision       ModelCapability = "vision"        // Image understanding
	CapabilityReasoning    ModelCapability = "reasoning"     // Step-by-step reasoning
	CapabilityCode         ModelCapability = "code"          // Code generation
	CapabilityMultimodal   ModelCapability = "multimodal"    // Multiple input types
)

// Provider defines the interface that all LLM providers must implement
type Provider interface {
	// Name returns the provider name
	Name() string

	// Complete performs a completion request
	Complete(ctx context.Context, req *Request) (*Response, error)

	// Stream performs a streaming completion request
	Stream(ctx context.Context, req *Request) (StreamReader, error)

	// Models returns the list of supported models
	Models() []ModelInfo

	// Validate checks if the provider is properly configured
	Validate() error
}

// ProviderConfig holds configuration for a provider
type ProviderConfig struct {
	APIKey  string         `json:"api_key"`            // API key
	BaseURL string         `json:"base_url,omitempty"` // Custom base URL
	Extra   map[string]any `json:"extra,omitempty"`    // Provider-specific config
}
