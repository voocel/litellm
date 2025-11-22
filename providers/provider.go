package providers

import (
	"context"
)

type ModelInfo struct {
	ID           string
	Provider     string
	Name         string
	MaxTokens    int
	Capabilities []string
}

type Message struct {
	Role         string
	Content      string
	Contents     []MessageContent
	ToolCalls    []ToolCall
	ToolCallID   string
	CacheControl *CacheControl
}

type MessageContent struct {
	Type        string
	Text        string
	ImageURL    *MessageImageURL
	Annotations []map[string]interface{}
	Logprobs    []map[string]interface{}
}

type MessageImageURL struct {
	URL string
}

// CacheControl defines prompt caching behavior for providers
type CacheControl struct {
	Type string // "ephemeral" or "persistent"
	TTL  *int   // Time to live in seconds (optional)
}

type ToolCall struct {
	ID       string
	Type     string
	Function FunctionCall
}

type FunctionCall struct {
	Name      string
	Arguments string
}

type Tool struct {
	Type     string
	Function FunctionDef
}

type FunctionDef struct {
	Name        string
	Description string
	Parameters  any
}

type ResponseFormat struct {
	Type       string
	JSONSchema *JSONSchema
}

type JSONSchema struct {
	Name        string
	Description string
	Schema      any
	Strict      *bool
}

type Request struct {
	Model             string
	Messages          []Message
	MaxTokens         *int
	Temperature       *float64
	TopP              *float64
	TopLogProbs       *int
	Stream            bool
	Tools             []Tool
	ToolChoice        any
	ResponseFormat    *ResponseFormat
	Stop              []string
	ReasoningEffort   string
	ReasoningSummary  string
	UseResponsesAPI   bool
	ServiceTier       string
	Store             *bool
	ParallelToolCalls *bool
	SafetyIdentifier  string

	ResponsesParams *ResponsesParams

	// Provider-specific extensions
	Extra map[string]interface{}
}

// ResponsesParams describes Response-API-specific or more granular controls
type ResponsesParams struct {
	Instructions         string
	Conversation         string
	PreviousResponseID   string
	Metadata             map[string]string
	Store                *bool
	MaxOutputTokens      *int
	MaxInputTokens       *int
	MaxToolCalls         *int
	ParallelToolCalls    *bool
	Include              []string
	SafetyIdentifier     string
	ServiceTier          string
	Temperature          *float64
	TopP                 *float64
	ToolChoice           any
	ResponseFormat       *ResponseFormat
	PromptCacheKey       string
	PromptCacheRetention string
	Background           *bool
	Prompt               map[string]interface{}
	ModelOverride        string
}

type Usage struct {
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
	ReasoningTokens  int

	// Cache-related token statistics
	CacheCreationInputTokens int // Tokens written to cache
	CacheReadInputTokens     int // Tokens read from cache
}

type ReasoningData struct {
	Summary    string
	Content    string
	TokensUsed int
}

type Response struct {
	Content      string
	Contents     []MessageContent
	ToolCalls    []ToolCall
	Usage        Usage
	Model        string
	Provider     string
	FinishReason string
	Reasoning    *ReasoningData
}

type StreamChunk struct {
	Type          string
	Content       string
	ContentIndex  *int
	OutputIndex   *int
	ItemID        string
	ToolCallDelta *ToolCallDelta
	FinishReason  string
	Model         string
	Provider      string
	Done          bool
	Reasoning     *ReasoningChunk
	Usage         *Usage
}

type ToolCallDelta struct {
	Index          int
	ID             string
	Type           string
	FunctionName   string
	ArgumentsDelta string
	OutputIndex    *int
	ItemID         string
}

type ReasoningChunk struct {
	Summary string
	Content string
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
