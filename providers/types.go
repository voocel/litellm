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
	Role       string
	Content    string
	ToolCalls  []ToolCall
	ToolCallID string
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
	Model            string
	Messages         []Message
	MaxTokens        *int
	Temperature      *float64
	Stream           bool
	Tools            []Tool
	ToolChoice       any
	ResponseFormat   *ResponseFormat
	ReasoningEffort  string
	ReasoningSummary string
	UseResponsesAPI  bool

	// Provider-specific extensions
	Extra map[string]interface{}
}

type Usage struct {
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
	ReasoningTokens  int
}

type ReasoningData struct {
	Summary    string
	Content    string
	TokensUsed int
}

type Response struct {
	Content      string
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
	ToolCallDelta *ToolCallDelta
	FinishReason  string
	Model         string
	Provider      string
	Done          bool
	Reasoning     *ReasoningChunk
}

type ToolCallDelta struct {
	Index          int
	ID             string
	Type           string
	FunctionName   string
	ArgumentsDelta string
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
