package litellm

import "github.com/voocel/litellm/providers"

// Core types are sourced from providers; litellm re-exports them.
type (
	Message         = providers.Message
	MessageContent  = providers.MessageContent
	MessageImageURL = providers.MessageImageURL
	CacheControl    = providers.CacheControl

	Tool          = providers.Tool
	FunctionDef   = providers.FunctionDef
	ToolCall      = providers.ToolCall
	FunctionCall  = providers.FunctionCall
	ToolCallDelta = providers.ToolCallDelta

	ResponseFormat         = providers.ResponseFormat
	JSONSchema             = providers.JSONSchema
	OpenAIResponsesRequest = providers.OpenAIResponsesRequest
	ThinkingConfig         = providers.ThinkingConfig

	Request       = providers.Request
	Response      = providers.Response
	Usage         = providers.Usage
	ReasoningData = providers.ReasoningData

	StreamChunk    = providers.StreamChunk
	ReasoningChunk = providers.ReasoningChunk
	StreamReader   = providers.StreamReader

	Provider       = providers.Provider
	ProviderConfig = providers.ProviderConfig
)

// ProviderFactory is used to register custom providers.
type ProviderFactory func(config ProviderConfig) Provider

// CacheControl type constants.
const (
	CacheTypeEphemeral  = "ephemeral"
	CacheTypePersistent = "persistent"
)

// Stream chunk type constants.
const (
	ChunkTypeContent       = "content"
	ChunkTypeToolCallDelta = "tool_call_delta"
	ChunkTypeReasoning     = "reasoning"
)

// ResponseFormat type constants.
const (
	ResponseFormatText       = "text"
	ResponseFormatJSONObject = "json_object"
	ResponseFormatJSONSchema = "json_schema"
)

// Thinking type constants.
const (
	ThinkingEnabled  = "enabled"
	ThinkingDisabled = "disabled"
)

// RequestOption configures a Request using the functional options pattern.
type RequestOption func(*Request)

// NewRequest creates a new Request with the given model and user prompt.
// Additional options can be passed to configure the request.
//
// Example:
//
//	req := litellm.NewRequest("gpt-4", "Hello",
//	    litellm.WithSystemPrompt("You are helpful"),
//	    litellm.WithMaxTokens(1024),
//	    litellm.WithTemperature(0.7),
//	)
func NewRequest(model, prompt string, opts ...RequestOption) *Request {
	req := &Request{
		Model: model,
		Messages: []Message{
			{Role: "user", Content: prompt},
		},
	}
	for _, opt := range opts {
		opt(req)
	}
	return req
}

// NewRequestWithMessages creates a new Request with the given model and messages.
// Use this when you need full control over the message history.
//
// Example:
//
//	req := litellm.NewRequestWithMessages("gpt-4",
//	    []litellm.Message{
//	        litellm.SystemMessage("You are helpful"),
//	        litellm.UserMessage("Hello"),
//	    },
//	    litellm.WithMaxTokens(1024),
//	)
func NewRequestWithMessages(model string, messages []Message, opts ...RequestOption) *Request {
	req := &Request{
		Model:    model,
		Messages: messages,
	}
	for _, opt := range opts {
		opt(req)
	}
	return req
}

// WithSystemPrompt prepends a system message to the request.
// If a system message already exists, it will be replaced.
func WithSystemPrompt(content string) RequestOption {
	return func(r *Request) {
		if len(r.Messages) > 0 && r.Messages[0].Role == "system" {
			r.Messages[0].Content = content
			return
		}
		r.Messages = append([]Message{{Role: "system", Content: content}}, r.Messages...)
	}
}

// WithMaxTokens sets the maximum number of tokens to generate.
func WithMaxTokens(n int) RequestOption {
	return func(r *Request) {
		r.MaxTokens = &n
	}
}

// WithTemperature sets the sampling temperature (0.0 to 2.0).
// Higher values make output more random, lower values more deterministic.
func WithTemperature(t float64) RequestOption {
	return func(r *Request) {
		r.Temperature = &t
	}
}

// WithTopP sets the nucleus sampling parameter (0.0 to 1.0).
func WithTopP(p float64) RequestOption {
	return func(r *Request) {
		r.TopP = &p
	}
}

// WithTools adds tool definitions to the request.
func WithTools(tools ...Tool) RequestOption {
	return func(r *Request) {
		r.Tools = append(r.Tools, tools...)
	}
}

// WithToolChoice sets the tool choice behavior.
// Accepts: "auto", "none", "required", or a specific tool name.
func WithToolChoice(choice any) RequestOption {
	return func(r *Request) {
		r.ToolChoice = choice
	}
}

// WithJSONMode enables JSON object output mode.
// The model will return valid JSON without enforcing a specific schema.
func WithJSONMode() RequestOption {
	return func(r *Request) {
		r.ResponseFormat = &ResponseFormat{Type: ResponseFormatJSONObject}
	}
}

// WithJSONSchema enables structured JSON output with schema validation.
func WithJSONSchema(name, description string, schema any, strict bool) RequestOption {
	return func(r *Request) {
		r.ResponseFormat = &ResponseFormat{
			Type: ResponseFormatJSONSchema,
			JSONSchema: &JSONSchema{
				Name:        name,
				Description: description,
				Schema:      schema,
				Strict:      &strict,
			},
		}
	}
}

// WithResponseFormat sets a custom response format.
func WithResponseFormat(format *ResponseFormat) RequestOption {
	return func(r *Request) {
		r.ResponseFormat = format
	}
}

// WithStop sets the stop sequences.
func WithStop(sequences ...string) RequestOption {
	return func(r *Request) {
		r.Stop = sequences
	}
}

// WithThinking enables thinking/reasoning mode with an optional token budget.
// Set budget to 0 for provider default.
func WithThinking(budget int) RequestOption {
	return func(r *Request) {
		cfg := &ThinkingConfig{Type: ThinkingEnabled}
		if budget > 0 {
			cfg.BudgetTokens = &budget
		}
		r.Thinking = cfg
	}
}

// WithoutThinking explicitly disables thinking/reasoning mode.
func WithoutThinking() RequestOption {
	return func(r *Request) {
		r.Thinking = &ThinkingConfig{Type: ThinkingDisabled}
	}
}

// WithExtra sets a provider-specific parameter.
func WithExtra(key string, value any) RequestOption {
	return func(r *Request) {
		if r.Extra == nil {
			r.Extra = make(map[string]any)
		}
		r.Extra[key] = value
	}
}

// WithExtras sets multiple provider-specific parameters.
func WithExtras(extras map[string]any) RequestOption {
	return func(r *Request) {
		if r.Extra == nil {
			r.Extra = make(map[string]any)
		}
		for k, v := range extras {
			r.Extra[k] = v
		}
	}
}

// MultiContentMessage creates a message with multiple content items (text, images, etc).
//
// Example:
//
//	msg := litellm.MultiContentMessage("user",
//	    litellm.TextContent("What's in this image?"),
//	    litellm.ImageContent("https://example.com/image.png"),
//	)
func MultiContentMessage(role string, contents ...MessageContent) Message {
	return Message{Role: role, Contents: contents}
}

// TextContent creates a text content item for multi-content messages.
func TextContent(text string) MessageContent {
	return MessageContent{Type: "text", Text: text}
}

// ImageContent creates an image content item from a URL.
func ImageContent(url string) MessageContent {
	return MessageContent{Type: "image_url", ImageURL: &MessageImageURL{URL: url}}
}

// ImageContentWithDetail creates an image content item with detail level.
// Detail can be "auto", "low", or "high".
func ImageContentWithDetail(url, detail string) MessageContent {
	return MessageContent{
		Type:     "image_url",
		ImageURL: &MessageImageURL{URL: url, Detail: detail},
	}
}

// NewTool creates a function tool definition.
func NewTool(name, description string, parameters any) Tool {
	return Tool{
		Type: "function",
		Function: FunctionDef{
			Name:        name,
			Description: description,
			Parameters:  parameters,
		},
	}
}

// NewEphemeralCache creates an ephemeral cache control (TTL is provider-defined, typically ~5 minutes).
func NewEphemeralCache() *CacheControl {
	return &CacheControl{Type: CacheTypeEphemeral}
}

// NewPersistentCache creates a persistent cache control with a custom TTL (seconds).
func NewPersistentCache(ttlSeconds int) *CacheControl {
	return &CacheControl{
		Type: CacheTypePersistent,
		TTL:  &ttlSeconds,
	}
}

// NewThinkingEnabled enables thinking with an optional budget.
func NewThinkingEnabled(budgetTokens int) *ThinkingConfig {
	cfg := &ThinkingConfig{Type: ThinkingEnabled}
	if budgetTokens > 0 {
		cfg.BudgetTokens = &budgetTokens
	}
	return cfg
}

// NewThinkingDisabled disables thinking output.
func NewThinkingDisabled() *ThinkingConfig {
	return &ThinkingConfig{Type: ThinkingDisabled}
}
