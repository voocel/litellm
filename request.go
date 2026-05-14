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
	OpenAIResponsesTool    = providers.OpenAIResponsesTool
	ThinkingConfig         = providers.ThinkingConfig

	Request  = providers.Request
	Response = providers.Response
	Usage    = providers.Usage

	StreamChunk  = providers.StreamChunk
	StreamReader = providers.StreamReader

	Provider       = providers.Provider
	ProviderConfig = providers.ProviderConfig
	ModelInfo      = providers.ModelInfo
	ModelLister    = providers.ModelLister
)

// ProviderFactory is used to register custom providers.
type ProviderFactory func(config ProviderConfig) Provider

// ProviderDescriptor describes a custom provider registration.
// Keep this intentionally small: only static metadata needed by the client
// runtime should live here.
type ProviderDescriptor struct {
	Name       string
	DefaultURL string
	Factory    ProviderFactory
}

// CacheControl type constants. Anthropic Messages API only defines "ephemeral";
// retention is controlled via the optional ttl field ("5m" / "1h").
const (
	CacheTypeEphemeral = "ephemeral"
)

// Cache TTL constants for Anthropic ephemeral cache_control.
const (
	CacheTTL5m = "5m"
	CacheTTL1h = "1h"
)

// Stream chunk type constants.
const (
	ChunkTypeContent       = "content"
	ChunkTypeToolCallDelta = "tool_call_delta"
	ChunkTypeReasoning     = "reasoning"
)

// Lifecycle event types — bracket start/end of each content block during streaming.
const (
	ChunkTypeContentStart   = "content_start"
	ChunkTypeContentEnd     = "content_end"
	ChunkTypeReasoningStart = "reasoning_start"
	ChunkTypeReasoningEnd   = "reasoning_end"
	ChunkTypeToolCallStart  = "tool_call_start"
	ChunkTypeToolCallEnd    = "tool_call_end"
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

// FinishReason constants — canonical values returned by all providers.
const (
	FinishReasonStop     = providers.FinishReasonStop
	FinishReasonLength   = providers.FinishReasonLength
	FinishReasonToolCall = providers.FinishReasonToolCall
	FinishReasonError    = providers.FinishReasonError
	FinishReasonSafety   = providers.FinishReasonSafety
)

// NormalizeFinishReason maps provider-specific stop reasons to canonical constants.
func NormalizeFinishReason(raw string) string {
	return providers.NormalizeFinishReason(raw)
}

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

// WithThinking enables thinking/reasoning mode.
//
//	WithThinking()              // enable with provider defaults
//	WithThinking("high")        // set reasoning effort level (low/medium/high)
//	WithThinking(8192)          // set token budget
//	WithThinking("high", 8192)  // both level and budget
func WithThinking(opts ...any) RequestOption {
	return func(r *Request) {
		cfg := &ThinkingConfig{Type: ThinkingEnabled}
		for _, o := range opts {
			switch v := o.(type) {
			case string:
				cfg.Level = v
			case int:
				if v > 0 {
					cfg.BudgetTokens = &v
				}
			}
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

// WithOnPayload sets a debug hook that receives the serialized JSON body
// before each HTTP request is sent to the provider.
func WithOnPayload(fn func(provider string, payload []byte)) RequestOption {
	return func(r *Request) {
		r.OnPayload = fn
	}
}

// WithOnWarning sets a hook for portability warnings, such as an unsupported
// option being omitted for a provider instead of failing the request.
func WithOnWarning(fn func(provider string, message string)) RequestOption {
	return func(r *Request) {
		r.OnWarning = fn
	}
}

// WithCacheRetention enables automatic prompt-caching placement.
//
// Default behavior (option unset): no automatic caching — callers are
// expected to attach cache_control to specific messages themselves. This is
// opt-in by design so application frameworks can control exactly which
// segments use the scarce per-request marker budget (Anthropic caps at 4).
//
// Accepted values (case-insensitive, applied per-provider):
//   - "" / "none"        : disable auto-caching (default)
//   - "short" / "5m"     : ephemeral cache, ~5-minute TTL (provider default)
//   - "long" / "1h"      : ephemeral cache, 1-hour TTL where supported
//
// Provider behavior when enabled:
//   - Anthropic / OpenRouter→Anthropic: places cache_control on each system
//     content block that doesn't already carry one, and on the last user
//     message's last content block. User-provided CacheControl on individual
//     messages always takes precedence.
//   - Bedrock (Anthropic models, Converse API): appends cachePoint markers
//     after the last system block, last user message content block, and
//     tools section.
//   - OpenAI / Azure OpenAI: prefer prompt_cache_retention via Extra
//     ("in_memory" or "24h"); cache_control breakpoints are not applicable.
//
// Other providers ignore this option.
func WithCacheRetention(retention string) RequestOption {
	return WithExtra("cache_retention", retention)
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

// ToolRefContent creates a tool_reference content item for tool search results.
func ToolRefContent(toolName string) MessageContent {
	return MessageContent{Type: "tool_reference", ToolName: toolName}
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

// NewEphemeralCache creates an ephemeral cache_control breakpoint with the
// provider default TTL (~5 minutes for Anthropic).
func NewEphemeralCache() *CacheControl {
	return &CacheControl{Type: CacheTypeEphemeral}
}

// NewEphemeralCacheTTL creates an ephemeral cache_control breakpoint with an
// explicit TTL. Per Anthropic Messages API, accepted values are "5m" (default)
// and "1h"; other values are passed through as-is and may be rejected by the
// provider.
func NewEphemeralCacheTTL(ttl string) *CacheControl {
	return &CacheControl{Type: CacheTypeEphemeral, TTL: ttl}
}

// NewThinkingEnabled enables thinking with an optional budget.
func NewThinkingEnabled(budgetTokens int) *ThinkingConfig {
	cfg := &ThinkingConfig{Type: ThinkingEnabled}
	if budgetTokens > 0 {
		cfg.BudgetTokens = &budgetTokens
	}
	return cfg
}

// NewThinkingWithLevel enables thinking with a reasoning level.
// Level is provider-specific: "low", "medium", "high" etc.
func NewThinkingWithLevel(level string) *ThinkingConfig {
	return &ThinkingConfig{Type: ThinkingEnabled, Level: level}
}

// NewThinkingDisabled disables thinking output.
func NewThinkingDisabled() *ThinkingConfig {
	return &ThinkingConfig{Type: ThinkingDisabled}
}
