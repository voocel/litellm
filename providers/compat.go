package providers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// ---------------------------------------------------------------------------
// Compat — captures all behavioral differences between OpenAI-compatible providers
// ---------------------------------------------------------------------------

// Compat configures how an OpenAI-compatible provider differs from the
// standard OpenAI Chat Completions API.  Zero values select the most common
// defaults so that a minimal config works for simple providers.
type Compat struct {
	// ProviderName is used in errors, Response.Provider, etc.
	ProviderName string

	// DefaultBaseURL is the fallback when ProviderConfig.BaseURL is empty.
	DefaultBaseURL string

	// EndpointPath is appended to BaseURL for chat completions.
	// Default: "/chat/completions"
	EndpointPath string

	// --- Headers ---

	// ExtraHeaders are sent with every request (e.g. OpenRouter: HTTP-Referer).
	ExtraHeaders map[string]string

	// StreamHeaders are sent only for stream requests.
	StreamHeaders map[string]string

	// --- Request body ---

	// MaxTokensField overrides the JSON key for max tokens.  Default: "max_tokens".
	MaxTokensField string

	// IncludeStreamUsage sends stream_options.include_usage when streaming.
	IncludeStreamUsage bool

	// MaxStopSequences limits stop sequences sent.  0 = unlimited.
	MaxStopSequences int

	// OmitStop suppresses stop sequences entirely.
	OmitStop bool

	// SupportsJSONSchema enables full json_schema in response_format.
	SupportsJSONSchema bool

	// JSONSchemaToPrompt injects the JSON schema into the last user message
	// instead of using response_format.  Used by GLM.
	JSONSchemaToPrompt bool

	// ThinkingMapper converts ThinkingConfig into provider-specific request
	// body fields.  Returns nil to skip.  If nil, uses default
	// {"thinking":{"type":"enabled/disabled"}} format.
	ThinkingMapper func(thinking *ThinkingConfig, model string) map[string]any

	// ResponseFormatMapper converts ResponseFormat to the provider-specific
	// value for the "response_format" key.  Return nil to omit.
	// If the function itself is nil, defaults to json_object-only support.
	ResponseFormatMapper func(rf *ResponseFormat) any

	// CustomMessageConverter replaces the default ConvertMessages.
	// Must return a JSON-serializable slice.
	CustomMessageConverter func(messages []Message) any

	// CustomToolConverter replaces the default ConvertTools.
	CustomToolConverter func(tools []Tool) any

	// CleanSchema recursively cleans JSON schemas for strict mode.
	CleanSchema func(schema any) any

	// --- Response parsing ---

	// ReasoningField is the preferred JSON key for reasoning in message/delta.
	// When set, it is probed first; remaining default fields are tried as fallback.
	// Default probe order: "reasoning_content", "reasoning", "reasoning_text".
	ReasoningField string

	// ReasoningCondition controls when reasoning is extracted.
	//   ""/"always"                    — whenever the field is non-empty
	//   "model_contains:<substring>"   — only when model name contains substring
	ReasoningCondition string

	// ContentAsInterface parses Message.Content as interface{} (string or array)
	// instead of plain string.  Used by OpenRouter.
	ContentAsInterface bool

	// ModelFromResponse takes the model name from response JSON.
	// When false, uses the request model.
	ModelFromResponse bool

	// --- Usage parsing ---

	// HasCompletionTokenDetails looks for completion_tokens_details.reasoning_tokens.
	HasCompletionTokenDetails bool

	// HasCacheTokens looks for prompt_cache_hit_tokens / prompt_cache_miss_tokens.
	HasCacheTokens bool

	// --- Stream parsing ---

	// DataPrefix is the SSE data line prefix.  Default: "data: ".
	DataPrefix string
}

// defaultReasoningFields lists all known reasoning field names in probe order.
var defaultReasoningFields = []string{"reasoning_content", "reasoning", "reasoning_text"}

// reasoningFields returns the ordered list of fields to probe for reasoning content.
// If ReasoningField is configured, it gets highest priority; remaining defaults follow.
func (c *Compat) reasoningFields() []string {
	if c.ReasoningField == "" {
		return defaultReasoningFields
	}
	fields := make([]string, 0, len(defaultReasoningFields)+1)
	fields = append(fields, c.ReasoningField)
	for _, f := range defaultReasoningFields {
		if f != c.ReasoningField {
			fields = append(fields, f)
		}
	}
	return fields
}

// findReasoning probes a message/delta map for the first non-empty reasoning field.
// Returns the value and the field name that matched ("" if none).
func (c *Compat) findReasoning(m map[string]any) (value, field string) {
	for _, f := range c.reasoningFields() {
		if v, ok := m[f].(string); ok && v != "" {
			return v, f
		}
	}
	return "", ""
}

// dataPrefix returns the configured SSE prefix or the default.
func (c *Compat) dataPrefix() string {
	if c.DataPrefix != "" {
		return c.DataPrefix
	}
	return "data: "
}

// endpointPath returns the configured endpoint or the default.
func (c *Compat) endpointPath() string {
	if c.EndpointPath != "" {
		return c.EndpointPath
	}
	return "/chat/completions"
}

// maxTokensField returns the configured field or the default.
func (c *Compat) maxTokensField() string {
	if c.MaxTokensField != "" {
		return c.MaxTokensField
	}
	return "max_tokens"
}

// shouldExtractReasoning decides if reasoning content should be extracted
// based on the provider's condition and the actual model name.
func (c *Compat) shouldExtractReasoning(model string) bool {
	cond := c.ReasoningCondition
	if cond == "" || cond == "always" {
		return true
	}
	if after, ok := strings.CutPrefix(cond, "model_contains:"); ok {
		return strings.Contains(strings.ToLower(model), strings.ToLower(after))
	}
	return true
}

// ---------------------------------------------------------------------------
// OpenAICompatProvider — generic provider for all OpenAI-compatible APIs
// ---------------------------------------------------------------------------

// OpenAICompatProvider implements Provider for any OpenAI-compatible API.
type OpenAICompatProvider struct {
	*BaseProvider
	compat Compat
}

// NewOpenAICompat creates a new OpenAI-compatible provider.
func NewOpenAICompat(config ProviderConfig, compat Compat) *OpenAICompatProvider {
	return &OpenAICompatProvider{
		BaseProvider: NewBaseProvider(compat.ProviderName, config),
		compat:       compat,
	}
}

// Chat sends a non-streaming chat completion request.
func (p *OpenAICompatProvider) Chat(ctx context.Context, req *Request) (*Response, error) {
	if err := p.validate(req); err != nil {
		return nil, err
	}

	body, err := p.buildRequestBody(req, false)
	if err != nil {
		return nil, err
	}

	p.NotifyPayload(req, body)

	httpReq, err := p.newHTTPRequest(ctx, body, req, false)
	if err != nil {
		return nil, err
	}

	resp, err := p.HTTPClient().Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return nil, NewHTTPError(p.compat.ProviderName, resp.StatusCode, string(b))
	}

	var raw compatResponse
	if err := json.NewDecoder(resp.Body).Decode(&raw); err != nil {
		return nil, fmt.Errorf("%s: decode response: %w", p.compat.ProviderName, err)
	}

	return p.convertResponse(&raw, req)
}

// Stream sends a streaming chat completion request.
func (p *OpenAICompatProvider) Stream(ctx context.Context, req *Request) (StreamReader, error) {
	if err := p.validate(req); err != nil {
		return nil, err
	}

	body, err := p.buildRequestBody(req, true)
	if err != nil {
		return nil, err
	}

	p.NotifyPayload(req, body)

	httpReq, err := p.newHTTPRequest(ctx, body, req, true)
	if err != nil {
		return nil, err
	}

	resp, err := p.HTTPClient().Do(httpReq)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, NewHTTPError(p.compat.ProviderName, resp.StatusCode, string(b))
	}

	return newCompatStreamReader(resp, req, &p.compat), nil
}

// ListModels returns available models from the provider.
func (p *OpenAICompatProvider) ListModels(ctx context.Context) ([]ModelInfo, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}

	url := p.modelsURL()
	httpReq, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("%s: create models request: %w", p.compat.ProviderName, err)
	}
	p.setHeaders(httpReq, nil, false)

	resp, err := p.HTTPClient().Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return nil, NewHTTPError(p.compat.ProviderName, resp.StatusCode, string(b))
	}

	var payload struct {
		Data []struct {
			ID            string `json:"id"`
			Name          string `json:"name,omitempty"`
			Description   string `json:"description,omitempty"`
			Created       int64  `json:"created,omitempty"`
			OwnedBy       string `json:"owned_by,omitempty"`
			ContextLength int    `json:"context_length,omitempty"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return nil, fmt.Errorf("%s: decode models response: %w", p.compat.ProviderName, err)
	}

	models := make([]ModelInfo, 0, len(payload.Data))
	for _, item := range payload.Data {
		name := item.Name
		if name == "" {
			name = item.ID
		}
		models = append(models, ModelInfo{
			ID:            item.ID,
			Name:          name,
			Provider:      p.compat.ProviderName,
			Description:   item.Description,
			Created:       item.Created,
			OwnedBy:       item.OwnedBy,
			ContextLength: item.ContextLength,
		})
	}
	return models, nil
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

func (p *OpenAICompatProvider) validate(req *Request) error {
	if err := p.Validate(); err != nil {
		return err
	}
	// OpenAI-compatible APIs accept arbitrary extra parameters — skip validation.
	// req.Extra is merged into the request body by buildRequestBody.
	return p.BaseProvider.ValidateRequest(req)
}

func (p *OpenAICompatProvider) buildURL() string {
	base := strings.TrimSuffix(p.Config().BaseURL, "/")
	return base + p.compat.endpointPath()
}

func (p *OpenAICompatProvider) modelsURL() string {
	base := strings.TrimSuffix(p.Config().BaseURL, "/")
	return base + "/models"
}

func (p *OpenAICompatProvider) setHeaders(httpReq *http.Request, req *Request, stream bool) {
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+p.ResolveAPIKey(req))
	for k, v := range p.compat.ExtraHeaders {
		httpReq.Header.Set(k, v)
	}
	if stream {
		httpReq.Header.Set("Accept", "text/event-stream")
		for k, v := range p.compat.StreamHeaders {
			httpReq.Header.Set(k, v)
		}
	}
}

func (p *OpenAICompatProvider) newHTTPRequest(ctx context.Context, body []byte, req *Request, stream bool) (*http.Request, error) {
	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.buildURL(), bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("%s: create request: %w", p.compat.ProviderName, err)
	}
	p.setHeaders(httpReq, req, stream)
	return httpReq, nil
}

// buildRequestBody constructs the standard OpenAI-compatible JSON request body.
func (p *OpenAICompatProvider) buildRequestBody(req *Request, stream bool) ([]byte, error) {
	c := &p.compat
	body := make(map[string]any)

	body["model"] = req.Model

	// Messages — may need mutation for JSONSchemaToPrompt
	messages := PrepareMessages(req.Messages)
	if c.JSONSchemaToPrompt && req.ResponseFormat != nil &&
		req.ResponseFormat.Type == "json_schema" && req.ResponseFormat.JSONSchema != nil {
		messages = p.injectJSONSchemaToMessages(messages, req.ResponseFormat.JSONSchema)
	}

	if c.CustomMessageConverter != nil {
		body["messages"] = c.CustomMessageConverter(messages)
	} else {
		body["messages"] = ConvertMessages(messages)
	}

	// Stream
	if stream {
		body["stream"] = true
		if c.IncludeStreamUsage {
			body["stream_options"] = map[string]any{"include_usage": true}
		}
	}

	// Token limit
	if req.MaxTokens != nil {
		body[c.maxTokensField()] = *req.MaxTokens
	}

	// Sampling
	if req.Temperature != nil {
		body["temperature"] = *req.Temperature
	}
	if req.TopP != nil {
		body["top_p"] = *req.TopP
	}

	// Stop sequences
	if !c.OmitStop && len(req.Stop) > 0 {
		stops := req.Stop
		if c.MaxStopSequences > 0 && len(stops) > c.MaxStopSequences {
			stops = stops[:c.MaxStopSequences]
		}
		body["stop"] = stops
	}

	// Tools
	if len(req.Tools) > 0 {
		if c.CustomToolConverter != nil {
			body["tools"] = c.CustomToolConverter(req.Tools)
		} else {
			body["tools"] = ConvertTools(req.Tools)
		}
	}
	if req.ToolChoice != nil {
		body["tool_choice"] = req.ToolChoice
	}

	// Thinking / reasoning
	if thinking := normalizeThinking(req); thinking != nil {
		if c.ThinkingMapper != nil {
			if extra := c.ThinkingMapper(thinking, req.Model); extra != nil {
				for k, v := range extra {
					body[k] = v
				}
			}
		} else {
			// Default: {"thinking": {"type": "enabled/disabled"}}
			body["thinking"] = map[string]string{"type": thinking.Type}
		}
	}

	// Response format
	p.applyResponseFormat(body, req)

	// Provider-specific extra parameters — passthrough to API body
	for k, v := range req.Extra {
		body[k] = v
	}

	return json.Marshal(body)
}

// injectJSONSchemaToMessages copies messages and appends JSON schema instructions
// to the last user message.  Used when JSONSchemaToPrompt is true.
func (p *OpenAICompatProvider) injectJSONSchemaToMessages(messages []Message, schema *JSONSchema) []Message {
	if schema == nil || schema.Schema == nil {
		return messages
	}
	copied := make([]Message, len(messages))
	copy(copied, messages)
	for i := len(copied) - 1; i >= 0; i-- {
		if copied[i].Role == "user" {
			schemaJSON, _ := json.Marshal(schema.Schema)
			copied[i].Content += "\n\nPlease respond with a valid JSON object that strictly follows this schema:\n" +
				string(schemaJSON) + "\n\nRespond with JSON only, no additional text."
			break
		}
	}
	return copied
}

// applyResponseFormat adds response_format to the request body based on
// provider configuration.
func (p *OpenAICompatProvider) applyResponseFormat(body map[string]any, req *Request) {
	if req.ResponseFormat == nil {
		return
	}
	c := &p.compat

	// JSONSchemaToPrompt: inject schema instructions into last user message
	if c.JSONSchemaToPrompt && req.ResponseFormat.Type == "json_schema" && req.ResponseFormat.JSONSchema != nil {
		// Already injected via message mutation in the provider config
		// Fall through to set response_format as json_object
	}

	if c.ResponseFormatMapper != nil {
		if v := c.ResponseFormatMapper(req.ResponseFormat); v != nil {
			body["response_format"] = v
		}
		return
	}

	// Default behavior: json_object support only, no json_schema
	if c.SupportsJSONSchema {
		rf := map[string]any{"type": req.ResponseFormat.Type}
		if req.ResponseFormat.JSONSchema != nil {
			schema := req.ResponseFormat.JSONSchema.Schema
			if c.CleanSchema != nil {
				schema = c.CleanSchema(schema)
			}
			jsMap := map[string]any{
				"name":   req.ResponseFormat.JSONSchema.Name,
				"schema": schema,
			}
			if req.ResponseFormat.JSONSchema.Description != "" {
				jsMap["description"] = req.ResponseFormat.JSONSchema.Description
			}
			if req.ResponseFormat.JSONSchema.Strict != nil {
				jsMap["strict"] = *req.ResponseFormat.JSONSchema.Strict
			}
			rf["json_schema"] = jsMap
		}
		body["response_format"] = rf
	} else if req.ResponseFormat.Type == "json_object" {
		body["response_format"] = map[string]string{"type": "json_object"}
	}
}

// ---------------------------------------------------------------------------
// Response conversion
// ---------------------------------------------------------------------------

// compatResponse is a generic response envelope for OpenAI-compatible APIs.
type compatResponse struct {
	ID      string            `json:"id"`
	Model   string            `json:"model"`
	Choices []compatChoice    `json:"choices"`
	Usage   json.RawMessage   `json:"usage,omitempty"`
}

type compatChoice struct {
	Index        int             `json:"index"`
	Message      json.RawMessage `json:"message"`
	FinishReason string          `json:"finish_reason"`
}

func (p *OpenAICompatProvider) convertResponse(raw *compatResponse, req *Request) (*Response, error) {
	c := &p.compat
	resp := &Response{
		Provider: c.ProviderName,
	}

	// Model
	if c.ModelFromResponse && raw.Model != "" {
		resp.Model = raw.Model
	} else {
		resp.Model = req.Model
	}

	// Usage
	if len(raw.Usage) > 0 {
		resp.Usage = parseUsage(raw.Usage, c)
	}

	// Choices
	if len(raw.Choices) > 0 {
		choice := raw.Choices[0]
		resp.FinishReason = NormalizeFinishReason(choice.FinishReason)

		var msg map[string]any
		if err := json.Unmarshal(choice.Message, &msg); err != nil {
			return nil, fmt.Errorf("%s: decode message: %w", c.ProviderName, err)
		}

		// Content
		if c.ContentAsInterface {
			resp.Content = extractContentFromInterface(msg["content"])
		} else {
			resp.Content, _ = msg["content"].(string)
		}

		// Tool calls
		if rawCalls, ok := msg["tool_calls"]; ok {
			resp.ToolCalls = extractToolCalls(rawCalls)
		}

		// Reasoning — probe multiple field names
		if !isThinkingDisabled(req) && c.shouldExtractReasoning(resp.Model) {
			if reasoning, _ := c.findReasoning(msg); reasoning != "" {
				resp.Reasoning = &ReasoningData{
					Content:    reasoning,
					TokensUsed: resp.Usage.ReasoningTokens,
				}
			}
		}

		// Extra — capture non-standard message fields (logprobs, annotations, etc.)
		resp.Extra = extractMessageExtras(msg, c.reasoningFields())
	}

	return resp, nil
}

// ---------------------------------------------------------------------------
// Shared parsing helpers
// ---------------------------------------------------------------------------

// parseUsage extracts Usage from raw JSON based on provider compat config.
func parseUsage(raw json.RawMessage, c *Compat) Usage {
	var u Usage

	var std struct {
		PromptTokens          int `json:"prompt_tokens"`
		CompletionTokens      int `json:"completion_tokens"`
		TotalTokens           int `json:"total_tokens"`
		PromptCacheHitTokens  int `json:"prompt_cache_hit_tokens"`
		PromptCacheMissTokens int `json:"prompt_cache_miss_tokens"`
		PromptTokensDetails   *struct {
			CachedTokens int `json:"cached_tokens"`
		} `json:"prompt_tokens_details,omitempty"`
		CompletionTokensDetails *struct {
			ReasoningTokens int `json:"reasoning_tokens"`
		} `json:"completion_tokens_details,omitempty"`
	}
	if err := json.Unmarshal(raw, &std); err != nil {
		return u
	}

	u.PromptTokens = std.PromptTokens
	u.CompletionTokens = std.CompletionTokens
	u.TotalTokens = std.TotalTokens

	if c.HasCompletionTokenDetails && std.CompletionTokensDetails != nil {
		u.ReasoningTokens = std.CompletionTokensDetails.ReasoningTokens
	}
	if c.HasCacheTokens {
		// DeepSeek: top-level prompt_cache_hit/miss_tokens
		u.CacheReadInputTokens = std.PromptCacheHitTokens
		u.CacheCreationInputTokens = std.PromptCacheMissTokens
	}
	// Qwen/GLM/OpenAI style: prompt_tokens_details.cached_tokens
	if std.PromptTokensDetails != nil && std.PromptTokensDetails.CachedTokens > 0 {
		u.CacheReadInputTokens = std.PromptTokensDetails.CachedTokens
	}

	return u
}

// extractContentFromInterface handles Content that may be string or array.
func extractContentFromInterface(v any) string {
	switch val := v.(type) {
	case string:
		return val
	case []any:
		var b strings.Builder
		for _, item := range val {
			if m, ok := item.(map[string]any); ok {
				if text, ok := m["text"].(string); ok {
					b.WriteString(text)
				}
			}
		}
		return b.String()
	default:
		return ""
	}
}

// extractToolCalls parses tool_calls from a raw any (unmarshaled from JSON).
func extractToolCalls(v any) []ToolCall {
	arr, ok := v.([]any)
	if !ok || len(arr) == 0 {
		return nil
	}
	calls := make([]ToolCall, 0, len(arr))
	for _, item := range arr {
		m, ok := item.(map[string]any)
		if !ok {
			continue
		}
		tc := ToolCall{
			ID:   stringVal(m, "id"),
			Type: stringVal(m, "type"),
		}
		if fn, ok := m["function"].(map[string]any); ok {
			tc.Function = FunctionCall{
				Name:      stringVal(fn, "name"),
				Arguments: stringVal(fn, "arguments"),
			}
		}
		calls = append(calls, tc)
	}
	return calls
}

// stringVal extracts a string value from a map, returning "" if missing or wrong type.
func stringVal(m map[string]any, key string) string {
	v, _ := m[key].(string)
	return v
}

// extractMessageExtras returns non-standard fields from a message map.
// Known keys (role, content, tool_calls, reasoning fields) are excluded.
func extractMessageExtras(msg map[string]any, reasoningFields []string) map[string]any {
	known := map[string]bool{
		"role": true, "content": true, "tool_calls": true,
		"refusal": true,
	}
	for _, f := range reasoningFields {
		known[f] = true
	}
	var extra map[string]any
	for k, v := range msg {
		if known[k] || v == nil {
			continue
		}
		if extra == nil {
			extra = make(map[string]any)
		}
		extra[k] = v
	}
	return extra
}
