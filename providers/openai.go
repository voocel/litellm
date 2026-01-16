package providers

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

func init() {
	RegisterBuiltin("openai", func(cfg ProviderConfig) Provider {
		return NewOpenAI(cfg)
	}, "https://api.openai.com")
}

// OpenAIProvider implements OpenAI API integration
type OpenAIProvider struct {
	*BaseProvider
}

// NewOpenAI creates a new OpenAI provider
func NewOpenAI(config ProviderConfig) *OpenAIProvider {
	baseProvider := NewBaseProvider("openai", config)

	return &OpenAIProvider{
		BaseProvider: baseProvider,
	}
}

// needsMaxCompletionTokens checks if the model requires max_completion_tokens instead of max_tokens
func (p *OpenAIProvider) needsMaxCompletionTokens(model string) bool {
	modelLower := strings.ToLower(model)

	// o-series reasoning models (o1, o3, o4)
	if strings.HasPrefix(modelLower, "o1") ||
		strings.HasPrefix(modelLower, "o3") ||
		strings.HasPrefix(modelLower, "o4") {
		return true
	}

	// GPT-5 series
	if strings.HasPrefix(modelLower, "gpt-5") {
		return true
	}

	return false
}

// isReasoningModel checks if the model supports advanced reasoning (o-series and GPT-5)
func (p *OpenAIProvider) isReasoningModel(model string) bool {
	return p.needsMaxCompletionTokens(model)
}

// resolveTokenParams resolves token parameter conflicts based on model type
func (p *OpenAIProvider) resolveTokenParams(req *Request) (maxTokens *int, maxCompletionTokens *int) {
	if req.MaxTokens == nil {
		return nil, nil
	}

	if p.needsMaxCompletionTokens(req.Model) {
		// Reasoning models use max_completion_tokens
		return nil, req.MaxTokens
	}

	// Traditional models use max_tokens
	return req.MaxTokens, nil
}

func (p *OpenAIProvider) Chat(ctx context.Context, req *Request) (*Response, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}
	if err := p.BaseProvider.ValidateExtra(req.Extra, nil); err != nil {
		return nil, err
	}

	// Validate request parameters using base provider validation
	// Note: Temperature validation is skipped for reasoning models as they don't support it
	if !p.isReasoningModel(req.Model) {
		if err := p.BaseProvider.ValidateRequest(req); err != nil {
			return nil, err
		}
	} else {
		if err := validateThinking(req.Thinking); err != nil {
			return nil, fmt.Errorf("openai: %w", err)
		}
		// For reasoning models, only validate model and messages
		if req.Model == "" {
			return nil, fmt.Errorf("openai: model is required")
		}
		if len(req.Messages) == 0 {
			return nil, fmt.Errorf("openai: at least one message is required")
		}
		if req.MaxTokens != nil && *req.MaxTokens <= 0 {
			return nil, fmt.Errorf("openai: max_tokens must be positive, got %d", *req.MaxTokens)
		}
	}

	modelName := req.Model

	// Check if should use Responses API
	// Use Chat Completions API
	return p.completeWithChatAPI(ctx, req, modelName)
}

// completeWithChatAPI uses traditional Chat Completions API
func (p *OpenAIProvider) completeWithChatAPI(ctx context.Context, req *Request, modelName string) (*Response, error) {
	openaiReq := openaiRequest{
		Model:      modelName,
		Stream:     false,
		ToolChoice: req.ToolChoice,
	}

	// Convert response format
	if req.ResponseFormat != nil {
		openaiReq.ResponseFormat = p.convertResponseFormat(req.ResponseFormat)
	}

	openaiReq.TopP = req.TopP

	// Resolve token parameters based on model type
	maxTokens, maxCompletionTokens := p.resolveTokenParams(req)
	openaiReq.MaxTokens = maxTokens
	openaiReq.MaxCompletionTokens = maxCompletionTokens

	// Set temperature only for non-reasoning models
	if !p.isReasoningModel(modelName) {
		openaiReq.Temperature = req.Temperature
	}

	// Set stop sequences (up to 4 sequences)
	if len(req.Stop) > 0 {
		openaiReq.Stop = req.Stop
	}

	// Convert tool definitions
	if len(req.Tools) > 0 {
		openaiReq.Tools = ConvertTools(req.Tools)
	}

	openaiReq.Messages = ConvertMessagesToOpenAI(req.Messages)

	body, err := json.Marshal(openaiReq)
	if err != nil {
		return nil, fmt.Errorf("openai: marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.buildURL("/chat/completions"), bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("openai: create request: %w", err)
	}

	p.setHeaders(httpReq)

	resp, err := p.HTTPClient().Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, NewHTTPError("openai", resp.StatusCode, string(body))
	}

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("openai: read response: %w", err)
	}

	var openaiResp openaiResponse
	if err := json.Unmarshal(respBody, &openaiResp); err != nil {
		return nil, fmt.Errorf("openai: decode response: %w", err)
	}

	response := &Response{
		Usage: Usage{
			PromptTokens:     openaiResp.Usage.PromptTokens,
			CompletionTokens: openaiResp.Usage.CompletionTokens,
			TotalTokens:      openaiResp.Usage.TotalTokens,
		},
		Model:    openaiResp.Model,
		Provider: "openai",
	}

	// Handle reasoning tokens for reasoning models
	if openaiResp.Usage.CompletionTokensDetails != nil {
		response.Usage.ReasoningTokens = openaiResp.Usage.CompletionTokensDetails.ReasoningTokens
	}

	if len(openaiResp.Choices) > 0 {
		choice := openaiResp.Choices[0]
		response.Contents = convertOpenAIContentToMessageContents(choice.Message.Content)
		response.Content = joinMessageContentsText(response.Contents)
		response.FinishReason = choice.FinishReason

		// Extract reasoning content (OpenAI format only)
		if !isThinkingDisabled(req) && choice.ReasoningSummary != nil && choice.ReasoningSummary.Text != "" {
			response.Reasoning = &ReasoningData{
				Summary:    choice.ReasoningSummary.Text,
				TokensUsed: response.Usage.ReasoningTokens,
			}
		}

		// Convert tool calls
		if len(choice.Message.ToolCalls) > 0 {
			response.ToolCalls = make([]ToolCall, len(choice.Message.ToolCalls))
			for i, toolCall := range choice.Message.ToolCalls {
				response.ToolCalls[i] = ToolCall{
					ID:   toolCall.ID,
					Type: toolCall.Type,
					Function: FunctionCall{
						Name:      toolCall.Function.Name,
						Arguments: toolCall.Function.Arguments,
					},
				}
			}
		}
	}

	return response, nil
}

func (p *OpenAIProvider) Stream(ctx context.Context, req *Request) (StreamReader, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}
	if err := p.BaseProvider.ValidateExtra(req.Extra, nil); err != nil {
		return nil, err
	}
	if err := validateThinking(req.Thinking); err != nil {
		return nil, fmt.Errorf("openai: %w", err)
	}

	modelName := req.Model

	openaiReq := openaiRequest{
		Model:    modelName,
		Messages: make([]openaiMessage, len(req.Messages)),
		Stream:   true,
		// Enable usage statistics in streaming responses
		StreamOptions: &openaiStreamOptions{
			IncludeUsage: true,
		},
	}

	// Convert response format
	if req.ResponseFormat != nil {
		openaiReq.ResponseFormat = p.convertResponseFormat(req.ResponseFormat)
	}

	// Set correct parameters based on model type
	if p.isReasoningModel(modelName) {
		// Reasoning models always use max_completion_tokens
		openaiReq.MaxCompletionTokens = req.MaxTokens
	} else {
		// Traditional models use max_tokens and temperature
		openaiReq.MaxTokens = req.MaxTokens
		openaiReq.Temperature = req.Temperature
	}
	openaiReq.TopP = req.TopP

	// Set stop sequences (up to 4 sequences)
	if len(req.Stop) > 0 {
		openaiReq.Stop = req.Stop
	}

	// Convert tool definitions
	if len(req.Tools) > 0 {
		openaiReq.Tools = ConvertTools(req.Tools)
	}

	// Set tool choice
	if req.ToolChoice != nil {
		openaiReq.ToolChoice = req.ToolChoice
	}

	// Convert messages
	openaiReq.Messages = ConvertMessagesToOpenAI(req.Messages)

	body, err := json.Marshal(openaiReq)
	if err != nil {
		return nil, fmt.Errorf("openai: marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.buildURL("/chat/completions"), bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("openai: create request: %w", err)
	}

	p.setHeaders(httpReq)

	resp, err := p.HTTPClient().Do(httpReq)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, NewHTTPError("openai", resp.StatusCode, string(body))
	}

	scanner := bufio.NewScanner(resp.Body)
	// Increase buffer size to handle large tokens (default 64KB, max 1MB)
	buf := make([]byte, 0, 64*1024)
	scanner.Buffer(buf, 1024*1024)

	return &openaiStreamReader{
		resp:             resp,
		scanner:          scanner,
		provider:         "openai",
		includeReasoning: !isThinkingDisabled(req),
	}, nil
}

// ListModels returns available models for OpenAI.
func (p *OpenAIProvider) ListModels(ctx context.Context) ([]ModelInfo, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, "GET", p.buildURL("/models"), nil)
	if err != nil {
		return nil, fmt.Errorf("openai: create models request: %w", err)
	}
	p.setHeaders(httpReq)

	resp, err := p.HTTPClient().Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, NewHTTPError("openai", resp.StatusCode, string(body))
	}

	var payload openaiModelList
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return nil, fmt.Errorf("openai: decode models response: %w", err)
	}

	models := make([]ModelInfo, 0, len(payload.Data))
	for _, item := range payload.Data {
		models = append(models, ModelInfo{
			ID:       item.ID,
			Name:     item.ID,
			Provider: "openai",
			Created:  item.Created,
			OwnedBy:  item.OwnedBy,
		})
	}

	return models, nil
}

func (p *OpenAIProvider) buildURL(endpoint string) string {
	baseURL := strings.TrimSuffix(p.Config().BaseURL, "/")
	if baseURL == "" {
		baseURL = "https://api.openai.com"
	}
	if strings.HasSuffix(baseURL, "/v1") {
		return baseURL + endpoint
	}
	return baseURL + "/v1" + endpoint
}

func (p *OpenAIProvider) setHeaders(req *http.Request) {
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+p.Config().APIKey)
}

// openaiStreamReader implements streaming for OpenAI
type openaiStreamReader struct {
	resp             *http.Response
	scanner          *bufio.Scanner
	provider         string
	includeReasoning bool
	pendingChunks    []*StreamChunk
	done             bool
}

func (r *openaiStreamReader) Next() (*StreamChunk, error) {
	if r.done {
		return &StreamChunk{Done: true, Provider: r.provider}, nil
	}
	if len(r.pendingChunks) > 0 {
		chunk := r.pendingChunks[0]
		r.pendingChunks = r.pendingChunks[1:]
		return chunk, nil
	}

	for r.scanner.Scan() {
		line := r.scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			r.done = true
			return &StreamChunk{Done: true, Provider: r.provider}, nil
		}

		var chunk openaiStreamChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			return nil, fmt.Errorf("openai: failed to parse stream chunk: %w", err)
		}

		// Handle usage chunk (comes with empty choices array)
		if chunk.Usage != nil && len(chunk.Choices) == 0 {
			streamChunk := &StreamChunk{
				Provider: r.provider,
				Model:    chunk.Model,
				Done:     true,
				Usage: &Usage{
					PromptTokens:     chunk.Usage.PromptTokens,
					CompletionTokens: chunk.Usage.CompletionTokens,
					TotalTokens:      chunk.Usage.TotalTokens,
				},
			}
			if chunk.Usage.CompletionTokensDetails != nil {
				streamChunk.Usage.ReasoningTokens = chunk.Usage.CompletionTokensDetails.ReasoningTokens
			}
			r.done = true
			return streamChunk, nil
		}

		if len(chunk.Choices) > 0 {
			choice := chunk.Choices[0]
			pending := make([]*StreamChunk, 0, 4)

			if choice.Delta.Content != "" {
				pending = append(pending, &StreamChunk{
					Provider: r.provider,
					Model:    chunk.Model,
					Type:     "content",
					Content:  choice.Delta.Content,
				})
			}

			// Handle reasoning streaming (OpenAI format only)
			if r.includeReasoning && choice.Delta.ReasoningSummary != nil && choice.Delta.ReasoningSummary.Text != "" {
				pending = append(pending, &StreamChunk{
					Provider: r.provider,
					Model:    chunk.Model,
					Type:     "reasoning",
					Reasoning: &ReasoningChunk{
						Summary: choice.Delta.ReasoningSummary.Text,
					},
				})
			}

			// Handle tool call deltas
			if len(choice.Delta.ToolCalls) > 0 {
				for _, toolCallDelta := range choice.Delta.ToolCalls {
					chunk := &StreamChunk{
						Provider: r.provider,
						Model:    chunk.Model,
						Type:     "tool_call_delta",
						ToolCallDelta: &ToolCallDelta{
							Index: toolCallDelta.Index,
							ID:    toolCallDelta.ID,
							Type:  toolCallDelta.Type,
						},
					}

					if toolCallDelta.Function != nil {
						chunk.ToolCallDelta.FunctionName = toolCallDelta.Function.Name
						chunk.ToolCallDelta.ArgumentsDelta = toolCallDelta.Function.Arguments
					}

					pending = append(pending, chunk)
				}
			}

			// Check if stream is finished (usage chunk will come separately after this)
			if choice.FinishReason != "" {
				pending = append(pending, &StreamChunk{
					Provider:     r.provider,
					Model:        chunk.Model,
					FinishReason: choice.FinishReason,
				})
			}

			if len(pending) > 0 {
				r.pendingChunks = append(r.pendingChunks, pending[1:]...)
				return pending[0], nil
			}
		}
	}

	if err := r.scanner.Err(); err != nil {
		return nil, fmt.Errorf("openai: stream read error: %w", err)
	}

	r.done = true
	return &StreamChunk{Done: true, Provider: r.provider}, nil
}

func (r *openaiStreamReader) Close() error {
	return r.resp.Body.Close()
}

// OpenAI API structures
// OpenAI API request/response structures
// See: https://platform.openai.com/docs/api-reference/chat/create
type openaiRequest struct {
	// Required
	Model    string          `json:"model"`
	Messages []openaiMessage `json:"messages"`

	// Token limits
	MaxTokens           *int `json:"max_tokens,omitempty"`            // Deprecated: use max_completion_tokens
	MaxCompletionTokens *int `json:"max_completion_tokens,omitempty"` // Recommended for newer models

	// Sampling parameters
	Temperature      *float64       `json:"temperature,omitempty"`       // 0-2, default 1
	TopP             *float64       `json:"top_p,omitempty"`             // 0-1, default 1
	FrequencyPenalty *float64       `json:"frequency_penalty,omitempty"` // -2.0 to 2.0, default 0
	PresencePenalty  *float64       `json:"presence_penalty,omitempty"`  // -2.0 to 2.0, default 0
	LogitBias        map[string]int `json:"logit_bias,omitempty"`        // Token ID to bias (-100 to 100)

	// Output configuration
	N        *int  `json:"n,omitempty"`        // Number of completions, default 1
	Logprobs *bool `json:"logprobs,omitempty"` // Return log probabilities

	// Streaming
	Stream        bool                 `json:"stream,omitempty"`
	StreamOptions *openaiStreamOptions `json:"stream_options,omitempty"`

	// Stop sequences (up to 4, not supported with o3/o4-mini)
	Stop []string `json:"stop,omitempty"`

	// Tool/Function calling
	Tools      []openaiTool `json:"tools,omitempty"`
	ToolChoice any          `json:"tool_choice,omitempty"` // "none", "auto", "required", or specific tool

	// Response format
	ResponseFormat *openaiResponseFormat `json:"response_format,omitempty"`

	// Caching
	PromptCacheKey       string `json:"prompt_cache_key,omitempty"`       // Replaces user field
	PromptCacheRetention string `json:"prompt_cache_retention,omitempty"` // e.g., "24h"

	// Prediction (for faster responses when output is partially known)
	Prediction *openaiPrediction `json:"prediction,omitempty"`

	// Metadata and identification
	Metadata map[string]string `json:"metadata,omitempty"` // Max 16 key-value pairs

	// Output modalities (for audio-capable models)
	Modalities []string `json:"modalities,omitempty"` // ["text"] or ["text", "audio"]

	// Deprecated
	Seed *int `json:"seed,omitempty"` // Deprecated: best-effort deterministic sampling
}

// openaiPrediction configures predicted output for faster responses
type openaiPrediction struct {
	Type    string `json:"type"`    // "content"
	Content string `json:"content"` // The predicted content
}

type openaiStreamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

type openaiResponseFormat struct {
	Type       string            `json:"type"` // "text", "json_object", "json_schema"
	JSONSchema *openaiJSONSchema `json:"json_schema,omitempty"`
}

type openaiJSONSchema struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Schema      any    `json:"schema"`
	Strict      *bool  `json:"strict,omitempty"`
}

type openaiMessage struct {
	Role       string           `json:"role"`
	Content    interface{}      `json:"content,omitempty"`
	ToolCalls  []openaiToolCall `json:"tool_calls,omitempty"`
	ToolCallID string           `json:"tool_call_id,omitempty"`
}

type openaiContentPart struct {
	Type     string          `json:"type"`
	Text     string          `json:"text,omitempty"`
	ImageURL *openaiImageURL `json:"image_url,omitempty"`
}

type openaiImageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`
}

type openaiTool struct {
	Type     string              `json:"type"`
	Function *openaiToolFunction `json:"function,omitempty"`
}

type openaiToolFunction struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Parameters  any    `json:"parameters"`
}

type openaiToolCall struct {
	ID       string             `json:"id"`
	Type     string             `json:"type"`
	Function openaiToolCallFunc `json:"function"`
}

type openaiToolCallFunc struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// openaiToolCallDelta represents incremental tool call data from OpenAI
type openaiToolCallDelta struct {
	Index    int                      `json:"index"`
	ID       string                   `json:"id,omitempty"`
	Type     string                   `json:"type,omitempty"`
	Function *openaiToolCallFuncDelta `json:"function,omitempty"`
}

type openaiToolCallFuncDelta struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

type openaiResponse struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Model   string         `json:"model"`
	Choices []openaiChoice `json:"choices"`
	Usage   openaiUsage    `json:"usage"`
}

type openaiChoice struct {
	Index            int                     `json:"index"`
	Message          openaiMessage           `json:"message"`
	Delta            openaiDelta             `json:"delta,omitempty"`
	ReasoningSummary *openaiReasoningSummary `json:"reasoning_summary,omitempty"`
	FinishReason     string                  `json:"finish_reason,omitempty"`
}

type openaiReasoningSummary struct {
	Text string `json:"text,omitempty"`
}

type openaiDelta struct {
	Content          string                  `json:"content,omitempty"`
	ToolCalls        []openaiToolCallDelta   `json:"tool_calls,omitempty"`
	ReasoningSummary *openaiReasoningSummary `json:"reasoning_summary,omitempty"`
}

type openaiUsage struct {
	PromptTokens            int                            `json:"prompt_tokens"`
	CompletionTokens        int                            `json:"completion_tokens"`
	TotalTokens             int                            `json:"total_tokens"`
	CompletionTokensDetails *openaiCompletionTokensDetails `json:"completion_tokens_details,omitempty"`
}

type openaiCompletionTokensDetails struct {
	ReasoningTokens int `json:"reasoning_tokens,omitempty"`
}

type openaiStreamChunk struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Model   string         `json:"model"`
	Choices []openaiChoice `json:"choices"`
	Usage   *openaiUsage   `json:"usage,omitempty"`
}

type openaiModelList struct {
	Data []openaiModelInfo `json:"data"`
}

type openaiModelInfo struct {
	ID      string `json:"id"`
	Created int64  `json:"created,omitempty"`
	OwnedBy string `json:"owned_by,omitempty"`
}

// convertResponseFormat converts response format and ensures OpenAI compatibility
func (p *OpenAIProvider) convertResponseFormat(rf *ResponseFormat) *openaiResponseFormat {
	if rf == nil {
		return nil
	}

	result := &openaiResponseFormat{Type: rf.Type}

	if rf.JSONSchema != nil {
		schema := rf.JSONSchema.Schema

		// Clean schema for OpenAI compatibility
		schema = p.cleanSchemaForOpenAI(schema)

		// Apply strict mode if requested
		if rf.JSONSchema.Strict != nil && *rf.JSONSchema.Strict {
			schema = p.ensureStrictSchema(schema)
		}

		result.JSONSchema = &openaiJSONSchema{
			Name:        rf.JSONSchema.Name,
			Description: rf.JSONSchema.Description,
			Schema:      schema,
			Strict:      rf.JSONSchema.Strict,
		}
	}

	return result
}

// cleanSchemaForOpenAI removes unsupported validation rules and ensures OpenAI compatibility
func (p *OpenAIProvider) cleanSchemaForOpenAI(schema interface{}) interface{} {
	switch s := schema.(type) {
	case map[string]interface{}:
		result := make(map[string]interface{}, len(s))
		for k, v := range s {
			// Skip unsupported OpenAI schema properties
			if k == "examples" || k == "default" || k == "const" {
				continue
			}
			result[k] = p.cleanSchemaForOpenAI(v)
		}

		// Add additionalProperties: false for objects in strict mode
		if result["type"] == "object" {
			if _, hasAdditionalProps := result["additionalProperties"]; !hasAdditionalProps {
				result["additionalProperties"] = false
			}
		}

		return result
	case []interface{}:
		cleaned := make([]interface{}, len(s))
		for i, v := range s {
			cleaned[i] = p.cleanSchemaForOpenAI(v)
		}
		return cleaned
	default:
		return schema
	}
}

// ensureStrictSchema recursively adds additionalProperties: false to all objects for OpenAI strict mode
func (p *OpenAIProvider) ensureStrictSchema(schema interface{}) interface{} {
	return p.cleanSchemaForOpenAI(schema)
}

// ==================== Common conversion helpers (used by OpenAI-compatible providers) ====================

// OpenAICompatMessage is a generic OpenAI-compatible message format
// Used by DeepSeek/GLM/Qwen etc.; Content stays interface{} to allow flexible formats
type OpenAICompatMessage struct {
	Role             string           `json:"role"`
	Content          interface{}      `json:"content,omitempty"` // string or array (needed by OpenRouter)
	ToolCalls        []openaiToolCall `json:"tool_calls,omitempty"`
	ToolCallID       string           `json:"tool_call_id,omitempty"`
	ReasoningContent string           `json:"reasoning_content,omitempty"` // used by GLM/Qwen
}

// ConvertMessagesToOpenAI converts generic Message into native OpenAI format (string content or parts)
// Used internally by OpenAI provider
func ConvertMessagesToOpenAI(messages []Message) []openaiMessage {
	result := make([]openaiMessage, len(messages))
	for i, msg := range messages {
		openaiMsg := openaiMessage{
			Role:       msg.Role,
			ToolCallID: msg.ToolCallID,
		}

		if parts := buildOpenAIContentParts(msg); len(parts) > 0 {
			openaiMsg.Content = parts
		} else if msg.Content != "" {
			openaiMsg.Content = msg.Content
		}

		// Convert tool calls
		if len(msg.ToolCalls) > 0 {
			openaiMsg.ToolCalls = make([]openaiToolCall, len(msg.ToolCalls))
			for j, tc := range msg.ToolCalls {
				openaiMsg.ToolCalls[j] = openaiToolCall{
					ID:   tc.ID,
					Type: tc.Type,
					Function: openaiToolCallFunc{
						Name:      tc.Function.Name,
						Arguments: tc.Function.Arguments,
					},
				}
			}
		}

		result[i] = openaiMsg
	}
	return result
}

// ConvertMessages converts generic Message into OpenAI-compatible format (Content as interface{})
// For DeepSeek/GLM/Qwen/OpenRouter etc.
func ConvertMessages(messages []Message) []OpenAICompatMessage {
	result := make([]OpenAICompatMessage, len(messages))
	for i, msg := range messages {
		compatMsg := OpenAICompatMessage{
			Role:       msg.Role,
			ToolCallID: msg.ToolCallID,
		}

		if parts := buildOpenAIContentParts(msg); len(parts) > 0 {
			compatMsg.Content = parts
		} else {
			compatMsg.Content = msg.Content
		}

		// Convert tool calls
		if len(msg.ToolCalls) > 0 {
			compatMsg.ToolCalls = make([]openaiToolCall, len(msg.ToolCalls))
			for j, tc := range msg.ToolCalls {
				compatMsg.ToolCalls[j] = openaiToolCall{
					ID:   tc.ID,
					Type: tc.Type,
					Function: openaiToolCallFunc{
						Name:      tc.Function.Name,
						Arguments: tc.Function.Arguments,
					},
				}
			}
		}

		result[i] = compatMsg
	}
	return result
}

// ConvertTools converts generic Tool into OpenAI format; shared by all OpenAI-compatible providers
func ConvertTools(tools []Tool) []openaiTool {
	result := make([]openaiTool, len(tools))
	for i, tool := range tools {
		var function *openaiToolFunction
		if tool.Type == "function" {
			function = &openaiToolFunction{
				Name:        tool.Function.Name,
				Description: tool.Function.Description,
				Parameters:  tool.Function.Parameters,
			}
		}
		result[i] = openaiTool{
			Type:     tool.Type,
			Function: function,
		}
	}
	return result
}

func buildOpenAIContentParts(msg Message) []openaiContentPart {
	contents := normalizeMessageContents(msg)
	if len(contents) == 0 {
		return nil
	}

	parts := make([]openaiContentPart, 0, len(contents))
	for _, content := range contents {
		switch strings.ToLower(content.Type) {
		case "", "text", "input_text":
			if content.Text == "" {
				continue
			}
			parts = append(parts, openaiContentPart{
				Type: "text",
				Text: content.Text,
			})
		case "image_url", "input_image":
			if content.ImageURL == nil || content.ImageURL.URL == "" {
				continue
			}
			parts = append(parts, openaiContentPart{
				Type: "image_url",
				ImageURL: &openaiImageURL{
					URL:    content.ImageURL.URL,
					Detail: content.ImageURL.Detail,
				},
			})
		default:
			if content.Text != "" {
				parts = append(parts, openaiContentPart{
					Type: "text",
					Text: content.Text,
				})
			}
		}
	}

	return parts
}

func normalizeMessageContents(msg Message) []MessageContent {
	if len(msg.Contents) > 0 {
		return msg.Contents
	}
	if msg.Content != "" {
		return []MessageContent{{Type: "text", Text: msg.Content}}
	}
	return nil
}

func convertOpenAIContentToMessageContents(content interface{}) []MessageContent {
	switch val := content.(type) {
	case string:
		if val == "" {
			return nil
		}
		return []MessageContent{{Type: "text", Text: val}}
	case []interface{}:
		result := make([]MessageContent, 0, len(val))
		for _, item := range val {
			partMap, ok := item.(map[string]interface{})
			if !ok {
				continue
			}
			partType, _ := partMap["type"].(string)
			switch strings.ToLower(partType) {
			case "image_url":
				if imageData, ok := partMap["image_url"].(map[string]interface{}); ok {
					if urlValue, _ := imageData["url"].(string); urlValue != "" {
						detailValue, _ := imageData["detail"].(string)
						result = append(result, MessageContent{
							Type: "image_url",
							ImageURL: &MessageImageURL{
								URL:    urlValue,
								Detail: detailValue,
							},
						})
					}
				}

			default:
				textValue, _ := partMap["text"].(string)
				if textValue == "" {
					continue
				}
				result = append(result, MessageContent{
					Type:        "text",
					Text:        textValue,
					Annotations: extractAnnotations(partMap),
					Logprobs:    extractLogprobs(partMap),
				})
			}
		}
		return result
	default:
		return nil
	}
}

func joinMessageContentsText(contents []MessageContent) string {
	if len(contents) == 0 {
		return ""
	}
	var builder strings.Builder
	for _, content := range contents {
		if content.Type != "text" || content.Text == "" {
			continue
		}
		if builder.Len() > 0 {
			builder.WriteString("\n")
		}
		builder.WriteString(content.Text)
	}
	return builder.String()
}

func extractAnnotations(part map[string]interface{}) []map[string]interface{} {
	raw, ok := part["annotations"].([]interface{})
	if !ok || len(raw) == 0 {
		return nil
	}
	result := make([]map[string]interface{}, 0, len(raw))
	for _, entry := range raw {
		if data, ok := entry.(map[string]interface{}); ok {
			result = append(result, data)
		}
	}
	return result
}

func extractLogprobs(part map[string]interface{}) []map[string]interface{} {
	raw, ok := part["logprobs"].([]interface{})
	if !ok || len(raw) == 0 {
		return nil
	}
	result := make([]map[string]interface{}, 0, len(raw))
	for _, entry := range raw {
		if data, ok := entry.(map[string]interface{}); ok {
			result = append(result, data)
		}
	}
	return result
}
