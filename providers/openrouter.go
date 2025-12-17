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
	RegisterBuiltin("openrouter", func(cfg ProviderConfig) Provider {
		return NewOpenRouter(cfg)
	}, "https://openrouter.ai/api/v1")
}

// OpenRouterProvider implements OpenRouter API integration
type OpenRouterProvider struct {
	*BaseProvider
}

// NewOpenRouter creates a new OpenRouter provider
func NewOpenRouter(config ProviderConfig) *OpenRouterProvider {
	baseProvider := NewBaseProvider("openrouter", config)

	return &OpenRouterProvider{
		BaseProvider: baseProvider,
	}
}

func (p *OpenRouterProvider) SupportsModel(model string) bool {
	// OpenRouter supports many models from different providers
	// We'll accept any model name and let OpenRouter validate it
	return model != ""
}

func (p *OpenRouterProvider) Models() []ModelInfo {
	return []ModelInfo{
		{
			ID: "openai/gpt-4o", Provider: "openrouter", Name: "GPT-4o via OpenRouter", ContextWindow: 128000,
			Capabilities: []string{"chat", "vision", "function_call", "tool_use"},
		},
		{
			ID: "openai/gpt-4o-mini", Provider: "openrouter", Name: "GPT-4o Mini via OpenRouter", ContextWindow: 128000,
			Capabilities: []string{"chat", "vision", "function_call", "tool_use"},
		},
		{
			ID: "anthropic/claude-3.7-sonnet", Provider: "openrouter", Name: "Claude 3.7 Sonnet via OpenRouter", ContextWindow: 200000,
			Capabilities: []string{"chat", "vision", "function_call", "tool_use"},
		},
		{
			ID: "google/gemini-2.0-flash-exp", Provider: "openrouter", Name: "Gemini 2.0 Flash via OpenRouter", ContextWindow: 1000000,
			Capabilities: []string{"chat", "vision", "function_call", "tool_use"},
		},
		{
			ID: "meta-llama/llama-3.3-70b-instruct", Provider: "openrouter", Name: "Llama 3.3 70B via OpenRouter", ContextWindow: 131072,
			Capabilities: []string{"chat", "function_call", "tool_use"},
		},
		{
			ID: "deepseek/deepseek-chat", Provider: "openrouter", Name: "DeepSeek Chat via OpenRouter", ContextWindow: 64000,
			Capabilities: []string{"chat", "function_call", "tool_use"},
		},
		{
			ID: "qwen/qwen-2.5-72b-instruct", Provider: "openrouter", Name: "Qwen 2.5 72B via OpenRouter", ContextWindow: 32768,
			Capabilities: []string{"chat", "function_call", "tool_use"},
		},
	}
}

func (p *OpenRouterProvider) Chat(ctx context.Context, req *Request) (*Response, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}

	if err := p.BaseProvider.ValidateRequest(req); err != nil {
		return nil, err
	}

	// Build OpenRouter request
	openRouterReq := map[string]any{
		"model":    req.Model,
		"messages": p.convertMessages(req.Messages),
	}

	// Add optional parameters
	if req.MaxTokens != nil {
		openRouterReq["max_tokens"] = *req.MaxTokens
	}
	if req.Temperature != nil {
		openRouterReq["temperature"] = *req.Temperature
	}
	if len(req.Tools) > 0 {
		openRouterReq["tools"] = p.convertTools(req.Tools)
	}
	if req.ToolChoice != nil {
		openRouterReq["tool_choice"] = req.ToolChoice
	}

	// Handle response format
	if req.ResponseFormat != nil {
		responseFormat := map[string]any{"type": req.ResponseFormat.Type}
		if req.ResponseFormat.JSONSchema != nil {
			schema := req.ResponseFormat.JSONSchema.Schema

			// For strict mode or when using OpenAI models through OpenRouter,
			// ensure all objects have additionalProperties: false
			if req.ResponseFormat.JSONSchema.Strict != nil && *req.ResponseFormat.JSONSchema.Strict {
				schema = p.ensureStrictSchema(schema)
			}

			responseFormat["json_schema"] = map[string]any{
				"name":        req.ResponseFormat.JSONSchema.Name,
				"description": req.ResponseFormat.JSONSchema.Description,
				"schema":      schema,
			}
			if req.ResponseFormat.JSONSchema.Strict != nil {
				responseFormat["json_schema"].(map[string]any)["strict"] = *req.ResponseFormat.JSONSchema.Strict
			}
		}
		openRouterReq["response_format"] = responseFormat
	}

	// Handle reasoning parameters (for o1 models via OpenRouter)
	if req.ReasoningEffort != "" || req.ReasoningSummary != "" {
		reasoning := map[string]any{}
		if req.MaxTokens != nil {
			// Allocate reasonable tokens for reasoning
			totalTokens := *req.MaxTokens
			reasoningTokens := 1024
			if totalTokens >= 2048 {
				reasoningTokens = int(float64(totalTokens) * 0.3)
				if reasoningTokens < 1024 {
					reasoningTokens = 1024
				}
			}
			reasoning["max_tokens"] = reasoningTokens
		}
		openRouterReq["reasoning"] = reasoning
	}

	// Send request
	body, err := json.Marshal(openRouterReq)
	if err != nil {
		return nil, fmt.Errorf("openrouter: marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/chat/completions", p.Config().BaseURL)
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("openrouter: create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+p.Config().APIKey)
	httpReq.Header.Set("HTTP-Referer", "https://github.com/voocel/litellm") // OpenRouter requires referer
	httpReq.Header.Set("X-Title", "litellm")

	resp, err := p.HTTPClient().Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body))
	}

	var openRouterResp openRouterResponse
	if err := json.NewDecoder(resp.Body).Decode(&openRouterResp); err != nil {
		return nil, fmt.Errorf("openrouter: decode response: %w", err)
	}

	// Build response
	response := &Response{
		Model:    openRouterResp.Model,
		Provider: "openrouter",
	}

	if openRouterResp.Usage != nil {
		reasoningTokens := 0
		if openRouterResp.Usage.CompletionTokensDetails != nil {
			reasoningTokens = openRouterResp.Usage.CompletionTokensDetails.ReasoningTokens
		}

		response.Usage = Usage{
			PromptTokens:     openRouterResp.Usage.PromptTokens,
			CompletionTokens: openRouterResp.Usage.CompletionTokens,
			TotalTokens:      openRouterResp.Usage.TotalTokens,
			ReasoningTokens:  reasoningTokens,
		}
	}

	if len(openRouterResp.Choices) > 0 {
		choice := openRouterResp.Choices[0]

		// Handle content (can be string or array)
		if content, ok := choice.Message.Content.(string); ok {
			response.Content = content
		} else if contentArray, ok := choice.Message.Content.([]interface{}); ok {
			for _, item := range contentArray {
				if contentItem, ok := item.(map[string]interface{}); ok {
					if text, ok := contentItem["text"].(string); ok {
						response.Content += text
					}
				}
			}
		}

		response.FinishReason = choice.FinishReason

		// Handle tool calls
		if len(choice.Message.ToolCalls) > 0 {
			for _, toolCall := range choice.Message.ToolCalls {
				response.ToolCalls = append(response.ToolCalls, ToolCall{
					ID:   toolCall.ID,
					Type: toolCall.Type,
					Function: FunctionCall{
						Name:      toolCall.Function.Name,
						Arguments: toolCall.Function.Arguments,
					},
				})
			}
		}

		// Handle reasoning content (for o1 models via OpenRouter)
		if choice.Message.Reasoning != "" {
			response.Reasoning = &ReasoningData{
				Content:    choice.Message.Reasoning,
				Summary:    "OpenRouter reasoning process",
				TokensUsed: response.Usage.ReasoningTokens, // Use the already calculated value
			}
		}
	}

	return response, nil
}

func (p *OpenRouterProvider) Stream(ctx context.Context, req *Request) (StreamReader, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}

	// Build request (similar to Chat but with stream: true)
	openRouterReq := map[string]any{
		"model":    req.Model,
		"messages": p.convertMessages(req.Messages),
		"stream":   true,
	}

	if req.MaxTokens != nil {
		openRouterReq["max_tokens"] = *req.MaxTokens
	}
	if req.Temperature != nil {
		openRouterReq["temperature"] = *req.Temperature
	}
	if len(req.Tools) > 0 {
		openRouterReq["tools"] = p.convertTools(req.Tools)
	}

	body, err := json.Marshal(openRouterReq)
	if err != nil {
		return nil, fmt.Errorf("openrouter: marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/chat/completions", p.Config().BaseURL)
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("openrouter: create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+p.Config().APIKey)
	httpReq.Header.Set("Accept", "text/event-stream")
	httpReq.Header.Set("HTTP-Referer", "https://github.com/voocel/litellm")
	httpReq.Header.Set("X-Title", "litellm")

	resp, err := p.HTTPClient().Do(httpReq)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body))
	}

	return &openRouterStreamReader{
		resp:     resp,
		scanner:  bufio.NewScanner(resp.Body),
		provider: "openrouter",
	}, nil
}

// convertMessages converts standard messages to OpenRouter format
func (p *OpenRouterProvider) convertMessages(messages []Message) []openRouterMessage {
	openRouterMessages := make([]openRouterMessage, len(messages))
	for i, msg := range messages {
		openRouterMsg := openRouterMessage{
			Role: msg.Role,
		}

		// Handle content with cache control (Anthropic format for OpenRouter)
		if msg.CacheControl != nil {
			// Use Anthropic's content array format for caching
			openRouterMsg.Content = []openRouterContent{
				{
					Type: "text",
					Text: msg.Content,
					CacheControl: &openRouterCacheControl{
						Type: msg.CacheControl.Type,
					},
				},
			}
		} else {
			// Simple string content
			openRouterMsg.Content = msg.Content
		}

		// Handle tool calls
		if len(msg.ToolCalls) > 0 {
			for _, toolCall := range msg.ToolCalls {
				openRouterMsg.ToolCalls = append(openRouterMsg.ToolCalls, openaiToolCall{
					ID:   toolCall.ID,
					Type: toolCall.Type,
					Function: openaiToolCallFunc{
						Name:      toolCall.Function.Name,
						Arguments: toolCall.Function.Arguments,
					},
				})
			}
		}

		// Handle tool call responses
		if msg.ToolCallID != "" {
			openRouterMsg.ToolCallID = msg.ToolCallID
		}

		openRouterMessages[i] = openRouterMsg
	}
	return openRouterMessages
}

// convertTools converts standard tools to OpenRouter format
func (p *OpenRouterProvider) convertTools(tools []Tool) []openRouterTool {
	openRouterTools := make([]openRouterTool, len(tools))
	for i, tool := range tools {
		openRouterTools[i] = openRouterTool{
			Type: tool.Type,
			Function: openRouterToolFunction{
				Name:        tool.Function.Name,
				Description: tool.Function.Description,
				Parameters:  tool.Function.Parameters,
			},
		}
	}
	return openRouterTools
}

// OpenRouter API structures
type openRouterMessage struct {
	Role       string           `json:"role"`
	Content    interface{}      `json:"content,omitempty"` // Can be string or []openRouterContent for caching
	Reasoning  string           `json:"reasoning,omitempty"`
	ToolCalls  []openaiToolCall `json:"tool_calls,omitempty"`
	ToolCallID string           `json:"tool_call_id,omitempty"`
}

// openRouterContent represents content with cache control (Anthropic format)
type openRouterContent struct {
	Type         string                  `json:"type"`
	Text         string                  `json:"text"`
	CacheControl *openRouterCacheControl `json:"cache_control,omitempty"`
}

// openRouterCacheControl represents cache control for OpenRouter (Anthropic format)
type openRouterCacheControl struct {
	Type string `json:"type"`
}

type openRouterTool struct {
	Type     string                 `json:"type"`
	Function openRouterToolFunction `json:"function"`
}

type openRouterToolFunction struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Parameters  any    `json:"parameters"`
}

type openRouterResponse struct {
	ID      string             `json:"id"`
	Object  string             `json:"object"`
	Model   string             `json:"model"`
	Choices []openRouterChoice `json:"choices"`
	Usage   *openRouterUsage   `json:"usage,omitempty"`
}

type openRouterChoice struct {
	Index        int               `json:"index"`
	Message      openRouterMessage `json:"message"`
	FinishReason string            `json:"finish_reason"`
}

type openRouterUsage struct {
	PromptTokens            int                                `json:"prompt_tokens"`
	CompletionTokens        int                                `json:"completion_tokens"`
	TotalTokens             int                                `json:"total_tokens"`
	CompletionTokensDetails *openRouterCompletionTokensDetails `json:"completion_tokens_details,omitempty"`
}

type openRouterCompletionTokensDetails struct {
	ReasoningTokens int `json:"reasoning_tokens,omitempty"`
}

// openRouterStreamReader implements streaming for OpenRouter
type openRouterStreamReader struct {
	resp     *http.Response
	scanner  *bufio.Scanner
	provider string
	done     bool
}

func (r *openRouterStreamReader) Next() (*StreamChunk, error) {
	if r.done {
		return &StreamChunk{Done: true, Provider: r.provider}, nil
	}

	for r.scanner.Scan() {
		line := strings.TrimSpace(r.scanner.Text())

		// Skip empty lines and comments
		if line == "" || strings.HasPrefix(line, ":") {
			continue
		}

		// Handle data lines
		if strings.HasPrefix(line, "data: ") {
			data := strings.TrimPrefix(line, "data: ")

			// Check for end of stream
			if data == "[DONE]" {
				r.done = true
				return &StreamChunk{Done: true, Provider: r.provider}, nil
			}

			var streamResp openRouterStreamResponse
			if err := json.Unmarshal([]byte(data), &streamResp); err != nil {
				// Return error instead of silently ignoring malformed chunks
				return nil, fmt.Errorf("openrouter: failed to parse stream chunk: %w", err)
			}

			// Convert to StreamChunk
			if len(streamResp.Choices) > 0 {
				choice := streamResp.Choices[0]
				chunk := &StreamChunk{
					Provider: r.provider,
				}

				if choice.Delta.Content != "" {
					chunk.Type = "content"
					chunk.Content = choice.Delta.Content
				}

				if choice.Delta.Reasoning != "" {
					chunk.Type = "reasoning"
					chunk.Reasoning = &ReasoningChunk{
						Content: choice.Delta.Reasoning,
						Summary: "OpenRouter reasoning process",
					}
				}

				if len(choice.Delta.ToolCalls) > 0 {
					chunk.Type = "tool_call_delta"
					toolCall := choice.Delta.ToolCalls[0]
					chunk.ToolCallDelta = &ToolCallDelta{
						Index:          choice.Index,
						ID:             toolCall.ID,
						Type:           toolCall.Type,
						FunctionName:   toolCall.Function.Name,
						ArgumentsDelta: toolCall.Function.Arguments,
					}
				}

				if choice.FinishReason != "" {
					chunk.FinishReason = choice.FinishReason
					chunk.Done = true
					r.done = true
				}

				return chunk, nil
			}
		}
	}

	if err := r.scanner.Err(); err != nil {
		return nil, err
	}

	r.done = true
	return &StreamChunk{Done: true, Provider: r.provider}, nil
}

func (r *openRouterStreamReader) Close() error {
	return r.resp.Body.Close()
}

// ensureStrictSchema recursively adds additionalProperties: false to all objects for OpenAI strict mode
// This is needed because OpenRouter uses OpenAI models through Azure, which requires strict JSON schemas
func (p *OpenRouterProvider) ensureStrictSchema(schema interface{}) interface{} {
	switch s := schema.(type) {
	case map[string]interface{}:
		result := make(map[string]interface{}, len(s))
		for k, v := range s {
			result[k] = p.ensureStrictSchema(v)
		}
		if result["type"] == "object" {
			result["additionalProperties"] = false
		}
		return result
	case []interface{}:
		for i, v := range s {
			s[i] = p.ensureStrictSchema(v)
		}
		return s
	default:
		return schema
	}
}

// Streaming response structures
type openRouterStreamResponse struct {
	ID      string                   `json:"id"`
	Object  string                   `json:"object"`
	Model   string                   `json:"model"`
	Choices []openRouterStreamChoice `json:"choices"`
}

type openRouterStreamChoice struct {
	Index        int             `json:"index"`
	Delta        openRouterDelta `json:"delta"`
	FinishReason string          `json:"finish_reason"`
}

type openRouterDelta struct {
	Role      string           `json:"role,omitempty"`
	Content   string           `json:"content,omitempty"`
	Reasoning string           `json:"reasoning,omitempty"`
	ToolCalls []openaiToolCall `json:"tool_calls,omitempty"`
}
