package litellm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// OpenRouterProvider implements the Provider interface for OpenRouter
type OpenRouterProvider struct {
	*BaseProvider
}

// NewOpenRouterProvider creates a new OpenRouter provider
func NewOpenRouterProvider(config ProviderConfig) Provider {
	return &OpenRouterProvider{
		BaseProvider: &BaseProvider{
			name:       "openrouter",
			config:     config,
			httpClient: &http.Client{Timeout: 30 * time.Second},
		},
	}
}

// OpenRouter API request/response structures
type openRouterRequest struct {
	Model          string                    `json:"model"`
	Messages       []openRouterMessage       `json:"messages"`
	MaxTokens      *int                      `json:"max_tokens,omitempty"`
	Temperature    *float64                  `json:"temperature,omitempty"`
	Stream         bool                      `json:"stream,omitempty"`
	Tools          []openRouterTool          `json:"tools,omitempty"`
	ToolChoice     interface{}               `json:"tool_choice,omitempty"`
	ResponseFormat *openRouterResponseFormat `json:"response_format,omitempty"`
	Reasoning      *openRouterReasoning      `json:"reasoning,omitempty"`
}

type openRouterReasoning struct {
	MaxTokens *int `json:"max_tokens,omitempty"`
}

type openRouterResponseFormat struct {
	Type       string                `json:"type"` // "text", "json_object", "json_schema"
	JSONSchema *openRouterJSONSchema `json:"json_schema,omitempty"`
}

type openRouterJSONSchema struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Schema      any    `json:"schema"`
	Strict      *bool  `json:"strict,omitempty"`
}

type openRouterMessage struct {
	Role       string               `json:"role"`
	Content    string               `json:"content,omitempty"`
	Reasoning  string               `json:"reasoning,omitempty"`
	ToolCalls  []openRouterToolCall `json:"tool_calls,omitempty"`
	ToolCallID string               `json:"tool_call_id,omitempty"`
}

type openRouterTool struct {
	Type     string                 `json:"type"`
	Function openRouterToolFunction `json:"function"`
}

type openRouterToolFunction struct {
	Name        string      `json:"name"`
	Description string      `json:"description"`
	Parameters  interface{} `json:"parameters"`
}

type openRouterToolCall struct {
	ID       string                     `json:"id"`
	Type     string                     `json:"type"`
	Function openRouterToolCallFunction `json:"function"`
}

type openRouterToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type openRouterResponse struct {
	ID      string             `json:"id"`
	Object  string             `json:"object"`
	Model   string             `json:"model"`
	Choices []openRouterChoice `json:"choices"`
	Usage   openRouterUsage    `json:"usage"`
}

type openRouterChoice struct {
	Index        int               `json:"index"`
	Message      openRouterMessage `json:"message"`
	FinishReason string            `json:"finish_reason"`
	Delta        openRouterDelta   `json:"delta,omitempty"`
}

type openRouterDelta struct {
	Role      string               `json:"role,omitempty"`
	Content   string               `json:"content,omitempty"`
	Reasoning string               `json:"reasoning,omitempty"`
	ToolCalls []openRouterToolCall `json:"tool_calls,omitempty"`
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

// Complete implements the Provider interface
func (p *OpenRouterProvider) Complete(ctx context.Context, req *Request) (*Response, error) {
	openRouterReq := openRouterRequest{
		Model:       req.Model,
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
		Stream:      false,
		ToolChoice:  req.ToolChoice,
	}

	// Set reasoning parameters if provided
	if req.ReasoningEffort != "" || req.ReasoningSummary != "" {
		reasoning := &openRouterReasoning{}

		// OpenRouter uses max_tokens for reasoning
		if req.MaxTokens != nil {
			// Ensure we have enough tokens for reasoning (minimum 1024)
			totalTokens := *req.MaxTokens
			var reasoningTokens int

			if totalTokens >= 2048 {
				// Use 30% for reasoning if we have enough tokens
				reasoningTokens = int(float64(totalTokens) * 0.3)
				if reasoningTokens < 1024 {
					reasoningTokens = 1024
				}
			} else if totalTokens >= 1024 {
				// Use minimum if total is small but >= 1024
				reasoningTokens = 1024
			} else {
				// Increase total tokens if too small
				totalTokens = 2048
				reasoningTokens = 1024
				openRouterReq.MaxTokens = &totalTokens
			}

			reasoning.MaxTokens = &reasoningTokens
		} else {
			// Default: 2048 total, 1024 for reasoning
			defaultTotal := 2048
			defaultReasoning := 1024
			reasoning.MaxTokens = &defaultReasoning
			openRouterReq.MaxTokens = &defaultTotal
		}

		openRouterReq.Reasoning = reasoning
	}

	// Convert response format - OpenRouter supports both json_object and json_schema
	if req.ResponseFormat != nil {
		openRouterReq.ResponseFormat = &openRouterResponseFormat{
			Type: req.ResponseFormat.Type,
		}
		if req.ResponseFormat.JSONSchema != nil {
			openRouterReq.ResponseFormat.JSONSchema = &openRouterJSONSchema{
				Name:        req.ResponseFormat.JSONSchema.Name,
				Description: req.ResponseFormat.JSONSchema.Description,
				Schema:      req.ResponseFormat.JSONSchema.Schema,
				Strict:      req.ResponseFormat.JSONSchema.Strict,
			}
		}
	}

	// Convert tool definitions
	if len(req.Tools) > 0 {
		openRouterReq.Tools = make([]openRouterTool, len(req.Tools))
		for i, tool := range req.Tools {
			openRouterReq.Tools[i] = openRouterTool{
				Type: tool.Type,
				Function: openRouterToolFunction{
					Name:        tool.Function.Name,
					Description: tool.Function.Description,
					Parameters:  tool.Function.Parameters,
				},
			}
		}
	}

	// Convert messages
	openRouterReq.Messages = make([]openRouterMessage, len(req.Messages))
	for i, msg := range req.Messages {
		openRouterMsg := openRouterMessage{
			Role:    msg.Role,
			Content: msg.Content,
		}

		// Convert tool calls if present
		if len(msg.ToolCalls) > 0 {
			openRouterMsg.ToolCalls = make([]openRouterToolCall, len(msg.ToolCalls))
			for j, toolCall := range msg.ToolCalls {
				openRouterMsg.ToolCalls[j] = openRouterToolCall{
					ID:   toolCall.ID,
					Type: toolCall.Type,
					Function: openRouterToolCallFunction{
						Name:      toolCall.Function.Name,
						Arguments: toolCall.Function.Arguments,
					},
				}
			}
		}

		if msg.ToolCallID != "" {
			openRouterMsg.ToolCallID = msg.ToolCallID
		}

		openRouterReq.Messages[i] = openRouterMsg
	}

	body, err := json.Marshal(openRouterReq)
	if err != nil {
		return nil, fmt.Errorf("openrouter: marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.buildURL("/chat/completions"), bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("openrouter: create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+p.config.APIKey)

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("openrouter: request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("openrouter: API error %d: %s", resp.StatusCode, string(body))
	}

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("openrouter: read response: %w", err)
	}

	var openRouterResp openRouterResponse
	if err := json.Unmarshal(respBody, &openRouterResp); err != nil {
		return nil, fmt.Errorf("openrouter: decode response: %w", err)
	}

	response := &Response{
		Usage: Usage{
			PromptTokens:     openRouterResp.Usage.PromptTokens,
			CompletionTokens: openRouterResp.Usage.CompletionTokens,
			TotalTokens:      openRouterResp.Usage.TotalTokens,
		},
	}

	// Extract reasoning tokens if available
	if openRouterResp.Usage.CompletionTokensDetails != nil {
		response.Usage.ReasoningTokens = openRouterResp.Usage.CompletionTokensDetails.ReasoningTokens
	}

	if len(openRouterResp.Choices) > 0 {
		choice := openRouterResp.Choices[0]
		response.Content = choice.Message.Content

		// Extract reasoning content if present
		if choice.Message.Reasoning != "" {
			response.Reasoning = &ReasoningData{
				Content:    choice.Message.Reasoning,
				Summary:    choice.Message.Reasoning,
				TokensUsed: 0, // OpenRouter doesn't provide reasoning token count separately
			}
		}

		// Convert tool calls if present
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

// Stream implements the Provider interface for streaming responses
func (p *OpenRouterProvider) Stream(ctx context.Context, req *Request) (StreamReader, error) {
	openRouterReq := openRouterRequest{
		Model:       req.Model,
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
		Stream:      true,
		ToolChoice:  req.ToolChoice,
	}

	// Set reasoning parameters if provided
	if req.ReasoningEffort != "" || req.ReasoningSummary != "" {
		reasoning := &openRouterReasoning{}

		if req.MaxTokens != nil {
			totalTokens := *req.MaxTokens
			var reasoningTokens int

			if totalTokens >= 2048 {
				reasoningTokens = int(float64(totalTokens) * 0.3)
				if reasoningTokens < 1024 {
					reasoningTokens = 1024
				}
			} else if totalTokens >= 1024 {
				reasoningTokens = 1024
			} else {
				totalTokens = 2048
				reasoningTokens = 1024
				openRouterReq.MaxTokens = &totalTokens
			}

			reasoning.MaxTokens = &reasoningTokens
		} else {
			defaultTotal := 2048
			defaultReasoning := 1024
			reasoning.MaxTokens = &defaultReasoning
			openRouterReq.MaxTokens = &defaultTotal
		}

		openRouterReq.Reasoning = reasoning
	}

	// Convert response format - OpenRouter supports both json_object and json_schema
	if req.ResponseFormat != nil {
		openRouterReq.ResponseFormat = &openRouterResponseFormat{
			Type: req.ResponseFormat.Type,
		}
		if req.ResponseFormat.JSONSchema != nil {
			openRouterReq.ResponseFormat.JSONSchema = &openRouterJSONSchema{
				Name:        req.ResponseFormat.JSONSchema.Name,
				Description: req.ResponseFormat.JSONSchema.Description,
				Schema:      req.ResponseFormat.JSONSchema.Schema,
				Strict:      req.ResponseFormat.JSONSchema.Strict,
			}
		}
	}

	// Convert tool definitions
	if len(req.Tools) > 0 {
		openRouterReq.Tools = make([]openRouterTool, len(req.Tools))
		for i, tool := range req.Tools {
			openRouterReq.Tools[i] = openRouterTool{
				Type: tool.Type,
				Function: openRouterToolFunction{
					Name:        tool.Function.Name,
					Description: tool.Function.Description,
					Parameters:  tool.Function.Parameters,
				},
			}
		}
	}

	// Convert messages
	openRouterReq.Messages = make([]openRouterMessage, len(req.Messages))
	for i, msg := range req.Messages {
		openRouterMsg := openRouterMessage{
			Role:    msg.Role,
			Content: msg.Content,
		}

		if len(msg.ToolCalls) > 0 {
			openRouterMsg.ToolCalls = make([]openRouterToolCall, len(msg.ToolCalls))
			for j, toolCall := range msg.ToolCalls {
				openRouterMsg.ToolCalls[j] = openRouterToolCall{
					ID:   toolCall.ID,
					Type: toolCall.Type,
					Function: openRouterToolCallFunction{
						Name:      toolCall.Function.Name,
						Arguments: toolCall.Function.Arguments,
					},
				}
			}
		}

		if msg.ToolCallID != "" {
			openRouterMsg.ToolCallID = msg.ToolCallID
		}

		openRouterReq.Messages[i] = openRouterMsg
	}

	body, err := json.Marshal(openRouterReq)
	if err != nil {
		return nil, fmt.Errorf("openrouter: marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.buildURL("/chat/completions"), bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("openrouter: create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+p.config.APIKey)
	httpReq.Header.Set("Accept", "text/event-stream")

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("openrouter: request failed: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("openrouter: API error %d: %s", resp.StatusCode, string(body))
	}

	return &openRouterStreamReader{
		resp:     resp,
		scanner:  bufio.NewScanner(resp.Body),
		provider: "openrouter",
	}, nil
}

// Models returns the list of available models for OpenRouter
func (p *OpenRouterProvider) Models() []ModelInfo {
	// OpenRouter supports hundreds of models, but we'll list some popular ones
	// In practice, you might want to fetch this dynamically from OpenRouter's models API
	return []ModelInfo{
		{ID: "openai/gpt-5", Name: "GPT-5", Provider: "openai"},
		{ID: "openai/gpt-4o", Name: "GPT-4o", Provider: "openai"},
		{ID: "openai/gpt-4o-mini", Name: "GPT-4o Mini", Provider: "openai"},
		{ID: "anthropic/claude-3.5-sonnet", Name: "Claude 3.5 Sonnet", Provider: "anthropic"},
		{ID: "anthropic/claude-3.7-sonnet", Name: "Claude 3.7 Sonnet", Provider: "anthropic"},
		{ID: "google/gemini-pro-1.5", Name: "Gemini Pro 1.5", Provider: "google"},
		{ID: "meta-llama/llama-3.1-405b-instruct", Name: "Llama 3.1 405B", Provider: "meta"},
		{ID: "deepseek/deepseek-reasoner", Name: "DeepSeek Reasoner", Provider: "deepseek"},
	}
}

// Validate checks if the provider configuration is valid
func (p *OpenRouterProvider) Validate() error {
	if p.config.APIKey == "" {
		return fmt.Errorf("openrouter: API key is required")
	}
	return nil
}

// openRouterStreamReader implements StreamReader for OpenRouter streaming responses
type openRouterStreamReader struct {
	resp     *http.Response
	scanner  *bufio.Scanner
	provider string
	err      error
	done     bool
}

func (r *openRouterStreamReader) Read() (*StreamChunk, error) {
	if r.done {
		return &StreamChunk{Done: true, Provider: r.provider}, nil
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

		var streamResp openRouterResponse
		if err := json.Unmarshal([]byte(data), &streamResp); err != nil {
			continue
		}

		if len(streamResp.Choices) == 0 {
			continue
		}

		choice := streamResp.Choices[0]
		streamChunk := StreamChunk{Provider: r.provider}

		// Handle content streaming
		if choice.Delta.Content != "" {
			streamChunk.Type = ChunkTypeContent
			streamChunk.Content = choice.Delta.Content
			return &streamChunk, nil
		}

		// Handle reasoning streaming
		if choice.Delta.Reasoning != "" {
			streamChunk.Type = ChunkTypeReasoning
			streamChunk.Reasoning = &ReasoningChunk{
				Content: choice.Delta.Reasoning,
				Summary: choice.Delta.Reasoning,
			}
			return &streamChunk, nil
		}

		// Handle tool calls streaming
		if len(choice.Delta.ToolCalls) > 0 {
			streamChunk.Type = ChunkTypeToolCall
			streamChunk.ToolCalls = make([]ToolCall, len(choice.Delta.ToolCalls))
			for i, toolCall := range choice.Delta.ToolCalls {
				streamChunk.ToolCalls[i] = ToolCall{
					ID:   toolCall.ID,
					Type: toolCall.Type,
					Function: FunctionCall{
						Name:      toolCall.Function.Name,
						Arguments: toolCall.Function.Arguments,
					},
				}
			}
			return &streamChunk, nil
		}
	}

	r.done = true
	return &StreamChunk{Done: true, Provider: r.provider}, nil
}

func (r *openRouterStreamReader) Close() error {
	return r.resp.Body.Close()
}

func (r *openRouterStreamReader) Err() error {
	return r.err
}

// buildURL constructs the full URL for the given path
func (p *OpenRouterProvider) buildURL(path string) string {
	baseURL := p.config.BaseURL
	if baseURL == "" {
		baseURL = "https://openrouter.ai/api/v1"
	}
	return baseURL + path
}

func init() {
	RegisterProvider("openrouter", NewOpenRouterProvider)
}
