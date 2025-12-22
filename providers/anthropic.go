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
	RegisterBuiltin("anthropic", func(cfg ProviderConfig) Provider {
		return NewAnthropic(cfg)
	}, "https://api.anthropic.com")
}

// AnthropicProvider implements Anthropic Claude API integration
type AnthropicProvider struct {
	*BaseProvider
}

// NewAnthropic creates a new Anthropic provider
func NewAnthropic(config ProviderConfig) *AnthropicProvider {
	baseProvider := NewBaseProvider("anthropic", config)
	return &AnthropicProvider{
		BaseProvider: baseProvider,
	}
}

func (p *AnthropicProvider) SupportsModel(model string) bool {
	for _, m := range p.Models() {
		if m.ID == model {
			return true
		}
	}
	return false
}

func (p *AnthropicProvider) Models() []ModelInfo {
	return []ModelInfo{
		// Claude 4.5 family (200K context; 1M requires beta header)
		{
			ID: "claude-opus-4-5-20251101", Provider: "anthropic", Name: "Claude 4.5 Opus", ContextWindow: 200000, MaxOutputTokens: 32000,
			Capabilities: []string{"chat", "function_call", "vision", "extended_thinking"}},
		{
			ID: "claude-sonnet-4-5-20250929", Provider: "anthropic", Name: "Claude 4.5 Sonnet", ContextWindow: 200000, MaxOutputTokens: 64000,
			Capabilities: []string{"chat", "function_call", "vision", "extended_thinking"},
		},
		{
			ID: "claude-haiku-4-5-20251001", Provider: "anthropic", Name: "Claude 4.5 Haiku", ContextWindow: 200000, MaxOutputTokens: 64000,
			Capabilities: []string{"chat", "function_call", "vision", "extended_thinking"},
		},

		// Claude 4.1/4 family (200K context; 1M requires beta header)
		{
			ID: "claude-opus-4-1-20250805", Provider: "anthropic", Name: "Claude 4.1 Opus", ContextWindow: 200000, MaxOutputTokens: 32000,
			Capabilities: []string{"chat", "function_call", "vision", "extended_thinking"},
		},
		{
			ID: "claude-opus-4-20250522", Provider: "anthropic", Name: "Claude 4 Opus", ContextWindow: 200000, MaxOutputTokens: 32000,
			Capabilities: []string{"chat", "function_call", "vision", "extended_thinking"},
		},
		{
			ID: "claude-sonnet-4-20250522", Provider: "anthropic", Name: "Claude 4 Sonnet", ContextWindow: 200000, MaxOutputTokens: 64000,
			Capabilities: []string{"chat", "function_call", "vision", "extended_thinking"},
		},

		// Claude 3.7 / 3.5 family (200K context per docs)
		{
			ID: "claude-sonnet-3-7-20250219", Provider: "anthropic", Name: "Claude 3.7 Sonnet", ContextWindow: 200000, MaxOutputTokens: 16000,
			Capabilities: []string{"chat", "function_call", "vision"},
		},
		{
			ID: "claude-haiku-3-5-20241022", Provider: "anthropic", Name: "Claude 3.5 Haiku", ContextWindow: 200000, MaxOutputTokens: 8000,
			Capabilities: []string{"chat", "function_call", "vision"},
		},
	}
}

func (p *AnthropicProvider) Chat(ctx context.Context, req *Request) (*Response, error) {
	httpReq, err := p.buildHTTPRequest(ctx, req, false)
	if err != nil {
		return nil, err
	}

	resp, err := p.HTTPClient().Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, NewHTTPError("anthropic", resp.StatusCode, string(body))
	}

	var anthropicResp anthropicResponse
	if err := json.NewDecoder(resp.Body).Decode(&anthropicResp); err != nil {
		return nil, fmt.Errorf("anthropic: decode response: %w", err)
	}

	// Process response
	response := &Response{
		Usage: Usage{
			PromptTokens:             anthropicResp.Usage.InputTokens,
			CompletionTokens:         anthropicResp.Usage.OutputTokens,
			TotalTokens:              anthropicResp.Usage.InputTokens + anthropicResp.Usage.OutputTokens,
			CacheCreationInputTokens: anthropicResp.Usage.CacheCreationInputTokens,
			CacheReadInputTokens:     anthropicResp.Usage.CacheReadInputTokens,
		},
		Model:    anthropicResp.Model,
		Provider: "anthropic",
	}

	// Parse content and tool calls
	for _, content := range anthropicResp.Content {
		switch content.Type {
		case "text":
			if content.Text != "" {
				if response.Content != "" {
					response.Content += "\n\n"
				}
				response.Content += content.Text
			}
		case "thinking":
			if content.Thinking != "" {
				if response.Reasoning == nil {
					response.Reasoning = &ReasoningData{}
				}
				if response.Reasoning.Content != "" {
					response.Reasoning.Content += "\n\n"
				}
				response.Reasoning.Content += content.Thinking
				// Set summary from signature if available
				if content.Signature != "" {
					response.Reasoning.Summary = content.Signature
				}
			}
		case "tool_use":
			args, _ := json.Marshal(content.Input)
			response.ToolCalls = append(response.ToolCalls, ToolCall{
				ID:   content.ToolUseID,
				Type: "function",
				Function: FunctionCall{
					Name:      content.Name,
					Arguments: string(args),
				},
			})
		}
	}

	return response, nil
}

func (p *AnthropicProvider) Stream(ctx context.Context, req *Request) (StreamReader, error) {
	httpReq, err := p.buildHTTPRequest(ctx, req, true)
	if err != nil {
		return nil, err
	}

	resp, err := p.HTTPClient().Do(httpReq)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, NewHTTPError("anthropic", resp.StatusCode, string(body))
	}

	return &anthropicStreamReader{
		resp:     resp,
		scanner:  bufio.NewScanner(resp.Body),
		provider: "anthropic",
	}, nil
}

func (p *AnthropicProvider) buildHTTPRequest(ctx context.Context, req *Request, stream bool) (*http.Request, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}
	if err := p.BaseProvider.ValidateRequest(req); err != nil {
		return nil, err
	}

	maxTokens := 65536
	if req.MaxTokens != nil {
		maxTokens = *req.MaxTokens
	}

	anthropicReq := anthropicRequest{
		Model:         req.Model,
		MaxTokens:     maxTokens,
		Stream:        stream,
		Temperature:   req.Temperature,
		StopSequences: req.Stop,
		ToolChoice:    req.ToolChoice,
	}

	if len(req.Tools) > 0 {
		anthropicReq.Tools = make([]anthropicTool, 0, len(req.Tools))
		for _, tool := range req.Tools {
			inputSchema, err := p.convertToolParameters(tool)
			if err != nil {
				return nil, err
			}
			anthropicReq.Tools = append(anthropicReq.Tools, anthropicTool{
				Name:        tool.Function.Name,
				Description: tool.Function.Description,
				InputSchema: inputSchema,
			})
		}
	}

	anthropicReq.System, anthropicReq.Messages = p.convertMessages(req)

	body, err := json.Marshal(anthropicReq)
	if err != nil {
		return nil, fmt.Errorf("anthropic: marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.buildURL("/v1/messages"), bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("anthropic: create request: %w", err)
	}

	p.setHeaders(httpReq)
	return httpReq, nil
}

func (p *AnthropicProvider) convertToolParameters(tool Tool) (map[string]any, error) {
	if params, ok := tool.Function.Parameters.(map[string]any); ok {
		return params, nil
	}
	jsonBytes, err := json.Marshal(tool.Function.Parameters)
	if err != nil {
		return nil, fmt.Errorf("anthropic: failed to marshal tool parameters for '%s': %w", tool.Function.Name, err)
	}
	var inputSchema map[string]any
	if err := json.Unmarshal(jsonBytes, &inputSchema); err != nil {
		return nil, fmt.Errorf("anthropic: invalid tool parameters for '%s': %w", tool.Function.Name, err)
	}
	return inputSchema, nil
}

func (p *AnthropicProvider) convertMessages(req *Request) (any, []anthropicMessage) {
	var systemContents []anthropicContent
	var nonSystemMessages []Message

	for _, msg := range req.Messages {
		if msg.Role == "system" {
			systemContent := anthropicContent{Type: "text", Text: msg.Content}
			if msg.CacheControl != nil {
				systemContent.CacheControl = &anthropicCacheControl{Type: msg.CacheControl.Type}
			}
			systemContents = append(systemContents, systemContent)
		} else {
			nonSystemMessages = append(nonSystemMessages, msg)
		}
	}

	var system any
	if len(systemContents) == 1 && systemContents[0].CacheControl == nil {
		system = systemContents[0].Text
	} else if len(systemContents) > 0 {
		system = systemContents
	}

	messages := make([]anthropicMessage, len(nonSystemMessages))
	for i, msg := range nonSystemMessages {
		messages[i] = p.convertSingleMessage(msg, req.ResponseFormat, i == len(nonSystemMessages)-1)
	}

	return system, messages
}

func (p *AnthropicProvider) convertSingleMessage(msg Message, respFormat *ResponseFormat, isLast bool) anthropicMessage {
	anthropicMsg := anthropicMessage{Role: msg.Role}

	content := msg.Content
	if respFormat != nil && isLast && msg.Role == "user" {
		content = p.addStructuredOutputInstructions(content, respFormat)
	}

	if content != "" {
		textContent := anthropicContent{Type: "text", Text: content}
		if msg.CacheControl != nil {
			textContent.CacheControl = &anthropicCacheControl{Type: msg.CacheControl.Type}
		}
		anthropicMsg.Content = []anthropicContent{textContent}
	}

	for _, toolCall := range msg.ToolCalls {
		var input map[string]any
		if err := json.Unmarshal([]byte(toolCall.Function.Arguments), &input); err != nil {
			input = map[string]any{}
		}
		anthropicMsg.Content = append(anthropicMsg.Content, anthropicContent{
			Type:      "tool_use",
			ToolUseID: toolCall.ID,
			Name:      toolCall.Function.Name,
			Input:     input,
		})
	}

	if msg.ToolCallID != "" {
		anthropicMsg.Content = []anthropicContent{
			{Type: "tool_result", ToolUseID: msg.ToolCallID, Text: msg.Content},
		}
	}

	return anthropicMsg
}

func (p *AnthropicProvider) buildURL(endpoint string) string {
	baseURL := strings.TrimSuffix(p.Config().BaseURL, "/")
	if baseURL == "" {
		baseURL = "https://api.anthropic.com"
	}
	return baseURL + endpoint
}

func (p *AnthropicProvider) setHeaders(req *http.Request) {
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", p.Config().APIKey)
	req.Header.Set("anthropic-version", "2023-06-01") // Latest stable version
	// Add beta headers for prompt caching and extended thinking support
	req.Header.Set("anthropic-beta", "prompt-caching-2024-07-31,extended-thinking-2025-01-01")
}

// addStructuredOutputInstructions adds JSON formatting instructions for structured output
func (p *AnthropicProvider) addStructuredOutputInstructions(content string, format *ResponseFormat) string {
	switch format.Type {
	case "json_object":
		return content + "\n\nPlease respond with a valid JSON object only."
	case "json_schema":
		if format.JSONSchema != nil {
			schemaJSON, _ := json.Marshal(format.JSONSchema.Schema)
			return fmt.Sprintf("%s\n\nPlease respond with a valid JSON object that strictly follows this schema:\n%s\n\nRespond with JSON only, no additional text.",
				content, string(schemaJSON))
		}
		return content + "\n\nPlease respond with a valid JSON object only."
	default:
		return content
	}
}

// anthropicStreamReader implements streaming for Anthropic
type anthropicStreamReader struct {
	resp     *http.Response
	scanner  *bufio.Scanner
	provider string
	done     bool
}

func (r *anthropicStreamReader) Next() (*StreamChunk, error) {
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

		var chunk anthropicStreamChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			// Return error instead of silently ignoring malformed chunks
			return nil, fmt.Errorf("anthropic: failed to parse stream chunk: %w", err)
		}

		// Handle text content deltas
		if chunk.Type == "content_block_delta" && chunk.Delta.Type == "text_delta" {
			return &StreamChunk{
				Type:     "content",
				Content:  chunk.Delta.Text,
				Done:     false,
				Provider: r.provider,
			}, nil
		}

		// Handle thinking block start
		if chunk.Type == "content_block_start" && chunk.ContentBlock != nil && chunk.ContentBlock.Type == "thinking" {
			return &StreamChunk{
				Type:      "reasoning",
				Provider:  r.provider,
				Reasoning: &ReasoningChunk{Content: ""},
			}, nil
		}

		// Handle thinking content deltas (extended thinking)
		if chunk.Type == "content_block_delta" && chunk.Delta.Type == "thinking_delta" {
			return &StreamChunk{
				Type:      "reasoning",
				Content:   "", // Keep content empty for reasoning chunks
				Reasoning: &ReasoningChunk{Content: chunk.Delta.Text},
				Done:      false,
				Provider:  r.provider,
			}, nil
		}

		// Handle tool use deltas
		if chunk.Type == "content_block_delta" && chunk.Delta.Type == "input_json_delta" {
			return &StreamChunk{
				Type:     "tool_call_delta",
				Provider: r.provider,
				ToolCallDelta: &ToolCallDelta{
					Index:          chunk.Index,
					ArgumentsDelta: chunk.Delta.PartialJSON,
				},
			}, nil
		}

		// Handle tool use start
		if chunk.Type == "content_block_start" && chunk.ContentBlock != nil && chunk.ContentBlock.Type == "tool_use" {
			return &StreamChunk{
				Type:     "tool_call_delta",
				Provider: r.provider,
				ToolCallDelta: &ToolCallDelta{
					Index:        chunk.Index,
					ID:           chunk.ContentBlock.ID,
					Type:         "function",
					FunctionName: chunk.ContentBlock.Name,
				},
			}, nil
		}

		// Handle message completion
		if chunk.Type == "message_delta" && chunk.Delta.StopReason != "" {
			return &StreamChunk{
				FinishReason: chunk.Delta.StopReason,
				Provider:     r.provider,
			}, nil
		}
	}

	if err := r.scanner.Err(); err != nil {
		return nil, err
	}

	r.done = true
	return &StreamChunk{Done: true, Provider: r.provider}, nil
}

func (r *anthropicStreamReader) Close() error {
	return r.resp.Body.Close()
}

// Anthropic API request/response structures
type anthropicRequest struct {
	Model         string             `json:"model"`
	System        any                `json:"system,omitempty"` // System prompt (can be string or array of content blocks)
	MaxTokens     int                `json:"max_tokens"`
	Messages      []anthropicMessage `json:"messages"`
	Stream        bool               `json:"stream,omitempty"`
	Temperature   *float64           `json:"temperature,omitempty"`
	Tools         []anthropicTool    `json:"tools,omitempty"`
	ToolChoice    any                `json:"tool_choice,omitempty"`    // How the model should use the provided tools
	StopSequences []string           `json:"stop_sequences,omitempty"` // Custom text sequences that will cause the model to stop generating
}

type anthropicMessage struct {
	Role    string             `json:"role"`
	Content []anthropicContent `json:"content"`
}

type anthropicContent struct {
	Type         string                 `json:"type"`
	Text         string                 `json:"text,omitempty"`
	Thinking     string                 `json:"thinking,omitempty"`  // For extended thinking
	Signature    string                 `json:"signature,omitempty"` // For extended thinking
	ToolUseID    string                 `json:"tool_use_id,omitempty"`
	Name         string                 `json:"name,omitempty"`
	Input        map[string]any         `json:"input,omitempty"`
	CacheControl *anthropicCacheControl `json:"cache_control,omitempty"`
}

// anthropicCacheControl represents Anthropic's cache control structure
type anthropicCacheControl struct {
	Type string `json:"type"`
}

type anthropicTool struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	InputSchema map[string]any `json:"input_schema"`
}

type anthropicResponse struct {
	Content []anthropicContent `json:"content"`
	Usage   anthropicUsage     `json:"usage"`
	Model   string             `json:"model"`
}

type anthropicUsage struct {
	InputTokens              int `json:"input_tokens"`
	OutputTokens             int `json:"output_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens,omitempty"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens,omitempty"`
}

type anthropicStreamChunk struct {
	Type         string         `json:"type"`
	Index        int            `json:"index,omitempty"`
	Delta        anthropicDelta `json:"delta,omitempty"`
	Usage        anthropicUsage `json:"usage,omitempty"`
	ContentBlock *struct {
		Type string `json:"type"`
		ID   string `json:"id,omitempty"`
		Name string `json:"name,omitempty"`
	} `json:"content_block,omitempty"`
}

type anthropicDelta struct {
	Type        string `json:"type"`
	Text        string `json:"text,omitempty"`
	PartialJSON string `json:"partial_json,omitempty"`
	Name        string `json:"name,omitempty"`
	Input       string `json:"input,omitempty"`
	StopReason  string `json:"stop_reason,omitempty"`
}
