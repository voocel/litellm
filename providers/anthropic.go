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
		Model:        anthropicResp.Model,
		Provider:     "anthropic",
		FinishReason: NormalizeFinishReason(anthropicResp.StopReason),
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
			if isThinkingDisabled(req) {
				continue
			}
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
		resp:             resp,
		scanner:          bufio.NewScanner(resp.Body),
		provider:         "anthropic",
		includeReasoning: !isThinkingDisabled(req),
	}, nil
}

// ListModels returns available models for Anthropic.
func (p *AnthropicProvider) ListModels(ctx context.Context) ([]ModelInfo, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, "GET", p.buildURL("/v1/models"), nil)
	if err != nil {
		return nil, fmt.Errorf("anthropic: create models request: %w", err)
	}
	p.setModelHeaders(httpReq)

	resp, err := p.HTTPClient().Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, NewHTTPError("anthropic", resp.StatusCode, string(body))
	}

	var payload anthropicModelList
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return nil, fmt.Errorf("anthropic: decode models response: %w", err)
	}

	models := make([]ModelInfo, 0, len(payload.Data))
	for _, item := range payload.Data {
		name := item.DisplayName
		if name == "" {
			name = item.ID
		}
		models = append(models, ModelInfo{
			ID:       item.ID,
			Name:     name,
			Provider: "anthropic",
		})
	}

	return models, nil
}

func (p *AnthropicProvider) buildHTTPRequest(ctx context.Context, req *Request, stream bool) (*http.Request, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}
	if err := p.BaseProvider.ValidateExtra(req.Extra, nil); err != nil {
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

	thinking := normalizeThinking(req)
	if thinking.Type != "enabled" && thinking.Type != "disabled" {
		return nil, fmt.Errorf("anthropic: thinking type must be enabled or disabled")
	}
	if thinking.Type == "enabled" && thinking.BudgetTokens == nil {
		defaultBudget := 1024
		if maxTokens > 0 && maxTokens < defaultBudget {
			defaultBudget = maxTokens
		}
		thinking.BudgetTokens = &defaultBudget
	}
	anthropicReq.Thinking = thinking

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

func (p *AnthropicProvider) setModelHeaders(req *http.Request) {
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", p.Config().APIKey)
	req.Header.Set("anthropic-version", "2023-06-01")
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
	resp             *http.Response
	scanner          *bufio.Scanner
	provider         string
	model            string
	includeReasoning bool
	done             bool
	usage            *Usage // Accumulate usage from message_start and message_delta
}

func (r *anthropicStreamReader) Next() (*StreamChunk, error) {
	if r.done {
		return &StreamChunk{Done: true, Provider: r.provider, Model: r.model, Usage: r.usage}, nil
	}

	for r.scanner.Scan() {
		line := r.scanner.Text()

		// Handle event type line
		if strings.HasPrefix(line, "event: ") {
			continue
		}

		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimPrefix(line, "data: ")

		var chunk anthropicStreamChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			return nil, fmt.Errorf("anthropic: failed to parse stream chunk: %w", err)
		}

		switch chunk.Type {
		// === Message Lifecycle Events ===
		case "message_start":
			// Initialize usage from message_start (may be nil in extended thinking mode)
			if chunk.Message != nil {
				r.model = chunk.Message.Model
				if chunk.Message.Usage != nil {
					r.usage = &Usage{
						PromptTokens:             chunk.Message.Usage.InputTokens,
						CompletionTokens:         chunk.Message.Usage.OutputTokens,
						TotalTokens:              chunk.Message.Usage.InputTokens + chunk.Message.Usage.OutputTokens,
						CacheCreationInputTokens: chunk.Message.Usage.CacheCreationInputTokens,
						CacheReadInputTokens:     chunk.Message.Usage.CacheReadInputTokens,
					}
				}
			}
			continue

		case "message_delta":
			// Update usage (cumulative) - may contain full usage info
			if chunk.Usage != nil {
				if r.usage == nil {
					r.usage = &Usage{}
				}
				// message_delta usage is cumulative, so update all available fields
				if chunk.Usage.InputTokens > 0 {
					r.usage.PromptTokens = chunk.Usage.InputTokens
				}
				if chunk.Usage.OutputTokens > 0 {
					r.usage.CompletionTokens = chunk.Usage.OutputTokens
				}
				if chunk.Usage.CacheCreationInputTokens > 0 {
					r.usage.CacheCreationInputTokens = chunk.Usage.CacheCreationInputTokens
				}
				if chunk.Usage.CacheReadInputTokens > 0 {
					r.usage.CacheReadInputTokens = chunk.Usage.CacheReadInputTokens
				}
				r.usage.TotalTokens = r.usage.PromptTokens + r.usage.CompletionTokens
			}
			// Return finish reason
			if chunk.Delta != nil && chunk.Delta.StopReason != "" {
				return &StreamChunk{
					FinishReason: NormalizeFinishReason(chunk.Delta.StopReason),
					Provider:     r.provider,
					Model:        r.model,
				}, nil
			}
			continue

		case "message_stop":
			r.done = true
			return &StreamChunk{
				Done:     true,
				Provider: r.provider,
				Model:    r.model,
				Usage:    r.usage,
			}, nil

		// === Content Block Events ===
		case "content_block_start":
			if chunk.ContentBlock == nil {
				continue
			}
			switch chunk.ContentBlock.Type {
			case "thinking":
				if !r.includeReasoning {
					continue
				}
				return &StreamChunk{
					Type:      "reasoning",
					Provider:  r.provider,
					Model:     r.model,
					Reasoning: &ReasoningChunk{Content: ""},
				}, nil
			case "tool_use":
				return &StreamChunk{
					Type:     "tool_call_delta",
					Provider: r.provider,
					Model:    r.model,
					ToolCallDelta: &ToolCallDelta{
						Index:        chunk.Index,
						ID:           chunk.ContentBlock.ID,
						Type:         "function",
						FunctionName: chunk.ContentBlock.Name,
					},
				}, nil
			}
			continue

		case "content_block_delta":
			if chunk.Delta == nil {
				continue
			}
			switch chunk.Delta.Type {
			case "text_delta":
				return &StreamChunk{
					Type:     "content",
					Content:  chunk.Delta.Text,
					Provider: r.provider,
					Model:    r.model,
				}, nil
			case "thinking_delta":
				if !r.includeReasoning {
					continue
				}
				return &StreamChunk{
					Type:      "reasoning",
					Reasoning: &ReasoningChunk{Content: chunk.Delta.Thinking},
					Provider:  r.provider,
					Model:     r.model,
				}, nil
			case "signature_delta":
				// Signature for extended thinking verification, skip for now
				continue
			case "input_json_delta":
				return &StreamChunk{
					Type:     "tool_call_delta",
					Provider: r.provider,
					Model:    r.model,
					ToolCallDelta: &ToolCallDelta{
						Index:          chunk.Index,
						ArgumentsDelta: chunk.Delta.PartialJSON,
					},
				}, nil
			}
			continue

		case "content_block_stop":
			// Content block finished, continue processing
			continue

		// === Other Events ===
		case "ping":
			continue

		case "error":
			if chunk.Error != nil {
				return nil, fmt.Errorf("anthropic: stream error: [%s] %s", chunk.Error.Type, chunk.Error.Message)
			}
			return nil, fmt.Errorf("anthropic: unknown stream error")
		}
	}

	if err := r.scanner.Err(); err != nil {
		return nil, fmt.Errorf("anthropic: stream read error: %w", err)
	}

	r.done = true
	return &StreamChunk{Done: true, Provider: r.provider, Model: r.model, Usage: r.usage}, nil
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
	Thinking      *ThinkingConfig    `json:"thinking,omitempty"`
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
	Content    []anthropicContent `json:"content"`
	Usage      anthropicUsage     `json:"usage"`
	Model      string             `json:"model"`
	StopReason string             `json:"stop_reason"`
}

type anthropicModelList struct {
	Data []anthropicModelInfo `json:"data"`
}

type anthropicModelInfo struct {
	ID          string `json:"id"`
	DisplayName string `json:"display_name,omitempty"`
}

type anthropicUsage struct {
	InputTokens              int `json:"input_tokens"`
	OutputTokens             int `json:"output_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens,omitempty"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens,omitempty"`
}

type anthropicStreamChunk struct {
	Type         string                  `json:"type"`
	Index        int                     `json:"index,omitempty"`
	Delta        *anthropicDelta         `json:"delta,omitempty"`
	Usage        *anthropicUsage         `json:"usage,omitempty"`
	Message      *anthropicStreamMessage `json:"message,omitempty"`
	Error        *anthropicStreamError   `json:"error,omitempty"`
	ContentBlock *struct {
		Type string `json:"type"`
		ID   string `json:"id,omitempty"`
		Name string `json:"name,omitempty"`
	} `json:"content_block,omitempty"`
}

type anthropicStreamMessage struct {
	ID    string          `json:"id,omitempty"`
	Model string          `json:"model,omitempty"`
	Usage *anthropicUsage `json:"usage,omitempty"`
}

type anthropicStreamError struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

type anthropicDelta struct {
	Type        string `json:"type"`
	Text        string `json:"text,omitempty"`
	Thinking    string `json:"thinking,omitempty"`
	Signature   string `json:"signature,omitempty"`
	PartialJSON string `json:"partial_json,omitempty"`
	Name        string `json:"name,omitempty"`
	Input       string `json:"input,omitempty"`
	StopReason  string `json:"stop_reason,omitempty"`
}
