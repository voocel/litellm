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
)

// AnthropicProvider implements the Provider interface for Anthropic
type AnthropicProvider struct {
	*BaseProvider
}

// NewAnthropicProvider creates a new Anthropic provider
func NewAnthropicProvider(config ProviderConfig) Provider {
	return &AnthropicProvider{
		BaseProvider: NewBaseProvider("anthropic", config),
	}
}

// Models returns the list of supported models
func (p *AnthropicProvider) Models() []ModelInfo {
	return []ModelInfo{
		{
			ID: "claude-4-sonnet", Provider: "anthropic", Name: "Claude 4 Sonnet", MaxTokens: 2000000,
			Capabilities: []ModelCapability{CapabilityChat, CapabilityFunctionCall, CapabilityVision},
		},
		{
			ID: "claude-4-opus", Provider: "anthropic", Name: "Claude 4 opus", MaxTokens: 2000000,
			Capabilities: []ModelCapability{CapabilityChat, CapabilityFunctionCall},
		},
		{
			ID: "claude-3.7-sonnet", Provider: "anthropic", Name: "Claude 3.7 Sonnet", MaxTokens: 2000000,
			Capabilities: []ModelCapability{CapabilityChat, CapabilityFunctionCall, CapabilityVision},
		},
	}
}

// Anthropic API request/response structures
type anthropicRequest struct {
	Model       string             `json:"model"`
	MaxTokens   int                `json:"max_tokens"`
	Messages    []anthropicMessage `json:"messages"`
	Stream      bool               `json:"stream,omitempty"`
	Temperature *float64           `json:"temperature,omitempty"`
	Tools       []anthropicTool    `json:"tools,omitempty"`
}

type anthropicMessage struct {
	Role    string             `json:"role"`
	Content []anthropicContent `json:"content"`
}

type anthropicContent struct {
	Type      string                 `json:"type"`
	Text      string                 `json:"text,omitempty"`
	ToolUseID string                 `json:"tool_use_id,omitempty"`
	Name      string                 `json:"name,omitempty"`
	Input     map[string]interface{} `json:"input,omitempty"`
}

type anthropicTool struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	InputSchema map[string]interface{} `json:"input_schema"`
}

type anthropicResponse struct {
	Content []anthropicContent `json:"content"`
	Usage   anthropicUsage     `json:"usage"`
	Model   string             `json:"model"`
}

type anthropicUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
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
}

func (p *AnthropicProvider) Complete(ctx context.Context, req *Request) (*Response, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}

	modelName := req.Model

	anthropicReq := anthropicRequest{
		Model:       modelName,
		MaxTokens:   4096,
		Stream:      false,
		Temperature: req.Temperature,
	}

	if req.MaxTokens != nil {
		anthropicReq.MaxTokens = *req.MaxTokens
	}

	// Convert tool definitions
	if len(req.Tools) > 0 {
		anthropicReq.Tools = make([]anthropicTool, len(req.Tools))
		for i, tool := range req.Tools {
			var inputSchema map[string]interface{}
			if params, ok := tool.Function.Parameters.(map[string]interface{}); ok {
				inputSchema = params
			} else {
				// If not a map, try to convert through JSON
				if jsonBytes, err := json.Marshal(tool.Function.Parameters); err == nil {
					json.Unmarshal(jsonBytes, &inputSchema)
				}
			}

			anthropicReq.Tools[i] = anthropicTool{
				Name:        tool.Function.Name,
				Description: tool.Function.Description,
				InputSchema: inputSchema,
			}
		}
	}

	// Convert messages
	anthropicReq.Messages = make([]anthropicMessage, len(req.Messages))
	for i, msg := range req.Messages {
		anthropicMsg := anthropicMessage{
			Role: msg.Role,
		}

		content := msg.Content
		// Handle structured output by adding instructions to the last user message
		if req.ResponseFormat != nil && i == len(req.Messages)-1 && msg.Role == "user" {
			content = p.addStructuredOutputInstructions(content, req.ResponseFormat)
		}

		if content != "" {
			anthropicMsg.Content = []anthropicContent{
				{Type: "text", Text: content},
			}
		}

		// Handle tool calls
		if len(msg.ToolCalls) > 0 {
			for _, toolCall := range msg.ToolCalls {
				var input map[string]interface{}
				json.Unmarshal([]byte(toolCall.Function.Arguments), &input)
				anthropicMsg.Content = append(anthropicMsg.Content, anthropicContent{
					Type:      "tool_use",
					ToolUseID: toolCall.ID,
					Name:      toolCall.Function.Name,
					Input:     input,
				})
			}
		}

		// Handle tool responses
		if msg.ToolCallID != "" {
			anthropicMsg.Content = []anthropicContent{
				{Type: "tool_result", ToolUseID: msg.ToolCallID, Text: msg.Content},
			}
		}

		anthropicReq.Messages[i] = anthropicMsg
	}

	body, err := json.Marshal(anthropicReq)
	if err != nil {
		return nil, fmt.Errorf("anthropic: marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.config.BaseURL+"/v1/messages", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("anthropic: create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-api-key", p.config.APIKey)
	httpReq.Header.Set("anthropic-version", "2023-06-01")

	resp, err := p.HTTPClient().Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("anthropic: request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("anthropic: API error %d: %s", resp.StatusCode, string(body))
	}

	var anthropicResp anthropicResponse
	if err := json.NewDecoder(resp.Body).Decode(&anthropicResp); err != nil {
		return nil, fmt.Errorf("anthropic: decode response: %w", err)
	}

	// Process response
	response := &Response{
		Usage: Usage{
			PromptTokens:     anthropicResp.Usage.InputTokens,
			CompletionTokens: anthropicResp.Usage.OutputTokens,
			TotalTokens:      anthropicResp.Usage.InputTokens + anthropicResp.Usage.OutputTokens,
		},
		Model:    anthropicResp.Model,
		Provider: "anthropic",
	}

	// Parse content and tool calls
	for _, content := range anthropicResp.Content {
		switch content.Type {
		case "text":
			response.Content += content.Text
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
	if err := p.Validate(); err != nil {
		return nil, err
	}

	modelName := req.Model

	anthropicReq := anthropicRequest{
		Model:       modelName,
		MaxTokens:   4096,
		Messages:    make([]anthropicMessage, len(req.Messages)),
		Stream:      true,
		Temperature: req.Temperature,
	}

	if req.MaxTokens != nil {
		anthropicReq.MaxTokens = *req.MaxTokens
	}

	// Convert tool definitions (same as Complete method)
	if len(req.Tools) > 0 {
		anthropicReq.Tools = make([]anthropicTool, len(req.Tools))
		for i, tool := range req.Tools {
			var inputSchema map[string]interface{}
			if params, ok := tool.Function.Parameters.(map[string]interface{}); ok {
				inputSchema = params
			} else {
				if jsonBytes, err := json.Marshal(tool.Function.Parameters); err == nil {
					json.Unmarshal(jsonBytes, &inputSchema)
				}
			}

			anthropicReq.Tools[i] = anthropicTool{
				Name:        tool.Function.Name,
				Description: tool.Function.Description,
				InputSchema: inputSchema,
			}
		}
	}

	// Convert messages (same as Complete method)
	anthropicReq.Messages = make([]anthropicMessage, len(req.Messages))
	for i, msg := range req.Messages {
		anthropicMsg := anthropicMessage{
			Role: msg.Role,
		}

		content := msg.Content
		// Handle structured output by adding instructions to the last user message
		if req.ResponseFormat != nil && i == len(req.Messages)-1 && msg.Role == "user" {
			content = p.addStructuredOutputInstructions(content, req.ResponseFormat)
		}

		if content != "" {
			anthropicMsg.Content = []anthropicContent{
				{Type: "text", Text: content},
			}
		}

		// Handle tool calls
		if len(msg.ToolCalls) > 0 {
			for _, toolCall := range msg.ToolCalls {
				var input map[string]interface{}
				json.Unmarshal([]byte(toolCall.Function.Arguments), &input)
				anthropicMsg.Content = append(anthropicMsg.Content, anthropicContent{
					Type:      "tool_use",
					ToolUseID: toolCall.ID,
					Name:      toolCall.Function.Name,
					Input:     input,
				})
			}
		}

		// Handle tool responses
		if msg.ToolCallID != "" {
			anthropicMsg.Content = []anthropicContent{
				{Type: "tool_result", ToolUseID: msg.ToolCallID, Text: msg.Content},
			}
		}

		anthropicReq.Messages[i] = anthropicMsg
	}

	body, err := json.Marshal(anthropicReq)
	if err != nil {
		return nil, fmt.Errorf("anthropic: marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.config.BaseURL+"/v1/messages", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("anthropic: create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-api-key", p.config.APIKey)
	httpReq.Header.Set("anthropic-version", "2023-06-01")

	resp, err := p.HTTPClient().Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("anthropic: request failed: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("anthropic: API error %d: %s", resp.StatusCode, string(body))
	}

	return &anthropicStreamReader{
		resp:     resp,
		scanner:  bufio.NewScanner(resp.Body),
		provider: "anthropic",
	}, nil
}

// anthropicStreamReader implements StreamReader for Anthropic
type anthropicStreamReader struct {
	resp     *http.Response
	scanner  *bufio.Scanner
	provider string
	err      error
	done     bool
}

func (r *anthropicStreamReader) Read() (*StreamChunk, error) {
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
			continue
		}

		if chunk.Type == "content_block_delta" && chunk.Delta.Type == "text_delta" {
			return &StreamChunk{
				Type:     ChunkTypeContent,
				Content:  chunk.Delta.Text,
				Done:     false,
				Provider: r.provider,
			}, nil
		}

		// Handle tool use deltas
		if chunk.Type == "content_block_delta" && chunk.Delta.Type == "input_json_delta" {
			return &StreamChunk{
				Type:     ChunkTypeToolCallDelta,
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
				Type:     ChunkTypeToolCallDelta,
				Provider: r.provider,
				ToolCallDelta: &ToolCallDelta{
					Index:        chunk.Index,
					ID:           chunk.ContentBlock.ID,
					Type:         "function",
					FunctionName: chunk.ContentBlock.Name,
				},
			}, nil
		}
	}

	if err := r.scanner.Err(); err != nil {
		r.err = err
		return nil, err
	}

	r.done = true
	return &StreamChunk{Done: true, Provider: r.provider}, nil
}

func (r *anthropicStreamReader) Close() error {
	return r.resp.Body.Close()
}

func (r *anthropicStreamReader) Err() error {
	return r.err
}

// addStructuredOutputInstructions adds JSON formatting instructions for structured output
func (p *AnthropicProvider) addStructuredOutputInstructions(content string, format *ResponseFormat) string {
	switch format.Type {
	case ResponseFormatJSONObject:
		return content + "\n\nPlease respond with a valid JSON object only."
	case ResponseFormatJSONSchema:
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

func init() {
	RegisterProvider("anthropic", NewAnthropicProvider)
}
