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
	RegisterBuiltin("qwen", func(cfg ProviderConfig) Provider {
		return NewQwen(cfg)
	}, "https://dashscope.aliyuncs.com/api/v1")
}

// QwenProvider implements Qwen/DashScope API integration
type QwenProvider struct {
	*BaseProvider
}

// NewQwen creates a new Qwen provider
func NewQwen(config ProviderConfig) *QwenProvider {
	baseProvider := NewBaseProvider("qwen", config)

	return &QwenProvider{
		BaseProvider: baseProvider,
	}
}

func (p *QwenProvider) SupportsModel(model string) bool {
	for _, m := range p.Models() {
		if m.ID == model {
			return true
		}
	}
	return false
}

func (p *QwenProvider) Models() []ModelInfo {
	return []ModelInfo{
		{
			ID: "qwen3-coder-plus", Provider: "qwen", Name: "Qwen Turbo", ContextWindow: 1000000,
			Capabilities: []string{"chat", "function_call", "reasoning"},
		},
		{
			ID: "qwen3-coder-flash", Provider: "qwen", Name: "Qwen Turbo", ContextWindow: 1000000,
			Capabilities: []string{"chat", "function_call", "reasoning"},
		},
		{
			ID: "qwen-plus", Provider: "qwen", Name: "Qwen Plus", ContextWindow: 32768,
			Capabilities: []string{"chat", "function_call", "reasoning"},
		},
		{
			ID: "qwen-max", Provider: "qwen", Name: "Qwen Max", ContextWindow: 8192,
			Capabilities: []string{"chat", "function_call", "reasoning"},
		},
		{
			ID: "qwen-max-longcontext", Provider: "qwen", Name: "Qwen Max Long Context", ContextWindow: 1000000,
			Capabilities: []string{"chat", "function_call", "reasoning"},
		},
	}
}

func (p *QwenProvider) Chat(ctx context.Context, req *Request) (*Response, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}

	if err := p.BaseProvider.ValidateRequest(req); err != nil {
		return nil, err
	}

	// Build Qwen request using map[string]any for flexibility
	qwenReq := map[string]any{
		"model": req.Model,
		"input": map[string]any{
			"messages": ConvertMessages(req.Messages),
		},
	}

	// Build parameters if needed
	params := make(map[string]any)
	if req.MaxTokens != nil {
		params["max_tokens"] = *req.MaxTokens
	}
	if req.Temperature != nil {
		params["temperature"] = *req.Temperature
	}
	if len(req.Tools) > 0 {
		params["tools"] = ConvertTools(req.Tools)
	}
	if req.ResponseFormat != nil && req.ResponseFormat.Type == "json_object" {
		params["result_format"] = "message"
	}

	if len(params) > 0 {
		qwenReq["parameters"] = params
	}

	// Enable thinking for reasoning-capable models
	if p.hasCapability(req.Model, "reasoning") {
		qwenReq["enable_thinking"] = true
	}

	body, err := json.Marshal(qwenReq)
	if err != nil {
		return nil, fmt.Errorf("qwen: marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/services/aigc/text-generation/generation", p.Config().BaseURL)
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("qwen: create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+p.Config().APIKey)

	resp, err := p.HTTPClient().Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body))
	}

	var qwenResp qwenResponse
	if err := json.NewDecoder(resp.Body).Decode(&qwenResp); err != nil {
		return nil, fmt.Errorf("qwen: decode response: %w", err)
	}

	response := &Response{
		Model:    req.Model,
		Provider: "qwen",
	}

	if qwenResp.Usage != nil {
		response.Usage = Usage{
			PromptTokens:     qwenResp.Usage.InputTokens,
			CompletionTokens: qwenResp.Usage.OutputTokens,
			TotalTokens:      qwenResp.Usage.TotalTokens,
		}
	}

	if qwenResp.Output != nil {
		if len(qwenResp.Output.Choices) > 0 {
			choice := qwenResp.Output.Choices[0]
			response.Content = choice.Message.Content
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

			// Handle reasoning content
			if choice.Message.ReasoningContent != "" {
				response.Reasoning = &ReasoningData{
					Content:    choice.Message.ReasoningContent,
					Summary:    "Qwen reasoning process",
					TokensUsed: 0, // Qwen doesn't separate reasoning tokens
				}
			}
		}
	}

	return response, nil
}

func (p *QwenProvider) Stream(ctx context.Context, req *Request) (StreamReader, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}

	// Build request using map[string]any for flexibility
	params := map[string]any{
		"incremental_output": true, // Enable streaming
	}
	if req.MaxTokens != nil {
		params["max_tokens"] = *req.MaxTokens
	}
	if req.Temperature != nil {
		params["temperature"] = *req.Temperature
	}
	if len(req.Tools) > 0 {
		params["tools"] = ConvertTools(req.Tools)
	}

	qwenReq := map[string]any{
		"model": req.Model,
		"input": map[string]any{
			"messages": ConvertMessages(req.Messages),
		},
		"parameters": params,
	}

	// Enable thinking for reasoning models
	if p.hasCapability(req.Model, "reasoning") {
		qwenReq["enable_thinking"] = true
	}

	body, err := json.Marshal(qwenReq)
	if err != nil {
		return nil, fmt.Errorf("qwen: marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/services/aigc/text-generation/generation", p.Config().BaseURL)
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("qwen: create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+p.Config().APIKey)
	httpReq.Header.Set("Accept", "text/event-stream")
	httpReq.Header.Set("X-DashScope-SSE", "enable")

	resp, err := p.HTTPClient().Do(httpReq)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body))
	}

	return &qwenStreamReader{
		resp:     resp,
		scanner:  bufio.NewScanner(resp.Body),
		provider: "qwen",
	}, nil
}

// hasCapability checks if a model has a specific capability
func (p *QwenProvider) hasCapability(modelID, capability string) bool {
	for _, m := range p.Models() {
		if m.ID == modelID {
			for _, c := range m.Capabilities {
				if c == capability {
					return true
				}
			}
			return false
		}
	}
	return false
}

// Qwen API structures (DashScope format)
type qwenMessage struct {
	Role             string           `json:"role"`
	Content          string           `json:"content"`
	ToolCalls        []openaiToolCall `json:"tool_calls,omitempty"`
	ToolCallID       string           `json:"tool_call_id,omitempty"`
	ReasoningContent string           `json:"reasoning_content,omitempty"`
}

type qwenResponse struct {
	Output    *qwenOutput `json:"output,omitempty"`
	Usage     *qwenUsage  `json:"usage,omitempty"`
	RequestID string      `json:"request_id,omitempty"`
}

type qwenOutput struct {
	Text    string       `json:"text,omitempty"`
	Choices []qwenChoice `json:"choices,omitempty"`
}

type qwenChoice struct {
	Message      qwenMessage `json:"message"`
	FinishReason string      `json:"finish_reason"`
}

type qwenUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

// qwenStreamReader implements streaming for Qwen
type qwenStreamReader struct {
	resp        *http.Response
	scanner     *bufio.Scanner
	provider    string
	done        bool
	lastContent string
}

func (r *qwenStreamReader) Next() (*StreamChunk, error) {
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
		if strings.HasPrefix(line, "data:") {
			data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
			if data == "[DONE]" {
				r.done = true
				return &StreamChunk{Done: true, Provider: r.provider}, nil
			}

			var streamResp qwenStreamResponse
			if err := json.Unmarshal([]byte(data), &streamResp); err != nil {
				// Return error instead of silently ignoring malformed chunks
				return nil, fmt.Errorf("qwen: failed to parse stream chunk: %w", err)
			}

			// Convert to StreamChunk
			if streamResp.Output != nil && len(streamResp.Output.Choices) > 0 {
				choice := streamResp.Output.Choices[0]
				chunk := &StreamChunk{
					Provider: r.provider,
				}

				currentContent := choice.Message.Content
				if currentContent != r.lastContent {
					deltaContent := ""
					if len(currentContent) > len(r.lastContent) && strings.HasPrefix(currentContent, r.lastContent) {
						deltaContent = currentContent[len(r.lastContent):]
					} else {
						deltaContent = currentContent
					}

					if deltaContent != "" {
						chunk.Type = "content"
						chunk.Content = deltaContent
						r.lastContent = currentContent
					}
				}

				// Handle reasoning content
				if choice.Message.ReasoningContent != "" {
					chunk.Type = "reasoning"
					chunk.Reasoning = &ReasoningChunk{
						Content: choice.Message.ReasoningContent,
						Summary: "Qwen reasoning process",
					}
				}

				if len(choice.Message.ToolCalls) > 0 {
					chunk.Type = "tool_call_delta"
					toolCall := choice.Message.ToolCalls[0]
					chunk.ToolCallDelta = &ToolCallDelta{
						Index:          0,
						ID:             toolCall.ID,
						Type:           toolCall.Type,
						FunctionName:   toolCall.Function.Name,
						ArgumentsDelta: toolCall.Function.Arguments,
					}
				}

				if choice.FinishReason != "" && choice.FinishReason != "null" {
					chunk.FinishReason = choice.FinishReason
					chunk.Done = true
					r.done = true
				}

				if chunk.Content != "" || chunk.Reasoning != nil || len(choice.Message.ToolCalls) > 0 || chunk.Done {
					return chunk, nil
				}
			}
		}
	}

	if err := r.scanner.Err(); err != nil {
		return nil, err
	}

	r.done = true
	return &StreamChunk{Done: true, Provider: r.provider}, nil
}

func (r *qwenStreamReader) Close() error {
	return r.resp.Body.Close()
}

// Streaming response structures
type qwenStreamResponse struct {
	Output    *qwenStreamOutput `json:"output,omitempty"`
	Usage     *qwenUsage        `json:"usage,omitempty"`
	RequestID string            `json:"request_id,omitempty"`
}

type qwenStreamOutput struct {
	Choices []qwenStreamChoice `json:"choices,omitempty"`
}

type qwenStreamChoice struct {
	Message      qwenMessage     `json:"message"` // Qwen uses message, not delta
	Delta        qwenStreamDelta `json:"delta"`   // Keep for backward compatibility
	FinishReason string          `json:"finish_reason"`
}

type qwenStreamDelta struct {
	Role             string           `json:"role,omitempty"`
	Content          string           `json:"content,omitempty"`
	ReasoningContent string           `json:"reasoning_content,omitempty"`
	ToolCalls        []openaiToolCall `json:"tool_calls,omitempty"`
}
