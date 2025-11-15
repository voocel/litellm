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
			ID: "qwen3-coder-plus", Provider: "qwen", Name: "Qwen Turbo", MaxTokens: 1000000,
			Capabilities: []string{"chat", "function_call", "reasoning"},
		},
		{
			ID: "qwen3-coder-flash", Provider: "qwen", Name: "Qwen Turbo", MaxTokens: 1000000,
			Capabilities: []string{"chat", "function_call", "reasoning"},
		},
		{
			ID: "qwen-plus", Provider: "qwen", Name: "Qwen Plus", MaxTokens: 32768,
			Capabilities: []string{"chat", "function_call", "reasoning"},
		},
		{
			ID: "qwen-max", Provider: "qwen", Name: "Qwen Max", MaxTokens: 8192,
			Capabilities: []string{"chat", "function_call", "reasoning"},
		},
		{
			ID: "qwen-max-longcontext", Provider: "qwen", Name: "Qwen Max Long Context", MaxTokens: 1000000,
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

	qwenReq := qwenRequest{
		Model: req.Model,
		Input: qwenInput{
			Messages: p.convertMessages(req.Messages),
		},
	}

	if req.MaxTokens != nil || req.Temperature != nil {
		qwenReq.Parameters = &qwenParameters{}

		if req.MaxTokens != nil {
			qwenReq.Parameters.MaxTokens = req.MaxTokens
		}
		if req.Temperature != nil {
			qwenReq.Parameters.Temperature = req.Temperature
		}
	}

	// Handle tools (Qwen uses different format)
	if len(req.Tools) > 0 {
		if qwenReq.Parameters == nil {
			qwenReq.Parameters = &qwenParameters{}
		}
		qwenReq.Parameters.Tools = p.convertTools(req.Tools)
	}

	if req.ResponseFormat != nil && req.ResponseFormat.Type == "json_object" {
		if qwenReq.Parameters == nil {
			qwenReq.Parameters = &qwenParameters{}
		}
		qwenReq.Parameters.ResultFormat = "message"
	}

	// Enable thinking for reasoning-capable models
	if p.isReasoningModel(req.Model) {
		qwenReq.EnableThinking = true
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
		return nil, fmt.Errorf("qwen: request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, err := io.ReadAll(resp.Body)
		if err != nil {
			return nil, fmt.Errorf("qwen: failed to read error response: %w", err)
		}
		return nil, fmt.Errorf("qwen: API error %d: %s", resp.StatusCode, string(body))
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

	// Build request (similar to Chat but with SSE)
	qwenReq := qwenRequest{
		Model: req.Model,
		Input: qwenInput{
			Messages: p.convertMessages(req.Messages),
		},
		Parameters: &qwenParameters{
			IncrementalOutput: true, // Enable streaming
		},
	}

	if req.MaxTokens != nil {
		qwenReq.Parameters.MaxTokens = req.MaxTokens
	}
	if req.Temperature != nil {
		qwenReq.Parameters.Temperature = req.Temperature
	}
	if len(req.Tools) > 0 {
		qwenReq.Parameters.Tools = p.convertTools(req.Tools)
	}

	// Enable thinking for reasoning models
	if p.isReasoningModel(req.Model) {
		qwenReq.EnableThinking = true
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
		return nil, fmt.Errorf("qwen: request failed: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		body, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			return nil, fmt.Errorf("qwen: failed to read stream error response: %w", err)
		}
		return nil, fmt.Errorf("qwen: API error %d: %s", resp.StatusCode, string(body))
	}

	return &qwenStreamReader{
		resp:     resp,
		scanner:  bufio.NewScanner(resp.Body),
		provider: "qwen",
	}, nil
}

// convertMessages converts standard messages to Qwen format
func (p *QwenProvider) convertMessages(messages []Message) []qwenMessage {
	qwenMessages := make([]qwenMessage, len(messages))
	for i, msg := range messages {
		qwenMessages[i] = qwenMessage{
			Role:    msg.Role,
			Content: msg.Content,
		}

		// Handle tool calls
		if len(msg.ToolCalls) > 0 {
			for _, toolCall := range msg.ToolCalls {
				qwenMessages[i].ToolCalls = append(qwenMessages[i].ToolCalls, qwenToolCall{
					ID:   toolCall.ID,
					Type: toolCall.Type,
					Function: qwenFunctionCall{
						Name:      toolCall.Function.Name,
						Arguments: toolCall.Function.Arguments,
					},
				})
			}
		}

		if msg.ToolCallID != "" {
			qwenMessages[i].ToolCallID = msg.ToolCallID
		}
	}
	return qwenMessages
}

// convertTools converts standard tools to Qwen format
func (p *QwenProvider) convertTools(tools []Tool) []qwenTool {
	qwenTools := make([]qwenTool, len(tools))
	for i, tool := range tools {
		qwenTools[i] = qwenTool{
			Type: tool.Type,
			Function: qwenFunction{
				Name:        tool.Function.Name,
				Description: tool.Function.Description,
				Parameters:  tool.Function.Parameters,
			},
		}
	}
	return qwenTools
}

// isReasoningModel checks if the model supports reasoning
func (p *QwenProvider) isReasoningModel(model string) bool {
	reasoningModels := []string{
		"qwen-plus", "qwen-max", "qwen-max-longcontext",
		"qwen2.5-72b-instruct", "qwen2.5-32b-instruct",
		"qwen2.5-coder-32b-instruct", "qwen2.5-math-72b-instruct",
		"qwen3-coder-plus", "qwen3-coder-flash",
	}
	for _, m := range reasoningModels {
		if m == model {
			return true
		}
	}
	return false
}

// Qwen API structures (DashScope format)
type qwenRequest struct {
	Model          string          `json:"model"`
	Input          qwenInput       `json:"input"`
	Parameters     *qwenParameters `json:"parameters,omitempty"`
	EnableThinking bool            `json:"enable_thinking,omitempty"`
}

type qwenInput struct {
	Messages []qwenMessage `json:"messages"`
}

type qwenParameters struct {
	MaxTokens         *int       `json:"max_tokens,omitempty"`
	Temperature       *float64   `json:"temperature,omitempty"`
	Tools             []qwenTool `json:"tools,omitempty"`
	ResultFormat      string     `json:"result_format,omitempty"`
	IncrementalOutput bool       `json:"incremental_output,omitempty"`
}

type qwenMessage struct {
	Role             string         `json:"role"`
	Content          string         `json:"content"`
	ToolCalls        []qwenToolCall `json:"tool_calls,omitempty"`
	ToolCallID       string         `json:"tool_call_id,omitempty"`
	ReasoningContent string         `json:"reasoning_content,omitempty"`
}

type qwenToolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"`
	Function qwenFunctionCall `json:"function"`
}

type qwenFunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type qwenTool struct {
	Type     string       `json:"type"`
	Function qwenFunction `json:"function"`
}

type qwenFunction struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Parameters  any    `json:"parameters"`
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
				continue // Skip malformed chunks
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
	Role             string         `json:"role,omitempty"`
	Content          string         `json:"content,omitempty"`
	ReasoningContent string         `json:"reasoning_content,omitempty"`
	ToolCalls        []qwenToolCall `json:"tool_calls,omitempty"`
}
