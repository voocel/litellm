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

// GLMProvider implements the Provider interface for ZhiPu GLM-4.5 models
type GLMProvider struct {
	*BaseProvider
}

// NewGLMProvider creates a new GLM provider instance
func NewGLMProvider(config ProviderConfig) Provider {
	return &GLMProvider{
		BaseProvider: NewBaseProvider("glm", config),
	}
}

// Models returns the list of supported GLM models
func (p *GLMProvider) Models() []ModelInfo {
	return []ModelInfo{
		{
			ID: "glm-4.5", Provider: "glm", Name: "GLM-4.5", MaxTokens: 128000,
			Capabilities: []ModelCapability{CapabilityChat, CapabilityFunctionCall, CapabilityCode, CapabilityReasoning},
		},
		{
			ID: "glm-4.5-air", Provider: "glm", Name: "GLM-4.5 Air", MaxTokens: 128000,
			Capabilities: []ModelCapability{CapabilityChat, CapabilityFunctionCall, CapabilityCode, CapabilityReasoning},
		},
		{
			ID: "glm-4.5-flash", Provider: "glm", Name: "GLM-4.5 Flash", MaxTokens: 128000,
			Capabilities: []ModelCapability{CapabilityChat, CapabilityFunctionCall, CapabilityCode},
		},
		{
			ID: "glm-4", Provider: "glm", Name: "GLM-4", MaxTokens: 128000,
			Capabilities: []ModelCapability{CapabilityChat, CapabilityFunctionCall, CapabilityCode},
		},
		{
			ID: "glm-4-flash", Provider: "glm", Name: "GLM-4 Flash", MaxTokens: 128000,
			Capabilities: []ModelCapability{CapabilityChat, CapabilityFunctionCall, CapabilityCode},
		},
		{
			ID: "glm-4-air", Provider: "glm", Name: "GLM-4 Air", MaxTokens: 128000,
			Capabilities: []ModelCapability{CapabilityChat, CapabilityFunctionCall, CapabilityCode},
		},
		{
			ID: "glm-4-airx", Provider: "glm", Name: "GLM-4 AirX", MaxTokens: 128000,
			Capabilities: []ModelCapability{CapabilityChat, CapabilityFunctionCall, CapabilityCode},
		},
	}
}

// GLM thinking configuration
type glmThinking struct {
	Type string `json:"type"` // "enabled" or "disabled"
}

// GLM request extends openaiRequest with GLM-specific parameters
type glmRequest struct {
	openaiRequest
	// GLM specific parameters
	DoSample  *bool                  `json:"do_sample,omitempty"`
	RequestID string                 `json:"request_id,omitempty"`
	UserID    string                 `json:"user_id,omitempty"`
	Thinking  *glmThinking           `json:"thinking,omitempty"`
	Extra     map[string]interface{} `json:"extra,omitempty"`
}

// glmMessage extends openaiMessage with reasoning content
type glmMessage struct {
	openaiMessage
	ReasoningContent string `json:"reasoning_content,omitempty"` // GLM thinking process content
}

// glmChoice extends openaiChoice with GLM message
type glmChoice struct {
	Index        int        `json:"index"`
	Message      glmMessage `json:"message"`
	FinishReason string     `json:"finish_reason"`
}

// GLM response format
type glmResponse struct {
	ID      string      `json:"id"`
	Object  string      `json:"object"`
	Created int64       `json:"created"`
	Model   string      `json:"model"`
	Choices []glmChoice `json:"choices"`
	Usage   openaiUsage `json:"usage"`
}

// glmStreamDelta extends openaiDelta with reasoning content
type glmStreamDelta struct {
	openaiDelta
	ReasoningContent string `json:"reasoning_content,omitempty"` // GLM thinking process content
}

// glmStreamChoice for streaming with reasoning
type glmStreamChoice struct {
	Index        int            `json:"index"`
	Delta        glmStreamDelta `json:"delta"`
	FinishReason string         `json:"finish_reason"`
}

// GLM streaming response format
type glmStreamResponse struct {
	ID      string            `json:"id"`
	Object  string            `json:"object"`
	Created int64             `json:"created"`
	Model   string            `json:"model"`
	Choices []glmStreamChoice `json:"choices"`
	Usage   *openaiUsage      `json:"usage,omitempty"`
}

// Complete implements the Provider interface for GLM
func (p *GLMProvider) Complete(ctx context.Context, req *Request) (*Response, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}

	// Convert LiteLLM request to GLM format
	glmReq := p.convertRequest(req)

	// Marshal request
	reqBody, err := json.Marshal(glmReq)
	if err != nil {
		return nil, fmt.Errorf("glm: marshal request: %w", err)
	}

	// Create HTTP request
	baseURL := p.Config().BaseURL
	if baseURL == "" {
		baseURL = "https://open.bigmodel.cn/api/paas/v4"
	}
	httpReq, err := http.NewRequestWithContext(ctx, "POST", baseURL+"/chat/completions", bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("glm: create request: %w", err)
	}

	// Set headers
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+p.Config().APIKey)

	// Make request
	client := &http.Client{}
	resp, err := client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("glm: request failed: %w", err)
	}
	defer resp.Body.Close()

	// Read response
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("glm: read response: %w", err)
	}

	// Handle non-200 status codes
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("glm: API error %d: %s", resp.StatusCode, string(respBody))
	}

	// Parse response
	var glmResp glmResponse
	if err := json.Unmarshal(respBody, &glmResp); err != nil {
		return nil, fmt.Errorf("glm: unmarshal response: %w", err)
	}

	// Check if we have choices
	if len(glmResp.Choices) == 0 {
		return nil, fmt.Errorf("glm: no choices in response")
	}

	// Build response directly (following qwen.go pattern)
	choice := glmResp.Choices[0]

	response := &Response{
		Content:      choice.Message.Content,
		Model:        glmResp.Model,
		Provider:     "glm",
		FinishReason: choice.FinishReason,
		Usage: Usage{
			PromptTokens:     glmResp.Usage.PromptTokens,
			CompletionTokens: glmResp.Usage.CompletionTokens,
			TotalTokens:      glmResp.Usage.TotalTokens,
		},
	}

	// Handle reasoning content if present (like qwen.go)
	if choice.Message.ReasoningContent != "" {
		response.Reasoning = &ReasoningData{
			Content: choice.Message.ReasoningContent,
			Summary: choice.Message.ReasoningContent, // GLM doesn't separate content and summary
		}
	}

	// Convert tool calls from OpenAI format to LiteLLM format
	if len(choice.Message.ToolCalls) > 0 {
		response.ToolCalls = make([]ToolCall, len(choice.Message.ToolCalls))
		for i, tc := range choice.Message.ToolCalls {
			response.ToolCalls[i] = ToolCall{
				ID:   tc.ID,
				Type: tc.Type,
				Function: FunctionCall{
					Name:      tc.Function.Name,
					Arguments: tc.Function.Arguments,
				},
			}
		}
	}

	return response, nil
}

// convertRequest converts LiteLLM request to GLM format
func (p *GLMProvider) convertRequest(req *Request) *glmRequest {
	// Create base OpenAI request
	openaiReq := openaiRequest{
		Model:       req.Model,
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
		Stream:      false,
		ToolChoice:  req.ToolChoice,
	}

	// Convert tools to OpenAI format
	if len(req.Tools) > 0 {
		openaiReq.Tools = make([]openaiTool, len(req.Tools))
		for i, tool := range req.Tools {
			openaiReq.Tools[i] = openaiTool{
				Type: tool.Type,
				Function: openaiToolFunction{
					Name:        tool.Function.Name,
					Description: tool.Function.Description,
					Parameters:  tool.Function.Parameters,
				},
			}
		}
	}

	// Convert messages to OpenAI format
	openaiReq.Messages = make([]openaiMessage, len(req.Messages))
	for i, msg := range req.Messages {
		openaiReq.Messages[i] = openaiMessage{
			Role:       msg.Role,
			Content:    msg.Content,
			ToolCallID: msg.ToolCallID,
		}

		// Convert tool calls
		if len(msg.ToolCalls) > 0 {
			openaiReq.Messages[i].ToolCalls = make([]openaiToolCall, len(msg.ToolCalls))
			for j, tc := range msg.ToolCalls {
				openaiReq.Messages[i].ToolCalls[j] = openaiToolCall{
					ID:   tc.ID,
					Type: tc.Type,
					Function: openaiToolCallFunc{
						Name:      tc.Function.Name,
						Arguments: tc.Function.Arguments,
					},
				}
			}
		}
	}

	// Create GLM request with OpenAI base
	glmReq := &glmRequest{
		openaiRequest: openaiReq,
	}

	// Handle extra parameters
	if req.Extra != nil {
		// Extract thinking configuration
		if thinkingType := extractThinkingType(req.Extra["thinking"]); thinkingType != "" {
			glmReq.Thinking = &glmThinking{Type: thinkingType}
		}

		// Copy other extra parameters (excluding thinking)
		if extraCopy := copyExtraExcluding(req.Extra, "thinking"); len(extraCopy) > 0 {
			glmReq.Extra = extraCopy
		}
	}

	return glmReq
}

// extractThinkingType extracts thinking type from various map formats
func extractThinkingType(thinking interface{}) string {
	switch t := thinking.(type) {
	case map[string]string:
		return t["type"]
	case map[string]interface{}:
		if typeVal, ok := t["type"].(string); ok {
			return typeVal
		}
	}
	return ""
}

// copyExtraExcluding copies extra parameters excluding specified keys
func copyExtraExcluding(extra map[string]interface{}, excludeKeys ...string) map[string]interface{} {
	excludeSet := make(map[string]bool)
	for _, key := range excludeKeys {
		excludeSet[key] = true
	}

	result := make(map[string]interface{})
	for k, v := range extra {
		if !excludeSet[k] {
			result[k] = v
		}
	}
	return result
}

// Stream implements streaming chat completions for GLM
func (p *GLMProvider) Stream(ctx context.Context, req *Request) (StreamReader, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}

	// Convert request and enable streaming
	glmReq := p.convertRequest(req)
	glmReq.Stream = true

	// Marshal request
	reqBody, err := json.Marshal(glmReq)
	if err != nil {
		return nil, fmt.Errorf("glm: marshal request: %w", err)
	}

	// Create HTTP request
	baseURL := p.Config().BaseURL
	if baseURL == "" {
		baseURL = "https://open.bigmodel.cn/api/paas/v4"
	}
	httpReq, err := http.NewRequestWithContext(ctx, "POST", baseURL+"/chat/completions", bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("glm: create request: %w", err)
	}

	// Set headers
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+p.Config().APIKey)
	httpReq.Header.Set("Accept", "text/event-stream")

	// Make request
	client := &http.Client{}
	resp, err := client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("glm: request failed: %w", err)
	}

	// Check status code
	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("glm: API error %d: %s", resp.StatusCode, string(respBody))
	}

	return &glmStreamReader{
		scanner: bufio.NewScanner(resp.Body),
		resp:    resp,
	}, nil
}

// glmStreamReader implements StreamReader for GLM
type glmStreamReader struct {
	scanner *bufio.Scanner
	resp    *http.Response
	err     error
	done    bool
}

func (r *glmStreamReader) Read() (*StreamChunk, error) {
	if r.done {
		return &StreamChunk{Done: true, Provider: "glm"}, nil
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
				return &StreamChunk{Done: true, Provider: "glm"}, nil
			}

			// Parse JSON
			var streamResp glmStreamResponse
			if err := json.Unmarshal([]byte(data), &streamResp); err != nil {
				continue // Skip malformed chunks
			}

			// Convert to StreamChunk
			if len(streamResp.Choices) > 0 {
				choice := streamResp.Choices[0]

				chunk := &StreamChunk{
					Type:     ChunkTypeContent,
					Content:  choice.Delta.Content,
					Provider: "glm",
				}

				// Handle reasoning content (like qwen.go)
				if choice.Delta.ReasoningContent != "" {
					chunk.Type = ChunkTypeReasoning
					chunk.Reasoning = &ReasoningChunk{
						Content: choice.Delta.ReasoningContent,
					}
				} else if choice.Delta.Content != "" {
					chunk.Type = ChunkTypeContent
				}

				// Handle tool calls - convert from OpenAI format
				if len(choice.Delta.ToolCalls) > 0 {
					chunk.Type = ChunkTypeToolCall
					chunk.ToolCalls = make([]ToolCall, len(choice.Delta.ToolCalls))
					for i, tc := range choice.Delta.ToolCalls {
						chunk.ToolCalls[i] = ToolCall{
							ID:   tc.ID,
							Type: tc.Type,
							Function: FunctionCall{
								Name:      tc.Function.Name,
								Arguments: tc.Function.Arguments,
							},
						}
					}
				}

				// Handle finish reason
				if choice.FinishReason != "" {
					chunk.FinishReason = choice.FinishReason
				}

				return chunk, nil
			}
		}
	}

	// Check for scanner errors
	if err := r.scanner.Err(); err != nil {
		return nil, fmt.Errorf("glm: stream read error: %w", err)
	}

	// 正常结束流式读取
	r.done = true
	return &StreamChunk{Done: true, Provider: "glm"}, nil
}

func (r *glmStreamReader) Close() error {
	return r.resp.Body.Close()
}

func (r *glmStreamReader) Err() error {
	return r.err
}

// Register GLM provider
func init() {
	RegisterProvider("glm", NewGLMProvider)
}
