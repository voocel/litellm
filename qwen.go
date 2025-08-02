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

// QwenProvider implements the Provider interface for Qwen (Alibaba Cloud DashScope)
type QwenProvider struct {
	*BaseProvider
}

// qwenRequest extends openaiRequest with Qwen-specific parameters
type qwenRequest struct {
	openaiRequest
	EnableThinking bool `json:"enable_thinking,omitempty"` // Enable reasoning mode for Qwen3
}

// qwenMessage extends openaiMessage with reasoning content
type qwenMessage struct {
	openaiMessage
	ReasoningContent string `json:"reasoning_content,omitempty"` // Reasoning process content
}

// qwenChoice extends openaiChoice with reasoning content
type qwenChoice struct {
	Index        int         `json:"index"`
	Message      qwenMessage `json:"message"`
	FinishReason string      `json:"finish_reason"`
}

// qwenResponse extends openaiResponse with reasoning support
type qwenResponse struct {
	ID      string       `json:"id"`
	Object  string       `json:"object"`
	Model   string       `json:"model"`
	Choices []qwenChoice `json:"choices"`
	Usage   openaiUsage  `json:"usage"`
}

// qwenStreamChoice extends openaiChoice for streaming with reasoning
type qwenStreamChoice struct {
	Index        int             `json:"index"`
	Delta        qwenStreamDelta `json:"delta"`
	FinishReason string          `json:"finish_reason"`
}

// qwenStreamDelta extends openaiDelta with reasoning content
type qwenStreamDelta struct {
	openaiDelta
	ReasoningContent string `json:"reasoning_content,omitempty"` // Reasoning process content
}

// qwenStreamResponse extends openaiResponse for streaming with reasoning
type qwenStreamResponse struct {
	ID      string             `json:"id"`
	Object  string             `json:"object"`
	Model   string             `json:"model"`
	Choices []qwenStreamChoice `json:"choices"`
}

// NewQwenProvider creates a new Qwen provider
func NewQwenProvider(config ProviderConfig) Provider {
	return &QwenProvider{
		BaseProvider: NewBaseProvider("qwen", config),
	}
}

// Models returns the list of supported models
func (p *QwenProvider) Models() []ModelInfo {
	return []ModelInfo{
		{
			ID: "qwen3-coder-plus", Provider: "qwen", Name: "Qwen3-Coder-Plus", MaxTokens: 1000000,
			Capabilities: []ModelCapability{CapabilityChat, CapabilityFunctionCall, CapabilityReasoning},
		},
		{
			ID: "qwen3-coder-plus-2025-07-22", Provider: "qwen", Name: "Qwen3-Coder-Plus (2025-07-22)", MaxTokens: 1000000,
			Capabilities: []ModelCapability{CapabilityChat, CapabilityFunctionCall, CapabilityReasoning},
		},
		{
			ID: "qwen3-coder-flash", Provider: "qwen", Name: "Qwen3-Coder-Flash", MaxTokens: 1000000,
			Capabilities: []ModelCapability{CapabilityChat, CapabilityFunctionCall, CapabilityReasoning},
		},
		{
			ID: "qwen3-coder-flash-2025-07-28", Provider: "qwen", Name: "Qwen3-Coder-Flash (2025-07-28)", MaxTokens: 1000000,
			Capabilities: []ModelCapability{CapabilityChat, CapabilityFunctionCall, CapabilityReasoning},
		},
		{
			ID: "qwen3-coder-480b-a35b-instruct", Provider: "qwen", Name: "Qwen3-Coder-480B-A35B-Instruct", MaxTokens: 262144,
			Capabilities: []ModelCapability{CapabilityChat, CapabilityFunctionCall, CapabilityReasoning},
		},
		{
			ID: "qwen3-coder-30b-a3b-instruct", Provider: "qwen", Name: "Qwen3-Coder-30B-A3B-Instruct", MaxTokens: 262144,
			Capabilities: []ModelCapability{CapabilityChat, CapabilityFunctionCall, CapabilityReasoning},
		},
	}
}

// buildURL constructs the full API URL
func (p *QwenProvider) buildURL(endpoint string) string {
	baseURL := p.config.BaseURL
	if baseURL == "" {
		baseURL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
	}
	return baseURL + endpoint
}

// Complete performs a completion request
func (p *QwenProvider) Complete(ctx context.Context, req *Request) (*Response, error) {
	// Convert LiteLLM request to Qwen-compatible format
	qwenReq := qwenRequest{
		openaiRequest: openaiRequest{
			Model:       req.Model,
			Messages:    make([]openaiMessage, len(req.Messages)),
			MaxTokens:   req.MaxTokens,
			Temperature: req.Temperature,
			Stream:      false,
		},
	}

	// Handle extra parameters for Qwen-specific features
	if req.Extra != nil {
		// Support enable_thinking parameter for Qwen3 reasoning
		if enableThinking, ok := req.Extra["enable_thinking"].(bool); ok && enableThinking {
			qwenReq.EnableThinking = true
		}
	}

	// Convert messages
	for i, msg := range req.Messages {
		qwenReq.openaiRequest.Messages[i] = openaiMessage{
			Role:    msg.Role,
			Content: msg.Content,
		}

		// Handle tool calls
		if len(msg.ToolCalls) > 0 {
			qwenReq.openaiRequest.Messages[i].ToolCalls = make([]openaiToolCall, len(msg.ToolCalls))
			for j, tc := range msg.ToolCalls {
				qwenReq.openaiRequest.Messages[i].ToolCalls[j] = openaiToolCall{
					ID:   tc.ID,
					Type: tc.Type,
					Function: openaiToolCallFunc{
						Name:      tc.Function.Name,
						Arguments: tc.Function.Arguments,
					},
				}
			}
		}

		// Handle tool call ID for tool responses
		if msg.ToolCallID != "" {
			qwenReq.openaiRequest.Messages[i].ToolCallID = msg.ToolCallID
		}
	}

	// Convert tool definitions
	if len(req.Tools) > 0 {
		qwenReq.openaiRequest.Tools = make([]openaiTool, len(req.Tools))
		for i, tool := range req.Tools {
			qwenReq.openaiRequest.Tools[i] = openaiTool{
				Type: tool.Type,
				Function: openaiToolFunction{
					Name:        tool.Function.Name,
					Description: tool.Function.Description,
					Parameters:  tool.Function.Parameters,
				},
			}
		}

		// Set tool choice if specified
		if req.ToolChoice != nil {
			qwenReq.openaiRequest.ToolChoice = req.ToolChoice
		}
	}

	body, err := json.Marshal(qwenReq)
	if err != nil {
		return nil, fmt.Errorf("qwen: marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.buildURL("/chat/completions"), bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("qwen: create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+p.config.APIKey)

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("qwen: request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("qwen1: API error %d: %s", resp.StatusCode, string(body))
	}

	// Read response body
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("qwen: read response: %w", err)
	}

	var qwenResp qwenResponse
	if err := json.Unmarshal(respBody, &qwenResp); err != nil {
		return nil, fmt.Errorf("qwen: unmarshal response: %w", err)
	}

	// Debug: log raw response for reasoning mode debugging
	if qwenReq.EnableThinking {
		fmt.Printf("DEBUG: Raw API response for thinking mode: %s\n", string(respBody))

		// Try to parse as generic JSON to see all fields
		var genericResp map[string]interface{}
		if err := json.Unmarshal(respBody, &genericResp); err == nil {
			if choices, ok := genericResp["choices"].([]interface{}); ok && len(choices) > 0 {
				if choice, ok := choices[0].(map[string]interface{}); ok {
					if message, ok := choice["message"].(map[string]interface{}); ok {
						fmt.Printf("DEBUG: Message fields: %+v\n", message)
					}
				}
			}
		}
	}

	if len(qwenResp.Choices) == 0 {
		return nil, fmt.Errorf("qwen: no choices in response")
	}

	choice := qwenResp.Choices[0]
	response := &Response{
		Content:      choice.Message.Content,
		Model:        qwenResp.Model,
		Provider:     "qwen",
		FinishReason: choice.FinishReason,
		Usage: Usage{
			PromptTokens:     qwenResp.Usage.PromptTokens,
			CompletionTokens: qwenResp.Usage.CompletionTokens,
			TotalTokens:      qwenResp.Usage.TotalTokens,
		},
	}

	// Handle reasoning content if present
	if choice.Message.ReasoningContent != "" {
		response.Reasoning = &ReasoningData{
			Content: choice.Message.ReasoningContent,
			Summary: choice.Message.ReasoningContent, // Qwen doesn't separate content and summary
		}
	}

	// Handle tool calls
	if len(choice.Message.openaiMessage.ToolCalls) > 0 {
		response.ToolCalls = make([]ToolCall, len(choice.Message.openaiMessage.ToolCalls))
		for i, tc := range choice.Message.openaiMessage.ToolCalls {
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

// Stream performs a streaming completion request
func (p *QwenProvider) Stream(ctx context.Context, req *Request) (StreamReader, error) {
	// Convert LiteLLM request to Qwen-compatible format
	qwenReq := qwenRequest{
		openaiRequest: openaiRequest{
			Model:       req.Model,
			Messages:    make([]openaiMessage, len(req.Messages)),
			MaxTokens:   req.MaxTokens,
			Temperature: req.Temperature,
			Stream:      true,
		},
	}

	// Handle extra parameters for Qwen-specific features
	if req.Extra != nil {
		// Support enable_thinking parameter for Qwen3 reasoning
		if enableThinking, ok := req.Extra["enable_thinking"].(bool); ok && enableThinking {
			qwenReq.EnableThinking = true
		}
	}

	// Convert messages (same logic as Complete method)
	for i, msg := range req.Messages {
		qwenReq.openaiRequest.Messages[i] = openaiMessage{
			Role:    msg.Role,
			Content: msg.Content,
		}

		if len(msg.ToolCalls) > 0 {
			qwenReq.openaiRequest.Messages[i].ToolCalls = make([]openaiToolCall, len(msg.ToolCalls))
			for j, tc := range msg.ToolCalls {
				qwenReq.openaiRequest.Messages[i].ToolCalls[j] = openaiToolCall{
					ID:   tc.ID,
					Type: tc.Type,
					Function: openaiToolCallFunc{
						Name:      tc.Function.Name,
						Arguments: tc.Function.Arguments,
					},
				}
			}
		}

		if msg.ToolCallID != "" {
			qwenReq.openaiRequest.Messages[i].ToolCallID = msg.ToolCallID
		}
	}

	// Convert tool definitions
	if len(req.Tools) > 0 {
		qwenReq.openaiRequest.Tools = make([]openaiTool, len(req.Tools))
		for i, tool := range req.Tools {
			qwenReq.openaiRequest.Tools[i] = openaiTool{
				Type: tool.Type,
				Function: openaiToolFunction{
					Name:        tool.Function.Name,
					Description: tool.Function.Description,
					Parameters:  tool.Function.Parameters,
				},
			}
		}

		if req.ToolChoice != nil {
			qwenReq.openaiRequest.ToolChoice = req.ToolChoice
		}
	}

	body, err := json.Marshal(qwenReq)
	if err != nil {
		return nil, fmt.Errorf("qwen: marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.buildURL("/chat/completions"), bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("qwen: create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+p.config.APIKey)

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("qwen: request failed: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("qwen: API error %d: %s", resp.StatusCode, string(body))
	}

	return &qwenStreamReader{
		resp:     resp,
		scanner:  bufio.NewScanner(resp.Body),
		provider: "qwen",
	}, nil
}

// qwenStreamReader implements StreamReader for Qwen streaming responses
type qwenStreamReader struct {
	resp     *http.Response
	scanner  *bufio.Scanner
	provider string
	err      error
	done     bool
}

func (r *qwenStreamReader) Read() (*StreamChunk, error) {
	if r.done {
		return nil, io.EOF
	}

	for r.scanner.Scan() {
		line := strings.TrimSpace(r.scanner.Text())

		if line == "" {
			continue
		}

		if line == "data: [DONE]" {
			r.done = true
			return nil, io.EOF
		}

		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			r.done = true
			return nil, io.EOF
		}

		var streamResp qwenStreamResponse
		if err := json.Unmarshal([]byte(data), &streamResp); err != nil {
			continue // Skip malformed chunks
		}

		if len(streamResp.Choices) == 0 {
			continue
		}

		choice := streamResp.Choices[0]
		chunk := &StreamChunk{
			Content:      choice.Delta.Content,
			Model:        streamResp.Model,
			Provider:     r.provider,
			FinishReason: choice.FinishReason,
		}

		// Handle reasoning content
		if choice.Delta.ReasoningContent != "" {
			chunk.Type = ChunkTypeReasoning
			chunk.Reasoning = &ReasoningChunk{
				Content: choice.Delta.ReasoningContent,
			}
		} else if choice.Delta.Content != "" {
			chunk.Type = ChunkTypeContent
		}

		// Handle tool calls in streaming
		if len(choice.Delta.openaiDelta.ToolCalls) > 0 {
			chunk.ToolCalls = make([]ToolCall, len(choice.Delta.openaiDelta.ToolCalls))
			for i, tc := range choice.Delta.openaiDelta.ToolCalls {
				chunk.ToolCalls[i] = ToolCall{
					ID:   tc.ID,
					Type: tc.Type,
					Function: FunctionCall{
						Name:      tc.Function.Name,
						Arguments: tc.Function.Arguments,
					},
				}
			}
			chunk.Type = ChunkTypeToolCall
		}

		return chunk, nil
	}

	if err := r.scanner.Err(); err != nil {
		r.err = err
		return nil, fmt.Errorf("qwen: stream read error: %w", err)
	}

	r.done = true
	return nil, io.EOF
}

func (r *qwenStreamReader) Close() error {
	if r.resp != nil {
		return r.resp.Body.Close()
	}
	return nil
}

func (r *qwenStreamReader) Err() error {
	return r.err
}

// Register the provider
func init() {
	RegisterProvider("qwen", NewQwenProvider)
}
