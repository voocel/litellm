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

// DeepSeekProvider implements the Provider interface for DeepSeek
type DeepSeekProvider struct {
	*BaseProvider
}

// NewDeepSeekProvider creates a new DeepSeek provider
func NewDeepSeekProvider(config ProviderConfig) Provider {
	return &DeepSeekProvider{
		BaseProvider: NewBaseProvider("deepseek", config),
	}
}

// Models returns the list of available DeepSeek models
func (p *DeepSeekProvider) Models() []ModelInfo {
	return []ModelInfo{
		{
			ID: "deepseek-chat", Provider: "deepseek", Name: "DeepSeek Chat", MaxTokens: 32768,
			Capabilities: []ModelCapability{CapabilityChat, CapabilityFunctionCall},
		},
		{
			ID: "deepseek-reasoner", Provider: "deepseek", Name: "DeepSeek Reasoner", MaxTokens: 65536,
			Capabilities: []ModelCapability{CapabilityChat, CapabilityReasoning, CapabilityFunctionCall},
		},
	}
}

// Complete implements the Provider interface for DeepSeek
func (p *DeepSeekProvider) Complete(ctx context.Context, req *Request) (*Response, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}

	// Handle response format - DeepSeek supports json_object but not json_schema
	messages := req.Messages
	if req.ResponseFormat != nil {
		switch req.ResponseFormat.Type {
		case ResponseFormatJSONObject:
			// DeepSeek supports json_object natively
			// Will be added to request below
		case ResponseFormatJSONSchema:
			// DeepSeek doesn't support json_schema, so we use json_object + prompt engineering
			// Add schema instructions to the last user message
			if len(messages) > 0 && messages[len(messages)-1].Role == "user" {
				lastMsg := messages[len(messages)-1]
				lastMsg.Content = p.addJSONSchemaInstructions(lastMsg.Content, req.ResponseFormat)
				messages = append(messages[:len(messages)-1], lastMsg)
			}
		}
	}

	// Build DeepSeek request (OpenAI compatible)
	deepseekReq := map[string]interface{}{
		"model":    req.Model,
		"messages": messages,
		"stream":   false,
	}

	if req.MaxTokens != nil {
		deepseekReq["max_tokens"] = *req.MaxTokens
	}
	if req.Temperature != nil {
		deepseekReq["temperature"] = *req.Temperature
	}

	// Add response format if specified
	if req.ResponseFormat != nil {
		switch req.ResponseFormat.Type {
		case ResponseFormatJSONObject, ResponseFormatJSONSchema:
			// For both cases, use json_object (schema handled via prompt engineering above)
			deepseekReq["response_format"] = map[string]interface{}{
				"type": "json_object",
			}
		}
	}

	// Handle extra parameters for provider-specific features
	if req.Extra != nil {
		if topP, ok := req.Extra["top_p"]; ok {
			deepseekReq["top_p"] = topP
		}
		if freqPenalty, ok := req.Extra["frequency_penalty"]; ok {
			deepseekReq["frequency_penalty"] = freqPenalty
		}
		if presPenalty, ok := req.Extra["presence_penalty"]; ok {
			deepseekReq["presence_penalty"] = presPenalty
		}
		if stop, ok := req.Extra["stop"]; ok {
			deepseekReq["stop"] = stop
		}
	}

	// Handle tools
	if len(req.Tools) > 0 {
		tools := make([]map[string]interface{}, len(req.Tools))
		for i, tool := range req.Tools {
			tools[i] = map[string]interface{}{
				"type": tool.Type,
				"function": map[string]interface{}{
					"name":        tool.Function.Name,
					"description": tool.Function.Description,
					"parameters":  tool.Function.Parameters,
				},
			}
		}
		deepseekReq["tools"] = tools
		if req.ToolChoice != "" {
			deepseekReq["tool_choice"] = req.ToolChoice
		}
	}

	// Marshal request
	reqBody, err := json.Marshal(deepseekReq)
	if err != nil {
		return nil, fmt.Errorf("deepseek: failed to marshal request: %w", err)
	}

	// Create HTTP request
	url := fmt.Sprintf("%s/chat/completions", p.config.BaseURL)
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("deepseek: failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+p.config.APIKey)

	// Send request
	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("deepseek: request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("deepseek: API error %d: %s", resp.StatusCode, string(body))
	}

	// Parse response
	var deepseekResp struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		Created int64  `json:"created"`
		Model   string `json:"model"`
		Choices []struct {
			Index   int `json:"index"`
			Message struct {
				Role             string `json:"role"`
				Content          string `json:"content"`
				ReasoningContent string `json:"reasoning_content,omitempty"`
				ToolCalls        []struct {
					ID       string `json:"id"`
					Type     string `json:"type"`
					Function struct {
						Name      string `json:"name"`
						Arguments string `json:"arguments"`
					} `json:"function"`
				} `json:"tool_calls,omitempty"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
			TotalTokens      int `json:"total_tokens"`
		} `json:"usage"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&deepseekResp); err != nil {
		return nil, fmt.Errorf("deepseek: failed to decode response: %w", err)
	}

	if len(deepseekResp.Choices) == 0 {
		return nil, fmt.Errorf("deepseek: no choices in response")
	}

	choice := deepseekResp.Choices[0]
	response := &Response{
		Content:  choice.Message.Content,
		Model:    deepseekResp.Model,
		Provider: "deepseek",
		Usage: Usage{
			PromptTokens:     deepseekResp.Usage.PromptTokens,
			CompletionTokens: deepseekResp.Usage.CompletionTokens,
			TotalTokens:      deepseekResp.Usage.TotalTokens,
		},
		FinishReason: choice.FinishReason,
	}

	if choice.Message.ReasoningContent != "" {
		response.Reasoning = &ReasoningData{
			Content: choice.Message.ReasoningContent,
			Summary: choice.Message.ReasoningContent,
		}
	}

	// Handle tool calls
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

// Stream implements streaming for DeepSeek
func (p *DeepSeekProvider) Stream(ctx context.Context, req *Request) (StreamReader, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}

	// Build streaming request
	deepseekReq := map[string]interface{}{
		"model":    req.Model,
		"messages": req.Messages,
		"stream":   true,
	}

	if req.MaxTokens != nil {
		deepseekReq["max_tokens"] = *req.MaxTokens
	}
	if req.Temperature != nil {
		deepseekReq["temperature"] = *req.Temperature
	}

	// Handle extra parameters for provider-specific features
	if req.Extra != nil {
		if topP, ok := req.Extra["top_p"]; ok {
			deepseekReq["top_p"] = topP
		}
	}

	// Handle tools
	if len(req.Tools) > 0 {
		tools := make([]map[string]interface{}, len(req.Tools))
		for i, tool := range req.Tools {
			tools[i] = map[string]interface{}{
				"type": tool.Type,
				"function": map[string]interface{}{
					"name":        tool.Function.Name,
					"description": tool.Function.Description,
					"parameters":  tool.Function.Parameters,
				},
			}
		}
		deepseekReq["tools"] = tools
		if req.ToolChoice != "" {
			deepseekReq["tool_choice"] = req.ToolChoice
		}
	}

	reqBody, err := json.Marshal(deepseekReq)
	if err != nil {
		return nil, fmt.Errorf("deepseek: failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/chat/completions", p.config.BaseURL)
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("deepseek: failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+p.config.APIKey)
	httpReq.Header.Set("Accept", "text/event-stream")

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("deepseek: request failed: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("deepseek: API error %d: %s", resp.StatusCode, string(body))
	}

	return &deepseekStreamReader{
		resp:     resp,
		scanner:  bufio.NewScanner(resp.Body),
		provider: "deepseek",
	}, nil
}

// deepseekStreamReader implements StreamReader for DeepSeek
type deepseekStreamReader struct {
	resp     *http.Response
	scanner  *bufio.Scanner
	provider string
	err      error
	done     bool
}

func (r *deepseekStreamReader) Read() (*StreamChunk, error) {
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

		var chunk struct {
			ID      string `json:"id"`
			Object  string `json:"object"`
			Created int64  `json:"created"`
			Model   string `json:"model"`
			Choices []struct {
				Index int `json:"index"`
				Delta struct {
					Content          string `json:"content,omitempty"`
					ReasoningContent string `json:"reasoning_content,omitempty"` // DeepSeek推理内容增量
					ToolCalls        []struct {
						Index    int    `json:"index"`
						ID       string `json:"id,omitempty"`
						Type     string `json:"type,omitempty"`
						Function struct {
							Name      string `json:"name,omitempty"`
							Arguments string `json:"arguments,omitempty"`
						} `json:"function,omitempty"`
					} `json:"tool_calls,omitempty"`
				} `json:"delta"`
				FinishReason string `json:"finish_reason,omitempty"`
			} `json:"choices"`
		}

		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}

		if len(chunk.Choices) > 0 {
			choice := chunk.Choices[0]
			streamChunk := &StreamChunk{
				Provider: r.provider,
				Model:    chunk.Model,
			}

			if choice.Delta.Content != "" {
				streamChunk.Type = ChunkTypeContent
				streamChunk.Content = choice.Delta.Content
			}

			if choice.Delta.ReasoningContent != "" {
				streamChunk.Type = ChunkTypeReasoning
				streamChunk.Reasoning = &ReasoningChunk{
					Content: choice.Delta.ReasoningContent,
				}
			}

			// Handle tool call deltas
			if len(choice.Delta.ToolCalls) > 0 {
				for _, toolCallDelta := range choice.Delta.ToolCalls {
					streamChunk.Type = ChunkTypeToolCallDelta
					streamChunk.ToolCallDelta = &ToolCallDelta{
						Index: toolCallDelta.Index,
						ID:    toolCallDelta.ID,
						Type:  toolCallDelta.Type,
					}

					if toolCallDelta.Function.Name != "" {
						streamChunk.ToolCallDelta.FunctionName = toolCallDelta.Function.Name
					}
					if toolCallDelta.Function.Arguments != "" {
						streamChunk.ToolCallDelta.ArgumentsDelta = toolCallDelta.Function.Arguments
					}

					return streamChunk, nil
				}
			}

			if choice.FinishReason != "" {
				streamChunk.FinishReason = choice.FinishReason
			}

			if streamChunk.Type != "" || streamChunk.FinishReason != "" {
				return streamChunk, nil
			}
		}
	}

	if err := r.scanner.Err(); err != nil {
		r.err = err
		return nil, err
	}

	r.done = true
	return &StreamChunk{Done: true, Provider: r.provider}, nil
}

func (r *deepseekStreamReader) Close() error {
	return r.resp.Body.Close()
}

func (r *deepseekStreamReader) Err() error {
	return r.err
}

// addJSONSchemaInstructions adds JSON schema formatting instructions for DeepSeek
func (p *DeepSeekProvider) addJSONSchemaInstructions(content string, format *ResponseFormat) string {
	if format.JSONSchema != nil && format.JSONSchema.Schema != nil {
		schemaJSON, _ := json.Marshal(format.JSONSchema.Schema)
		return fmt.Sprintf("%s\n\nPlease respond with a valid JSON object that strictly follows this schema:\n%s\n\nRespond with JSON only, no additional text.",
			content, string(schemaJSON))
	}
	return content
}

func init() {
	RegisterProvider("deepseek", NewDeepSeekProvider)
}
