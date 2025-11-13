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

// GLMProvider implements GLM/ZhiPu AI API integration
type GLMProvider struct {
	*BaseProvider
}

// NewGLM creates a new GLM provider
func NewGLM(config ProviderConfig) *GLMProvider {
	baseProvider := NewBaseProvider("glm", config)

	return &GLMProvider{
		BaseProvider: baseProvider,
	}
}

func (p *GLMProvider) SupportsModel(model string) bool {
	for _, m := range p.Models() {
		if m.ID == model {
			return true
		}
	}
	return false
}

func (p *GLMProvider) Models() []ModelInfo {
	return []ModelInfo{
		{
			ID: "glm-4.6", Provider: "glm", Name: "GLM-4.6", MaxTokens: 128000,
			Capabilities: []string{"chat", "function_call", "code", "reasoning"},
		},
		{
			ID: "glm-4.5", Provider: "glm", Name: "GLM-4.5", MaxTokens: 128000,
			Capabilities: []string{"chat", "function_call", "code", "reasoning"},
		},
		{
			ID: "glm-4.5-air", Provider: "glm", Name: "GLM-4.5 Air", MaxTokens: 128000,
			Capabilities: []string{"chat", "function_call", "code", "reasoning"},
		},
	}
}

func (p *GLMProvider) Chat(ctx context.Context, req *Request) (*Response, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}

	// Build GLM request (OpenAI compatible with extensions)
	glmReq := map[string]any{
		"model":    req.Model,
		"messages": p.convertMessages(req.Messages),
	}

	if req.MaxTokens != nil {
		glmReq["max_tokens"] = *req.MaxTokens
	}
	if req.Temperature != nil {
		glmReq["temperature"] = *req.Temperature
	}
	if len(req.Tools) > 0 {
		glmReq["tools"] = p.convertTools(req.Tools)
	}
	if req.ToolChoice != nil {
		glmReq["tool_choice"] = req.ToolChoice
	}

	// Handle response format
	if req.ResponseFormat != nil {
		if req.ResponseFormat.Type == "json_object" {
			glmReq["response_format"] = map[string]string{"type": "json_object"}
		}
		// GLM doesn't support json_schema, so we use json_object + prompt engineering
		if req.ResponseFormat.Type == "json_schema" {
			glmReq["response_format"] = map[string]string{"type": "json_object"}
			// Add schema instructions to the last user message
			if messages, ok := glmReq["messages"].([]glmMessage); ok && len(messages) > 0 {
				lastMsg := &messages[len(messages)-1]
				if lastMsg.Role == "user" && req.ResponseFormat.JSONSchema != nil {
					lastMsg.Content = p.addJSONSchemaInstructions(lastMsg.Content, req.ResponseFormat)
				}
			}
		}
	}

	// Handle GLM-specific thinking parameter
	if req.Extra != nil {
		if thinking, exists := req.Extra["thinking"]; exists {
			if thinkingMap, ok := thinking.(map[string]any); ok {
				if thinkingType, ok := thinkingMap["type"].(string); ok {
					glmReq["thinking"] = map[string]string{"type": thinkingType}
				}
			}
		}
	}

	body, err := json.Marshal(glmReq)
	if err != nil {
		return nil, fmt.Errorf("glm: marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/chat/completions", p.Config().BaseURL)
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("glm: create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+p.Config().APIKey)

	resp, err := p.HTTPClient().Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("glm: request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, err := io.ReadAll(resp.Body)
		if err != nil {
			return nil, fmt.Errorf("glm: failed to read error response: %w", err)
		}
		return nil, fmt.Errorf("glm: API error %d: %s", resp.StatusCode, string(body))
	}

	var glmResp glmResponse
	if err := json.NewDecoder(resp.Body).Decode(&glmResp); err != nil {
		return nil, fmt.Errorf("glm: decode response: %w", err)
	}

	response := &Response{
		Model:    glmResp.Model,
		Provider: "glm",
	}

	if glmResp.Usage != nil {
		response.Usage = Usage{
			PromptTokens:     glmResp.Usage.PromptTokens,
			CompletionTokens: glmResp.Usage.CompletionTokens,
			TotalTokens:      glmResp.Usage.TotalTokens,
		}
	}

	if len(glmResp.Choices) > 0 {
		choice := glmResp.Choices[0]
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

		// Handle reasoning content (GLM thinking mode)
		if choice.Message.ReasoningContent != "" {
			response.Reasoning = &ReasoningData{
				Content:    choice.Message.ReasoningContent,
				Summary:    "GLM reasoning process",
				TokensUsed: 0, // GLM doesn't separate reasoning tokens
			}
		}
	}

	return response, nil
}

func (p *GLMProvider) Stream(ctx context.Context, req *Request) (StreamReader, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}

	// Build request (similar to Chat but with stream: true)
	glmReq := map[string]any{
		"model":    req.Model,
		"messages": p.convertMessages(req.Messages),
		"stream":   true,
	}

	if req.MaxTokens != nil {
		glmReq["max_tokens"] = *req.MaxTokens
	}
	if req.Temperature != nil {
		glmReq["temperature"] = *req.Temperature
	}
	if len(req.Tools) > 0 {
		glmReq["tools"] = p.convertTools(req.Tools)
	}

	body, err := json.Marshal(glmReq)
	if err != nil {
		return nil, fmt.Errorf("glm: marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/chat/completions", p.Config().BaseURL)
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("glm: create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+p.Config().APIKey)
	httpReq.Header.Set("Accept", "text/event-stream")

	resp, err := p.HTTPClient().Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("glm: request failed: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		body, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			return nil, fmt.Errorf("glm: failed to read stream error response: %w", err)
		}
		return nil, fmt.Errorf("glm: API error %d: %s", resp.StatusCode, string(body))
	}

	return &glmStreamReader{
		resp:     resp,
		scanner:  bufio.NewScanner(resp.Body),
		provider: "glm",
	}, nil
}

// convertMessages converts standard messages to GLM format
func (p *GLMProvider) convertMessages(messages []Message) []glmMessage {
	glmMessages := make([]glmMessage, len(messages))
	for i, msg := range messages {
		glmMessages[i] = glmMessage{
			Role:    msg.Role,
			Content: msg.Content,
		}

		// Handle tool calls
		if len(msg.ToolCalls) > 0 {
			for _, toolCall := range msg.ToolCalls {
				glmMessages[i].ToolCalls = append(glmMessages[i].ToolCalls, glmToolCall{
					ID:   toolCall.ID,
					Type: toolCall.Type,
					Function: glmFunctionCall{
						Name:      toolCall.Function.Name,
						Arguments: toolCall.Function.Arguments,
					},
				})
			}
		}

		// Handle tool call responses
		if msg.ToolCallID != "" {
			glmMessages[i].ToolCallID = msg.ToolCallID
		}
	}
	return glmMessages
}

// convertTools converts standard tools to GLM format
func (p *GLMProvider) convertTools(tools []Tool) []glmTool {
	glmTools := make([]glmTool, len(tools))
	for i, tool := range tools {
		glmTools[i] = glmTool{
			Type: tool.Type,
			Function: glmFunction{
				Name:        tool.Function.Name,
				Description: tool.Function.Description,
				Parameters:  tool.Function.Parameters,
			},
		}
	}
	return glmTools
}

// addJSONSchemaInstructions adds JSON schema formatting instructions
func (p *GLMProvider) addJSONSchemaInstructions(content string, format *ResponseFormat) string {
	if format.JSONSchema != nil && format.JSONSchema.Schema != nil {
		schemaJSON, _ := json.Marshal(format.JSONSchema.Schema)
		return fmt.Sprintf("%s\n\nPlease respond with a valid JSON object that strictly follows this schema:\n%s\n\nRespond with JSON only, no additional text.",
			content, string(schemaJSON))
	}
	return content
}

// GLM API structures
type glmMessage struct {
	Role             string        `json:"role"`
	Content          string        `json:"content"`
	ToolCalls        []glmToolCall `json:"tool_calls,omitempty"`
	ToolCallID       string        `json:"tool_call_id,omitempty"`
	ReasoningContent string        `json:"reasoning_content,omitempty"`
}

type glmToolCall struct {
	ID       string          `json:"id"`
	Type     string          `json:"type"`
	Function glmFunctionCall `json:"function"`
}

type glmFunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type glmTool struct {
	Type     string      `json:"type"`
	Function glmFunction `json:"function"`
}

type glmFunction struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Parameters  any    `json:"parameters"`
}

type glmResponse struct {
	ID      string      `json:"id"`
	Object  string      `json:"object"`
	Created int64       `json:"created"`
	Model   string      `json:"model"`
	Choices []glmChoice `json:"choices"`
	Usage   *glmUsage   `json:"usage,omitempty"`
}

type glmChoice struct {
	Index        int        `json:"index"`
	Message      glmMessage `json:"message"`
	FinishReason string     `json:"finish_reason"`
}

type glmUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// glmStreamReader implements streaming for GLM
type glmStreamReader struct {
	resp     *http.Response
	scanner  *bufio.Scanner
	provider string
	done     bool
}

func (r *glmStreamReader) Next() (*StreamChunk, error) {
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

			// Parse JSON
			var streamResp glmStreamResponse
			if err := json.Unmarshal([]byte(data), &streamResp); err != nil {
				continue // Skip malformed chunks
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

				// Handle reasoning content
				if choice.Delta.ReasoningContent != "" {
					chunk.Type = "reasoning"
					chunk.Reasoning = &ReasoningChunk{
						Content: choice.Delta.ReasoningContent,
						Summary: "GLM reasoning process",
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

	// Check for scanner errors
	if err := r.scanner.Err(); err != nil {
		return nil, err
	}

	r.done = true
	return &StreamChunk{Done: true, Provider: r.provider}, nil
}

func (r *glmStreamReader) Close() error {
	return r.resp.Body.Close()
}

// Streaming response structures
type glmStreamResponse struct {
	ID      string            `json:"id"`
	Object  string            `json:"object"`
	Created int64             `json:"created"`
	Model   string            `json:"model"`
	Choices []glmStreamChoice `json:"choices"`
}

type glmStreamChoice struct {
	Index        int            `json:"index"`
	Delta        glmStreamDelta `json:"delta"`
	FinishReason string         `json:"finish_reason"`
}

type glmStreamDelta struct {
	Role             string        `json:"role,omitempty"`
	Content          string        `json:"content,omitempty"`
	ReasoningContent string        `json:"reasoning_content,omitempty"`
	ToolCalls        []glmToolCall `json:"tool_calls,omitempty"`
}
