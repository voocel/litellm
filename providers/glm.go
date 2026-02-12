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
	RegisterBuiltin("glm", func(cfg ProviderConfig) Provider {
		return NewGLM(cfg)
	}, "https://open.bigmodel.cn/api/paas/v4")
}

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

func (p *GLMProvider) validateExtra(req *Request) error {
	return p.BaseProvider.ValidateExtra(req.Extra, nil)
}

func (p *GLMProvider) Chat(ctx context.Context, req *Request) (*Response, error) {
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
		return nil, NewHTTPError("glm", resp.StatusCode, string(body))
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
		response.FinishReason = NormalizeFinishReason(choice.FinishReason)

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
		if !isThinkingDisabled(req) && choice.Message.ReasoningContent != "" {
			response.Reasoning = &ReasoningData{
				Content:    choice.Message.ReasoningContent,
				TokensUsed: 0, // GLM doesn't separate reasoning tokens
			}
		}
	}

	return response, nil
}

func (p *GLMProvider) Stream(ctx context.Context, req *Request) (StreamReader, error) {
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
		return nil, NewHTTPError("glm", resp.StatusCode, string(body))
	}

	return &glmStreamReader{
		resp:             resp,
		scanner:          bufio.NewScanner(resp.Body),
		provider:         "glm",
		model:            req.Model,
		includeReasoning: !isThinkingDisabled(req),
	}, nil
}

func (p *GLMProvider) buildHTTPRequest(ctx context.Context, req *Request, stream bool) (*http.Request, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}
	if err := p.validateExtra(req); err != nil {
		return nil, err
	}
	if err := p.BaseProvider.ValidateRequest(req); err != nil {
		return nil, err
	}

	messages := req.Messages
	if req.ResponseFormat != nil && req.ResponseFormat.Type == "json_schema" && req.ResponseFormat.JSONSchema != nil {
		messages = make([]Message, len(req.Messages))
		copy(messages, req.Messages)
		for i := len(messages) - 1; i >= 0; i-- {
			if messages[i].Role == "user" {
				msg := messages[i]
				msg.Content = p.addJSONSchemaInstructions(msg.Content, req.ResponseFormat)
				messages[i] = msg
				break
			}
		}
	}

	glmReq := map[string]any{
		"model":    req.Model,
		"messages": ConvertMessages(messages),
	}
	if stream {
		glmReq["stream"] = true
	}
	if req.MaxTokens != nil {
		glmReq["max_tokens"] = *req.MaxTokens
	}
	if req.Temperature != nil {
		glmReq["temperature"] = *req.Temperature
	}
	if req.TopP != nil {
		glmReq["top_p"] = *req.TopP
	}
	if len(req.Stop) > 0 {
		// GLM supports up to 1 stop word
		glmReq["stop"] = req.Stop[:1]
	}
	if len(req.Tools) > 0 {
		glmReq["tools"] = ConvertTools(req.Tools)
	}
	if req.ToolChoice != nil {
		glmReq["tool_choice"] = req.ToolChoice
	}

	if req.ResponseFormat != nil {
		if req.ResponseFormat.Type == "json_object" || req.ResponseFormat.Type == "json_schema" {
			glmReq["response_format"] = map[string]string{"type": "json_object"}
		}
	}

	// Handle thinking/reasoning mode
	// GLM thinking parameter supports: type (enabled/disabled), clear_thinking (boolean)
	// Note: GLM does NOT support budget_tokens like Anthropic
	if thinking := normalizeThinking(req); thinking != nil {
		switch thinking.Type {
		case "enabled":
			glmReq["thinking"] = map[string]any{"type": "enabled"}
		case "disabled":
			glmReq["thinking"] = map[string]any{"type": "disabled"}
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
	if stream {
		httpReq.Header.Set("Accept", "text/event-stream")
	}

	return httpReq, nil
}

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
	Role             string           `json:"role"`
	Content          string           `json:"content"`
	ToolCalls        []openaiToolCall `json:"tool_calls,omitempty"`
	ToolCallID       string           `json:"tool_call_id,omitempty"`
	ReasoningContent string           `json:"reasoning_content,omitempty"`
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
	resp             *http.Response
	scanner          *bufio.Scanner
	provider         string
	model            string
	includeReasoning bool
	done             bool
	usage            *Usage
}

func (r *glmStreamReader) Next() (*StreamChunk, error) {
	if r.done {
		return &StreamChunk{Done: true, Provider: r.provider, Model: r.model, Usage: r.usage}, nil
	}

	for r.scanner.Scan() {
		line := strings.TrimSpace(r.scanner.Text())

		// Skip empty lines and comments
		if line == "" || strings.HasPrefix(line, ":") {
			continue
		}

		// Handle data lines
		if data, found := strings.CutPrefix(line, "data: "); found {
			// Check for end of stream
			if data == "[DONE]" {
				r.done = true
				return &StreamChunk{Done: true, Provider: r.provider, Model: r.model, Usage: r.usage}, nil
			}

			// Parse JSON
			var streamResp glmStreamResponse
			if err := json.Unmarshal([]byte(data), &streamResp); err != nil {
				// Return error instead of silently ignoring malformed chunks
				return nil, fmt.Errorf("glm: failed to parse stream chunk: %w", err)
			}

			// Update model from response if available
			if streamResp.Model != "" {
				r.model = streamResp.Model
			}

			// Update usage if present
			if streamResp.Usage != nil {
				r.usage = &Usage{
					PromptTokens:     streamResp.Usage.PromptTokens,
					CompletionTokens: streamResp.Usage.CompletionTokens,
					TotalTokens:      streamResp.Usage.TotalTokens,
				}
			}

			// Convert to StreamChunk
			if len(streamResp.Choices) > 0 {
				choice := streamResp.Choices[0]
				chunk := &StreamChunk{
					Provider: r.provider,
					Model:    r.model,
				}

				if choice.Delta.Content != "" {
					chunk.Type = "content"
					chunk.Content = choice.Delta.Content
				}

				// Handle reasoning content
				if choice.Delta.ReasoningContent != "" && r.includeReasoning {
					chunk.Type = "reasoning"
					chunk.Reasoning = &ReasoningChunk{
						Content: choice.Delta.ReasoningContent,
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
					chunk.FinishReason = NormalizeFinishReason(choice.FinishReason)
					chunk.Done = true
					chunk.Usage = r.usage
					r.done = true
				}

				if chunk.Type != "" || chunk.Done {
					return chunk, nil
				}
			}
		}
	}

	// Check for scanner errors
	if err := r.scanner.Err(); err != nil {
		return nil, err
	}

	r.done = true
	return &StreamChunk{Done: true, Provider: r.provider, Model: r.model, Usage: r.usage}, nil
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
	Usage   *glmUsage         `json:"usage,omitempty"`
}

type glmStreamChoice struct {
	Index        int            `json:"index"`
	Delta        glmStreamDelta `json:"delta"`
	FinishReason string         `json:"finish_reason"`
}

type glmStreamDelta struct {
	Role             string           `json:"role,omitempty"`
	Content          string           `json:"content,omitempty"`
	ReasoningContent string           `json:"reasoning_content,omitempty"`
	ToolCalls        []openaiToolCall `json:"tool_calls,omitempty"`
}
