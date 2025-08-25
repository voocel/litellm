package providers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// DeepSeekProvider implements DeepSeek API integration
type DeepSeekProvider struct {
	*BaseProvider
}

// NewDeepSeek creates a new DeepSeek provider
func NewDeepSeek(config ProviderConfig) *DeepSeekProvider {
	baseProvider := NewBaseProvider("deepseek", config)
	return &DeepSeekProvider{
		BaseProvider: baseProvider,
	}
}

func (p *DeepSeekProvider) SupportsModel(model string) bool {
	for _, m := range p.Models() {
		if m.ID == model {
			return true
		}
	}
	return false
}

func (p *DeepSeekProvider) Models() []ModelInfo {
	return []ModelInfo{
		{
			ID: "deepseek-chat", Provider: "deepseek", Name: "DeepSeek Chat V3.1", MaxTokens: 64000,
			Capabilities: []string{"chat", "function_call", "tool_use"},
		},
		{
			ID: "deepseek-reasoner", Provider: "deepseek", Name: "DeepSeek Reasoner V3.1", MaxTokens: 64000,
			Capabilities: []string{"chat", "reasoning", "function_call", "tool_use"},
		},
		{
			ID: "deepseek-coder", Provider: "deepseek", Name: "DeepSeek Coder", MaxTokens: 64000,
			Capabilities: []string{"chat", "code", "function_call"},
		},
	}
}

func (p *DeepSeekProvider) Chat(ctx context.Context, req *Request) (*Response, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}

	// Build DeepSeek request (OpenAI compatible)
	deepseekReq := map[string]any{
		"model":    req.Model,
		"messages": p.convertMessages(req.Messages),
	}

	if req.MaxTokens != nil {
		deepseekReq["max_tokens"] = *req.MaxTokens
	}
	if req.Temperature != nil {
		deepseekReq["temperature"] = *req.Temperature
	}
	if len(req.Tools) > 0 {
		deepseekReq["tools"] = p.convertTools(req.Tools)
	}
	if req.ToolChoice != nil {
		deepseekReq["tool_choice"] = req.ToolChoice
	}

	// Handle response format
	if req.ResponseFormat != nil {
		if req.ResponseFormat.Type == "json_object" {
			deepseekReq["response_format"] = map[string]string{"type": "json_object"}
		}
	}

	body, err := json.Marshal(deepseekReq)
	if err != nil {
		return nil, fmt.Errorf("deepseek: marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/chat/completions", p.Config().BaseURL)
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("deepseek: create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+p.Config().APIKey)

	resp, err := p.HTTPClient().Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("deepseek: request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("deepseek: API error %d: %s", resp.StatusCode, string(body))
	}

	var deepseekResp deepseekResponse
	if err := json.NewDecoder(resp.Body).Decode(&deepseekResp); err != nil {
		return nil, fmt.Errorf("deepseek: decode response: %w", err)
	}

	response := &Response{
		Model:    deepseekResp.Model,
		Provider: "deepseek",
	}

	if deepseekResp.Usage != nil {
		reasoningTokens := 0
		if deepseekResp.Usage.CompletionTokensDetails != nil {
			reasoningTokens = deepseekResp.Usage.CompletionTokensDetails.ReasoningTokens
		}

		response.Usage = Usage{
			PromptTokens:     deepseekResp.Usage.PromptTokens,
			CompletionTokens: deepseekResp.Usage.CompletionTokens,
			TotalTokens:      deepseekResp.Usage.TotalTokens,
			ReasoningTokens:  reasoningTokens,
		}
	}

	if len(deepseekResp.Choices) > 0 {
		choice := deepseekResp.Choices[0]
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

		// Check for reasoning content (DeepSeek reasoner model)
		if strings.Contains(req.Model, "reasoner") && choice.Message.ReasoningContent != "" {
			reasoningTokens := 0
			if deepseekResp.Usage != nil && deepseekResp.Usage.CompletionTokensDetails != nil {
				reasoningTokens = deepseekResp.Usage.CompletionTokensDetails.ReasoningTokens
			}

			response.Reasoning = &ReasoningData{
				Content:    choice.Message.ReasoningContent,
				Summary:    "DeepSeek reasoning process",
				TokensUsed: reasoningTokens,
			}
		}
	}

	return response, nil
}

func (p *DeepSeekProvider) Stream(ctx context.Context, req *Request) (StreamReader, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}

	// Build request (similar to Chat but with stream: true)
	deepseekReq := map[string]any{
		"model":    req.Model,
		"messages": p.convertMessages(req.Messages),
		"stream":   true,
	}

	if req.MaxTokens != nil {
		deepseekReq["max_tokens"] = *req.MaxTokens
	}
	if req.Temperature != nil {
		deepseekReq["temperature"] = *req.Temperature
	}
	if len(req.Tools) > 0 {
		deepseekReq["tools"] = p.convertTools(req.Tools)
	}

	body, err := json.Marshal(deepseekReq)
	if err != nil {
		return nil, fmt.Errorf("deepseek: marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/chat/completions", p.Config().BaseURL)
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("deepseek: create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+p.Config().APIKey)
	httpReq.Header.Set("Accept", "text/event-stream")

	resp, err := p.HTTPClient().Do(httpReq)
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
		provider: "deepseek",
	}, nil
}

// convertMessages converts standard messages to DeepSeek format
func (p *DeepSeekProvider) convertMessages(messages []Message) []deepseekMessage {
	deepseekMessages := make([]deepseekMessage, len(messages))
	for i, msg := range messages {
		deepseekMessages[i] = deepseekMessage{
			Role:    msg.Role,
			Content: msg.Content,
		}

		if len(msg.ToolCalls) > 0 {
			for _, toolCall := range msg.ToolCalls {
				deepseekMessages[i].ToolCalls = append(deepseekMessages[i].ToolCalls, deepseekToolCall{
					ID:   toolCall.ID,
					Type: toolCall.Type,
					Function: deepseekFunctionCall{
						Name:      toolCall.Function.Name,
						Arguments: toolCall.Function.Arguments,
					},
				})
			}
		}

		if msg.ToolCallID != "" {
			deepseekMessages[i].ToolCallID = msg.ToolCallID
		}
	}
	return deepseekMessages
}

// convertTools converts standard tools to DeepSeek format
func (p *DeepSeekProvider) convertTools(tools []Tool) []deepseekTool {
	deepseekTools := make([]deepseekTool, len(tools))
	for i, tool := range tools {
		deepseekTools[i] = deepseekTool{
			Type: tool.Type,
			Function: deepseekFunction{
				Name:        tool.Function.Name,
				Description: tool.Function.Description,
				Parameters:  tool.Function.Parameters,
			},
		}
	}
	return deepseekTools
}

// DeepSeek API structures (OpenAI compatible)
type deepseekMessage struct {
	Role       string             `json:"role"`
	Content    string             `json:"content"`
	ToolCalls  []deepseekToolCall `json:"tool_calls,omitempty"`
	ToolCallID string             `json:"tool_call_id,omitempty"`
}

type deepseekToolCall struct {
	ID       string               `json:"id"`
	Type     string               `json:"type"`
	Function deepseekFunctionCall `json:"function"`
}

type deepseekFunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type deepseekTool struct {
	Type     string           `json:"type"`
	Function deepseekFunction `json:"function"`
}

type deepseekFunction struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Parameters  any    `json:"parameters"`
}

type deepseekResponse struct {
	ID      string           `json:"id"`
	Object  string           `json:"object"`
	Created int64            `json:"created"`
	Model   string           `json:"model"`
	Choices []deepseekChoice `json:"choices"`
	Usage   *deepseekUsage   `json:"usage,omitempty"`
}

type deepseekChoice struct {
	Index        int                 `json:"index"`
	Message      deepseekResponseMsg `json:"message"`
	FinishReason string              `json:"finish_reason"`
}

type deepseekResponseMsg struct {
	Role             string             `json:"role"`
	Content          string             `json:"content"`
	ReasoningContent string             `json:"reasoning_content,omitempty"`
	ToolCalls        []deepseekToolCall `json:"tool_calls,omitempty"`
}

type deepseekUsage struct {
	PromptTokens            int                              `json:"prompt_tokens"`
	CompletionTokens        int                              `json:"completion_tokens"`
	TotalTokens             int                              `json:"total_tokens"`
	PromptCacheHitTokens    int                              `json:"prompt_cache_hit_tokens,omitempty"`
	PromptCacheMissTokens   int                              `json:"prompt_cache_miss_tokens,omitempty"`
	CompletionTokensDetails *deepseekCompletionTokensDetails `json:"completion_tokens_details,omitempty"`
}

type deepseekCompletionTokensDetails struct {
	ReasoningTokens int `json:"reasoning_tokens,omitempty"`
}

// deepseekStreamReader implements streaming for DeepSeek
type deepseekStreamReader struct {
	resp     *http.Response
	provider string
	done     bool
}

func (r *deepseekStreamReader) Next() (*StreamChunk, error) {
	if r.done {
		return &StreamChunk{Done: true, Provider: r.provider}, nil
	}

	// DeepSeek uses OpenAI-compatible SSE format
	buffer := make([]byte, 4096)
	n, err := r.resp.Body.Read(buffer)
	if err != nil {
		if err == io.EOF {
			r.done = true
			return &StreamChunk{Done: true, Provider: r.provider}, nil
		}
		return nil, err
	}

	data := strings.TrimSpace(string(buffer[:n]))
	lines := strings.Split(data, "\n")

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "data: ") {
			jsonData := strings.TrimPrefix(line, "data: ")
			if jsonData == "[DONE]" {
				r.done = true
				return &StreamChunk{Done: true, Provider: r.provider}, nil
			}

			var streamResp deepseekStreamResponse
			if err := json.Unmarshal([]byte(jsonData), &streamResp); err != nil {
				continue // Skip invalid JSON
			}

			if len(streamResp.Choices) > 0 {
				choice := streamResp.Choices[0]
				chunk := &StreamChunk{
					Provider: r.provider,
				}

				if choice.Delta.Content != "" {
					chunk.Type = "content"
					chunk.Content = choice.Delta.Content
				}

				if choice.Delta.ReasoningContent != "" {
					chunk.Type = "reasoning"
					chunk.Reasoning = &ReasoningChunk{
						Content: choice.Delta.ReasoningContent,
						Summary: "DeepSeek reasoning process",
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

	return &StreamChunk{Provider: r.provider}, nil
}

func (r *deepseekStreamReader) Close() error {
	return r.resp.Body.Close()
}

// Streaming response structures
type deepseekStreamResponse struct {
	ID      string                 `json:"id"`
	Object  string                 `json:"object"`
	Created int64                  `json:"created"`
	Model   string                 `json:"model"`
	Choices []deepseekStreamChoice `json:"choices"`
}

type deepseekStreamChoice struct {
	Index        int                 `json:"index"`
	Delta        deepseekStreamDelta `json:"delta"`
	FinishReason string              `json:"finish_reason"`
}

type deepseekStreamDelta struct {
	Role             string             `json:"role,omitempty"`
	Content          string             `json:"content,omitempty"`
	ReasoningContent string             `json:"reasoning_content,omitempty"`
	ToolCalls        []deepseekToolCall `json:"tool_calls,omitempty"`
}
