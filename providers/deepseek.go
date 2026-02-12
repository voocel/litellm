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
	RegisterBuiltin("deepseek", func(cfg ProviderConfig) Provider {
		return NewDeepSeek(cfg)
	}, "https://api.deepseek.com")
}

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

func (p *DeepSeekProvider) Chat(ctx context.Context, req *Request) (*Response, error) {
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
		return nil, NewHTTPError("deepseek", resp.StatusCode, string(body))
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
			PromptTokens:             deepseekResp.Usage.PromptTokens,
			CompletionTokens:         deepseekResp.Usage.CompletionTokens,
			TotalTokens:              deepseekResp.Usage.TotalTokens,
			ReasoningTokens:          reasoningTokens,
			CacheReadInputTokens:     deepseekResp.Usage.PromptCacheHitTokens,
			CacheCreationInputTokens: deepseekResp.Usage.PromptCacheMissTokens,
		}
	}

	if len(deepseekResp.Choices) > 0 {
		choice := deepseekResp.Choices[0]
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

		// Check for reasoning content (DeepSeek reasoner model)
		if !isThinkingDisabled(req) && strings.Contains(req.Model, "reasoner") && choice.Message.ReasoningContent != "" {
			reasoningTokens := 0
			if deepseekResp.Usage != nil && deepseekResp.Usage.CompletionTokensDetails != nil {
				reasoningTokens = deepseekResp.Usage.CompletionTokensDetails.ReasoningTokens
			}

			response.Reasoning = &ReasoningData{
				Content:    choice.Message.ReasoningContent,
				TokensUsed: reasoningTokens,
			}
		}
	}

	return response, nil
}

func (p *DeepSeekProvider) Stream(ctx context.Context, req *Request) (StreamReader, error) {
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
		return nil, NewHTTPError("deepseek", resp.StatusCode, string(body))
	}

	return &deepseekStreamReader{
		resp:             resp,
		scanner:          bufio.NewScanner(resp.Body),
		provider:         "deepseek",
		model:            req.Model,
		includeReasoning: !isThinkingDisabled(req),
	}, nil
}

// ListModels returns available models for DeepSeek.
func (p *DeepSeekProvider) ListModels(ctx context.Context) ([]ModelInfo, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}

	baseURL := strings.TrimSuffix(p.Config().BaseURL, "/")
	if baseURL == "" {
		baseURL = "https://api.deepseek.com"
	}
	url := fmt.Sprintf("%s/models", baseURL)
	httpReq, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("deepseek: create models request: %w", err)
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
		return nil, NewHTTPError("deepseek", resp.StatusCode, string(body))
	}

	var payload deepseekModelList
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return nil, fmt.Errorf("deepseek: decode models response: %w", err)
	}

	models := make([]ModelInfo, 0, len(payload.Data))
	for _, item := range payload.Data {
		models = append(models, ModelInfo{
			ID:       item.ID,
			Name:     item.ID,
			Provider: "deepseek",
			OwnedBy:  item.OwnedBy,
		})
	}

	return models, nil
}

func (p *DeepSeekProvider) buildHTTPRequest(ctx context.Context, req *Request, stream bool) (*http.Request, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}
	if err := p.BaseProvider.ValidateExtra(req.Extra, nil); err != nil {
		return nil, err
	}
	if err := p.BaseProvider.ValidateRequest(req); err != nil {
		return nil, err
	}

	deepseekReq := map[string]any{
		"model":    req.Model,
		"messages": ConvertMessages(req.Messages),
	}
	if stream {
		deepseekReq["stream"] = true
		// Enable usage reporting in streaming mode
		deepseekReq["stream_options"] = map[string]any{
			"include_usage": true,
		}
	}
	if req.MaxTokens != nil {
		deepseekReq["max_tokens"] = *req.MaxTokens
	}
	if req.Temperature != nil {
		deepseekReq["temperature"] = *req.Temperature
	}
	if len(req.Stop) > 0 {
		deepseekReq["stop"] = req.Stop
	}
	// Handle thinking parameter for deepseek-reasoner model
	thinking := normalizeThinking(req)
	if thinking.Type == "enabled" {
		deepseekReq["thinking"] = map[string]string{"type": "enabled"}
	} else if thinking.Type == "disabled" {
		deepseekReq["thinking"] = map[string]string{"type": "disabled"}
	}
	if len(req.Tools) > 0 {
		deepseekReq["tools"] = ConvertTools(req.Tools)
	}
	if req.ToolChoice != nil {
		deepseekReq["tool_choice"] = req.ToolChoice
	}
	if req.ResponseFormat != nil && req.ResponseFormat.Type == "json_object" {
		deepseekReq["response_format"] = map[string]string{"type": "json_object"}
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
	if stream {
		httpReq.Header.Set("Accept", "text/event-stream")
	}

	return httpReq, nil
}

// DeepSeek API structures (OpenAI compatible)
type deepseekResponse struct {
	ID      string           `json:"id"`
	Object  string           `json:"object"`
	Created int64            `json:"created"`
	Model   string           `json:"model"`
	Choices []deepseekChoice `json:"choices"`
	Usage   *deepseekUsage   `json:"usage,omitempty"`
}

type deepseekModelList struct {
	Data []deepseekModelInfo `json:"data"`
}

type deepseekModelInfo struct {
	ID      string `json:"id"`
	OwnedBy string `json:"owned_by,omitempty"`
}

type deepseekChoice struct {
	Index        int                 `json:"index"`
	Message      deepseekResponseMsg `json:"message"`
	FinishReason string              `json:"finish_reason"`
}

type deepseekResponseMsg struct {
	Role             string           `json:"role"`
	Content          string           `json:"content"`
	ReasoningContent string           `json:"reasoning_content,omitempty"`
	ToolCalls        []openaiToolCall `json:"tool_calls,omitempty"`
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
	resp             *http.Response
	scanner          *bufio.Scanner
	provider         string
	model            string
	includeReasoning bool
	done             bool
	usage            *Usage
}

func (r *deepseekStreamReader) Next() (*StreamChunk, error) {
	if r.done {
		return &StreamChunk{Done: true, Provider: r.provider, Model: r.model, Usage: r.usage}, nil
	}

	// DeepSeek uses OpenAI-compatible SSE format
	for r.scanner.Scan() {
		line := r.scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			r.done = true
			return &StreamChunk{Done: true, Provider: r.provider, Model: r.model, Usage: r.usage}, nil
		}

		var streamResp deepseekStreamResponse
		if err := json.Unmarshal([]byte(data), &streamResp); err != nil {
			// Return error instead of silently ignoring malformed chunks
			return nil, fmt.Errorf("deepseek: failed to parse stream chunk: %w", err)
		}

		// Update model from response if available
		if streamResp.Model != "" {
			r.model = streamResp.Model
		}

		// Handle usage chunk (sent before [DONE] when stream_options.include_usage is true)
		if streamResp.Usage != nil {
			reasoningTokens := 0
			if streamResp.Usage.CompletionTokensDetails != nil {
				reasoningTokens = streamResp.Usage.CompletionTokensDetails.ReasoningTokens
			}
			r.usage = &Usage{
				PromptTokens:             streamResp.Usage.PromptTokens,
				CompletionTokens:         streamResp.Usage.CompletionTokens,
				TotalTokens:              streamResp.Usage.TotalTokens,
				ReasoningTokens:          reasoningTokens,
				CacheReadInputTokens:     streamResp.Usage.PromptCacheHitTokens,
				CacheCreationInputTokens: streamResp.Usage.PromptCacheMissTokens,
			}
		}

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

			// Only return chunk if it has content
			if chunk.Type != "" {
				return chunk, nil
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
	Usage   *deepseekUsage         `json:"usage,omitempty"` // Present when stream_options.include_usage is true
}

type deepseekStreamChoice struct {
	Index        int                 `json:"index"`
	Delta        deepseekStreamDelta `json:"delta"`
	FinishReason string              `json:"finish_reason"`
}

type deepseekStreamDelta struct {
	Role             string           `json:"role,omitempty"`
	Content          string           `json:"content,omitempty"`
	ReasoningContent string           `json:"reasoning_content,omitempty"`
	ToolCalls        []openaiToolCall `json:"tool_calls,omitempty"`
}
