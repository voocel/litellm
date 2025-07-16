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

// OpenAIProvider implements the Provider interface for OpenAI
type OpenAIProvider struct {
	*BaseProvider
}

// NewOpenAIProvider creates a new OpenAI provider
func NewOpenAIProvider(config ProviderConfig) Provider {
	return &OpenAIProvider{
		BaseProvider: NewBaseProvider("openai", config),
	}
}

// Models returns the list of supported models
func (p *OpenAIProvider) Models() []ModelInfo {
	return []ModelInfo{
		{
			ID: "gpt-4o", Provider: "openai", Name: "GPT-4o", MaxTokens: 128000,
			Capabilities: []ModelCapability{CapabilityChat, CapabilityFunctionCall, CapabilityVision},
		},
		{
			ID: "gpt-4o-mini", Provider: "openai", Name: "GPT-4o Mini", MaxTokens: 128000,
			Capabilities: []ModelCapability{CapabilityChat, CapabilityFunctionCall, CapabilityVision},
		},
		{
			ID: "gpt-4-turbo", Provider: "openai", Name: "GPT-4 Turbo", MaxTokens: 128000,
			Capabilities: []ModelCapability{CapabilityChat, CapabilityFunctionCall, CapabilityVision},
		},
		{
			ID: "gpt-3.5-turbo", Provider: "openai", Name: "GPT-3.5 Turbo", MaxTokens: 16385,
			Capabilities: []ModelCapability{CapabilityChat, CapabilityFunctionCall},
		},
		{
			ID: "o1", Provider: "openai", Name: "OpenAI o1", MaxTokens: 100000,
			Capabilities: []ModelCapability{CapabilityChat, CapabilityReasoning},
		},
		{
			ID: "o1-mini", Provider: "openai", Name: "OpenAI o1 Mini", MaxTokens: 65536,
			Capabilities: []ModelCapability{CapabilityChat, CapabilityReasoning},
		},
		{
			ID: "o3-mini", Provider: "openai", Name: "OpenAI o3 Mini", MaxTokens: 100000,
			Capabilities: []ModelCapability{CapabilityChat, CapabilityReasoning},
		},
	}
}

// isReasoningModel checks if the model is a reasoning model (o-series)
func (p *OpenAIProvider) isReasoningModel(model string) bool {
	model = strings.ToLower(model)
	return strings.HasPrefix(model, "o1") ||
		strings.HasPrefix(model, "o3") ||
		strings.HasPrefix(model, "o4")
}

// OpenAI API request/response structures
type openaiRequest struct {
	Model               string           `json:"model"`
	Messages            []openaiMessage  `json:"messages"`
	MaxTokens           *int             `json:"max_tokens,omitempty"`
	MaxCompletionTokens *int             `json:"max_completion_tokens,omitempty"`
	Temperature         *float64         `json:"temperature,omitempty"`
	Stream              bool             `json:"stream,omitempty"`
	Tools               []openaiTool     `json:"tools,omitempty"`
	ToolChoice          interface{}      `json:"tool_choice,omitempty"`
	Reasoning           *openaiReasoning `json:"reasoning,omitempty"`
}

type openaiReasoning struct {
	Effort  string `json:"effort,omitempty"`
	Summary string `json:"summary,omitempty"`
}

type openaiMessage struct {
	Role       string           `json:"role"`
	Content    string           `json:"content,omitempty"`
	ToolCalls  []openaiToolCall `json:"tool_calls,omitempty"`
	ToolCallID string           `json:"tool_call_id,omitempty"`
}

type openaiTool struct {
	Type     string             `json:"type"`
	Function openaiToolFunction `json:"function"`
}

type openaiToolFunction struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Parameters  any    `json:"parameters"`
}

type openaiToolCall struct {
	ID       string             `json:"id"`
	Type     string             `json:"type"`
	Function openaiToolCallFunc `json:"function"`
}

type openaiToolCallFunc struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type openaiResponse struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Model   string         `json:"model"`
	Choices []openaiChoice `json:"choices"`
	Usage   openaiUsage    `json:"usage"`
}

type openaiChoice struct {
	Index            int                     `json:"index"`
	Message          openaiMessage           `json:"message"`
	Delta            openaiDelta             `json:"delta,omitempty"`
	ReasoningSummary *openaiReasoningSummary `json:"reasoning_summary,omitempty"`
	FinishReason     string                  `json:"finish_reason,omitempty"`
}

type openaiReasoningSummary struct {
	Text string `json:"text,omitempty"`
}

type openaiDelta struct {
	Content          string                  `json:"content,omitempty"`
	ToolCalls        []openaiToolCall        `json:"tool_calls,omitempty"`
	ReasoningSummary *openaiReasoningSummary `json:"reasoning_summary,omitempty"`
}

type openaiUsage struct {
	PromptTokens            int                            `json:"prompt_tokens"`
	CompletionTokens        int                            `json:"completion_tokens"`
	TotalTokens             int                            `json:"total_tokens"`
	CompletionTokensDetails *openaiCompletionTokensDetails `json:"completion_tokens_details,omitempty"`
}

type openaiCompletionTokensDetails struct {
	ReasoningTokens int `json:"reasoning_tokens,omitempty"`
}

type openaiStreamChunk struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Model   string         `json:"model"`
	Choices []openaiChoice `json:"choices"`
}

// Responses API structures
type responsesAPIRequest struct {
	Model     string                 `json:"model"`
	Input     interface{}            `json:"input"`
	Reasoning *responsesAPIReasoning `json:"reasoning,omitempty"`
	Tools     []openaiTool           `json:"tools,omitempty"`
}

type responsesAPIReasoning struct {
	Effort  string `json:"effort,omitempty"`
	Summary string `json:"summary,omitempty"`
}

type responsesAPIResponse struct {
	ID         string                   `json:"id"`
	Object     string                   `json:"object"`
	Model      string                   `json:"model"`
	OutputText string                   `json:"output_text"`
	Output     []responsesAPIOutputItem `json:"output"`
	Usage      responsesAPIUsage        `json:"usage"`
}

type responsesAPIOutputItem struct {
	Type    string                    `json:"type"`
	Content string                    `json:"content,omitempty"`
	Summary []responsesAPISummaryItem `json:"summary,omitempty"`
}

type responsesAPISummaryItem struct {
	Text string `json:"text"`
}

type responsesAPIUsage struct {
	InputTokens         int                              `json:"input_tokens"`
	OutputTokens        int                              `json:"output_tokens"`
	TotalTokens         int                              `json:"total_tokens"`
	OutputTokensDetails *responsesAPIOutputTokensDetails `json:"output_tokens_details,omitempty"`
}

type responsesAPIOutputTokensDetails struct {
	ReasoningTokens int `json:"reasoning_tokens,omitempty"`
}

func (p *OpenAIProvider) Complete(ctx context.Context, req *Request) (*Response, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}

	modelName := req.Model

	// Check if should use Responses API
	shouldUseResponsesAPI := req.UseResponsesAPI ||
		(p.isReasoningModel(modelName) && (req.ReasoningEffort != "" || req.ReasoningSummary != ""))

	if shouldUseResponsesAPI {
		// Try using Responses API
		response, err := p.completeWithResponsesAPI(ctx, req, modelName)
		if err != nil {
			// If Responses API fails, fall back to traditional mode
			return p.completeWithChatAPI(ctx, req, modelName)
		}
		return response, nil
	}

	// Use traditional Chat Completions API
	return p.completeWithChatAPI(ctx, req, modelName)
}

// completeWithChatAPI uses traditional Chat Completions API
func (p *OpenAIProvider) completeWithChatAPI(ctx context.Context, req *Request, modelName string) (*Response, error) {
	openaiReq := openaiRequest{
		Model:      modelName,
		Stream:     false,
		ToolChoice: req.ToolChoice,
	}

	// Set correct parameters based on model type
	if p.isReasoningModel(modelName) {
		// Parameter settings for o series models (reasoning models)
		openaiReq.MaxCompletionTokens = req.MaxTokens
		// o series models don't support temperature and other parameters

		// Set reasoning parameters
		reasoning := &openaiReasoning{}
		if req.ReasoningEffort != "" {
			reasoning.Effort = req.ReasoningEffort
		}
		if req.ReasoningSummary != "" {
			reasoning.Summary = req.ReasoningSummary
		} else {
			reasoning.Summary = "auto"
		}

		if reasoning.Effort != "" || reasoning.Summary != "" {
			openaiReq.Reasoning = reasoning
		}
	} else {
		// Traditional models support all parameters
		openaiReq.MaxTokens = req.MaxTokens
		openaiReq.Temperature = req.Temperature
	}

	// Convert tool definitions
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

	// Convert messages
	openaiReq.Messages = make([]openaiMessage, len(req.Messages))
	for i, msg := range req.Messages {
		openaiMsg := openaiMessage{
			Role:       msg.Role,
			Content:    msg.Content,
			ToolCallID: msg.ToolCallID,
		}

		// Convert tool calls
		if len(msg.ToolCalls) > 0 {
			openaiMsg.ToolCalls = make([]openaiToolCall, len(msg.ToolCalls))
			for j, toolCall := range msg.ToolCalls {
				openaiMsg.ToolCalls[j] = openaiToolCall{
					ID:   toolCall.ID,
					Type: toolCall.Type,
					Function: openaiToolCallFunc{
						Name:      toolCall.Function.Name,
						Arguments: toolCall.Function.Arguments,
					},
				}
			}
		}

		openaiReq.Messages[i] = openaiMsg
	}

	body, err := json.Marshal(openaiReq)
	if err != nil {
		return nil, fmt.Errorf("openai: marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.config.BaseURL+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("openai: create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+p.config.APIKey)

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("openai: request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("openai: API error %d: %s", resp.StatusCode, string(body))
	}

	var openaiResp openaiResponse
	if err := json.NewDecoder(resp.Body).Decode(&openaiResp); err != nil {
		return nil, fmt.Errorf("openai: decode response: %w", err)
	}

	response := &Response{
		Usage: Usage{
			PromptTokens:     openaiResp.Usage.PromptTokens,
			CompletionTokens: openaiResp.Usage.CompletionTokens,
			TotalTokens:      openaiResp.Usage.TotalTokens,
		},
		Model:    openaiResp.Model,
		Provider: "openai",
	}

	// Handle reasoning tokens for reasoning models
	if openaiResp.Usage.CompletionTokensDetails != nil {
		response.Usage.ReasoningTokens = openaiResp.Usage.CompletionTokensDetails.ReasoningTokens
	}

	if len(openaiResp.Choices) > 0 {
		choice := openaiResp.Choices[0]
		response.Content = choice.Message.Content
		response.FinishReason = choice.FinishReason

		// Extract reasoning summary content
		if choice.ReasoningSummary != nil && choice.ReasoningSummary.Text != "" {
			response.Reasoning = &ReasoningData{
				Summary:    choice.ReasoningSummary.Text,
				TokensUsed: response.Usage.ReasoningTokens,
			}
		}

		// Convert tool calls
		if len(choice.Message.ToolCalls) > 0 {
			response.ToolCalls = make([]ToolCall, len(choice.Message.ToolCalls))
			for i, toolCall := range choice.Message.ToolCalls {
				response.ToolCalls[i] = ToolCall{
					ID:   toolCall.ID,
					Type: toolCall.Type,
					Function: FunctionCall{
						Name:      toolCall.Function.Name,
						Arguments: toolCall.Function.Arguments,
					},
				}
			}
		}
	}

	return response, nil
}

// completeWithResponsesAPI uses OpenAI Responses API to handle reasoning models
func (p *OpenAIProvider) completeWithResponsesAPI(ctx context.Context, req *Request, modelName string) (*Response, error) {
	// Build Responses API request
	responsesReq := responsesAPIRequest{
		Model: modelName,
		Input: req.Messages, // Pass messages array directly
	}

	// Set reasoning parameters
	if req.ReasoningEffort != "" || req.ReasoningSummary != "" {
		reasoning := &responsesAPIReasoning{}
		if req.ReasoningEffort != "" {
			reasoning.Effort = req.ReasoningEffort
		}
		if req.ReasoningSummary != "" {
			reasoning.Summary = req.ReasoningSummary
		} else {
			reasoning.Summary = "auto"
		}
		responsesReq.Reasoning = reasoning
	}

	// Convert tool definitions
	if len(req.Tools) > 0 {
		responsesReq.Tools = make([]openaiTool, len(req.Tools))
		for i, tool := range req.Tools {
			responsesReq.Tools[i] = openaiTool{
				Type: tool.Type,
				Function: openaiToolFunction{
					Name:        tool.Function.Name,
					Description: tool.Function.Description,
					Parameters:  tool.Function.Parameters,
				},
			}
		}
	}

	body, err := json.Marshal(responsesReq)
	if err != nil {
		return nil, fmt.Errorf("openai: marshal responses request: %w", err)
	}

	// Use Responses API
	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.config.BaseURL+"/v1/responses", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("openai: create responses request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+p.config.APIKey)

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("openai: responses request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("openai: responses API error %d: %s", resp.StatusCode, string(body))
	}

	var responsesResp responsesAPIResponse
	if err := json.NewDecoder(resp.Body).Decode(&responsesResp); err != nil {
		return nil, fmt.Errorf("openai: decode responses response: %w", err)
	}

	// Build unified response format
	response := &Response{
		Content: responsesResp.OutputText,
		Usage: Usage{
			PromptTokens:     responsesResp.Usage.InputTokens,
			CompletionTokens: responsesResp.Usage.OutputTokens,
			TotalTokens:      responsesResp.Usage.TotalTokens,
		},
		Model:    responsesResp.Model,
		Provider: "openai",
	}

	if responsesResp.Usage.OutputTokensDetails != nil {
		response.Usage.ReasoningTokens = responsesResp.Usage.OutputTokensDetails.ReasoningTokens
	}

	// Extract reasoning summary content
	for _, outputItem := range responsesResp.Output {
		if outputItem.Type == "reasoning" && len(outputItem.Summary) > 0 {
			// Merge all reasoning summary texts
			var reasoningText strings.Builder
			for _, summary := range outputItem.Summary {
				if summary.Text != "" {
					if reasoningText.Len() > 0 {
						reasoningText.WriteString("\n")
					}
					reasoningText.WriteString(summary.Text)
				}
			}
			if reasoningText.Len() > 0 {
				response.Reasoning = &ReasoningData{
					Content:    reasoningText.String(),
					TokensUsed: response.Usage.ReasoningTokens,
				}
			}
			break
		}
	}

	return response, nil
}

func (p *OpenAIProvider) Stream(ctx context.Context, req *Request) (StreamReader, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}

	modelName := req.Model

	openaiReq := openaiRequest{
		Model:    modelName,
		Messages: make([]openaiMessage, len(req.Messages)),
		Stream:   true,
	}

	// Set correct parameters based on model type
	if p.isReasoningModel(modelName) {
		openaiReq.MaxCompletionTokens = req.MaxTokens
		reasoning := &openaiReasoning{}
		if req.ReasoningEffort != "" {
			reasoning.Effort = req.ReasoningEffort
		}
		if req.ReasoningSummary != "" {
			reasoning.Summary = req.ReasoningSummary
		} else {
			reasoning.Summary = "auto"
		}

		if reasoning.Effort != "" || reasoning.Summary != "" {
			openaiReq.Reasoning = reasoning
		}
	} else {
		openaiReq.MaxTokens = req.MaxTokens
		openaiReq.Temperature = req.Temperature
	}

	for i, msg := range req.Messages {
		openaiReq.Messages[i] = openaiMessage{
			Role:    msg.Role,
			Content: msg.Content,
		}
	}

	body, err := json.Marshal(openaiReq)
	if err != nil {
		return nil, fmt.Errorf("openai: marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.config.BaseURL+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("openai: create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+p.config.APIKey)

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("openai: request failed: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("openai: API error %d: %s", resp.StatusCode, string(body))
	}

	return &openaiStreamReader{
		resp:     resp,
		scanner:  bufio.NewScanner(resp.Body),
		provider: "openai",
	}, nil
}

// openaiStreamReader implements StreamReader for OpenAI
type openaiStreamReader struct {
	resp     *http.Response
	scanner  *bufio.Scanner
	provider string
	err      error
	done     bool
}

func (r *openaiStreamReader) Read() (*StreamChunk, error) {
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

		var chunk openaiStreamChunk
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

			// Handle streaming reasoning summary content
			if choice.Delta.ReasoningSummary != nil && choice.Delta.ReasoningSummary.Text != "" {
				streamChunk.Type = ChunkTypeReasoning
				streamChunk.Reasoning = &ReasoningChunk{
					Summary: choice.Delta.ReasoningSummary.Text,
				}
			}

			if choice.FinishReason != "" {
				streamChunk.FinishReason = choice.FinishReason
			}

			return streamChunk, nil
		}
	}

	if err := r.scanner.Err(); err != nil {
		r.err = err
		return nil, err
	}

	r.done = true
	return &StreamChunk{Done: true, Provider: r.provider}, nil
}

func (r *openaiStreamReader) Close() error {
	return r.resp.Body.Close()
}

func (r *openaiStreamReader) Err() error {
	return r.err
}

func init() {
	RegisterProvider("openai", NewOpenAIProvider)
}
