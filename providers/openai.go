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

// OpenAIProvider implements OpenAI API integration
type OpenAIProvider struct {
	*BaseProvider
}

// NewOpenAI creates a new OpenAI provider
func NewOpenAI(config ProviderConfig) *OpenAIProvider {
	baseProvider := NewBaseProvider("openai", config)

	return &OpenAIProvider{
		BaseProvider: baseProvider,
	}
}

func (p *OpenAIProvider) SupportsModel(model string) bool {
	for _, m := range p.Models() {
		if m.ID == model {
			return true
		}
	}
	return false
}

func (p *OpenAIProvider) Models() []ModelInfo {
	return []ModelInfo{
		{
			ID: "gpt-5", Provider: "openai", Name: "GPT-5", MaxTokens: 128000,
			Capabilities: []string{"chat", "function_call", "vision", "code"},
		},
		{
			ID: "gpt-4o", Provider: "openai", Name: "GPT-4o", MaxTokens: 128000,
			Capabilities: []string{"chat", "function_call", "vision"},
		},
		{
			ID: "gpt-4o-mini", Provider: "openai", Name: "GPT-4o Mini", MaxTokens: 128000,
			Capabilities: []string{"chat", "function_call", "vision"},
		},
		{
			ID: "gpt-4.1", Provider: "openai", Name: "GPT-4.1", MaxTokens: 128000,
			Capabilities: []string{"chat", "function_call", "vision"},
		},
		{
			ID: "gpt-4.1-mini", Provider: "openai", Name: "GPT-4.1 Mini", MaxTokens: 16385,
			Capabilities: []string{"chat", "function_call"},
		},
		{
			ID: "gpt-4.1-nano", Provider: "openai", Name: "GPT-4.1 Nano", MaxTokens: 16385,
			Capabilities: []string{"chat", "function_call"},
		},
		{
			ID: "o3", Provider: "openai", Name: "OpenAI o3", MaxTokens: 100000,
			Capabilities: []string{"chat", "reasoning"},
		},
		{
			ID: "o3-mini", Provider: "openai", Name: "OpenAI o3 Mini", MaxTokens: 65536,
			Capabilities: []string{"chat", "reasoning"},
		},
		{
			ID: "o4-mini", Provider: "openai", Name: "OpenAI o4 Mini", MaxTokens: 100000,
			Capabilities: []string{"chat", "reasoning"},
		},
	}
}

// isReasoningModel checks if the model supports advanced reasoning (o-series and GPT-5)
func (p *OpenAIProvider) isReasoningModel(model string) bool {
	modelLower := strings.ToLower(model)
	return strings.HasPrefix(modelLower, "o1") ||
		strings.HasPrefix(modelLower, "o3") ||
		strings.HasPrefix(modelLower, "o4") ||
		strings.HasPrefix(modelLower, "gpt-5")
}

func (p *OpenAIProvider) Chat(ctx context.Context, req *Request) (*Response, error) {
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
			fallbackReq := *req
			fallbackReq.UseResponsesAPI = false

			// Only clear reasoning parameters for OpenAI endpoints
			if !strings.Contains(p.Config().BaseURL, "openrouter") {
				fallbackReq.ReasoningEffort = ""
				fallbackReq.ReasoningSummary = ""
			}

			return p.completeWithChatAPI(ctx, &fallbackReq, modelName)
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

	// Convert response format
	if req.ResponseFormat != nil {
		openaiReq.ResponseFormat = p.convertResponseFormat(req.ResponseFormat)
	}

	// Set reasoning parameters if provided (OpenAI format only)
	if req.ReasoningEffort != "" || req.ReasoningSummary != "" {
		reasoning := &openaiReasoning{}

		if req.ReasoningEffort != "" {
			reasoning.Effort = req.ReasoningEffort
		}
		if req.ReasoningSummary != "" {
			reasoning.Summary = req.ReasoningSummary
		} else {
			reasoning.Summary = "auto"
		}

		openaiReq.Reasoning = reasoning
	}

	// Set correct parameters based on model type
	if p.isReasoningModel(modelName) {
		// Reasoning models always use max_completion_tokens
		openaiReq.MaxCompletionTokens = req.MaxTokens
		// Only set reasoning parameters if provided
		if req.ReasoningEffort != "" || req.ReasoningSummary != "" {
			// Reasoning parameters provided - these will be used
		}
		// Reasoning models typically ignore temperature
	} else {
		// Traditional models use max_tokens and temperature
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

	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.buildURL("/chat/completions"), bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("openai: create request: %w", err)
	}

	p.setHeaders(httpReq)

	resp, err := p.HTTPClient().Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("openai: request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("openai: API error %d: %s", resp.StatusCode, string(body))
	}

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("openai: read response: %w", err)
	}

	var openaiResp openaiResponse
	if err := json.Unmarshal(respBody, &openaiResp); err != nil {
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

		// Extract reasoning content (OpenAI format only)
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
	responsesReq := responsesAPIRequest{
		Model: modelName,
		Input: req.Messages,
	}

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
	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.buildURL("/responses"), bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("openai: create responses request: %w", err)
	}

	p.setHeaders(httpReq)

	resp, err := p.HTTPClient().Do(httpReq)
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

	// Convert response format
	if req.ResponseFormat != nil {
		openaiReq.ResponseFormat = p.convertResponseFormat(req.ResponseFormat)
	}

	// Set correct parameters based on model type
	if p.isReasoningModel(modelName) {
		// Reasoning models always use max_completion_tokens
		openaiReq.MaxCompletionTokens = req.MaxTokens
		// Only set reasoning parameters if provided
		if req.ReasoningEffort != "" || req.ReasoningSummary != "" {
			reasoning := &openaiReasoning{}
			if req.ReasoningEffort != "" {
				reasoning.Effort = req.ReasoningEffort
			}
			if req.ReasoningSummary != "" {
				reasoning.Summary = req.ReasoningSummary
			} else {
				reasoning.Summary = "auto"
			}
			openaiReq.Reasoning = reasoning
		}
	} else {
		// Traditional models use max_tokens and temperature
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

	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.buildURL("/chat/completions"), bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("openai: create request: %w", err)
	}

	p.setHeaders(httpReq)

	resp, err := p.HTTPClient().Do(httpReq)
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

func (p *OpenAIProvider) convertMessages(messages []Message) []openaiMessage {
	openaiMessages := make([]openaiMessage, len(messages))
	for i, msg := range messages {
		openaiMessages[i] = openaiMessage{
			Role:    msg.Role,
			Content: msg.Content,
		}
	}
	return openaiMessages
}

func (p *OpenAIProvider) convertResponse(resp *openaiResponse) *Response {
	response := &Response{
		Usage: Usage{
			PromptTokens:     resp.Usage.PromptTokens,
			CompletionTokens: resp.Usage.CompletionTokens,
			TotalTokens:      resp.Usage.TotalTokens,
		},
		Model:    resp.Model,
		Provider: "openai",
	}

	if len(resp.Choices) > 0 {
		choice := resp.Choices[0]
		response.Content = choice.Message.Content
		response.FinishReason = choice.FinishReason
	}

	return response
}

func (p *OpenAIProvider) buildURL(endpoint string) string {
	baseURL := strings.TrimSuffix(p.Config().BaseURL, "/")
	if baseURL == "" {
		baseURL = "https://api.openai.com"
	}
	if strings.HasSuffix(baseURL, "/v1") {
		return baseURL + endpoint
	}
	return baseURL + "/v1" + endpoint
}

func (p *OpenAIProvider) setHeaders(req *http.Request) {
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+p.Config().APIKey)
}

// openaiStreamReader implements streaming for OpenAI
type openaiStreamReader struct {
	resp     *http.Response
	scanner  *bufio.Scanner
	provider string
	done     bool
}

func (r *openaiStreamReader) Next() (*StreamChunk, error) {
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
				streamChunk.Type = "content"
				streamChunk.Content = choice.Delta.Content
			}

			// Handle reasoning streaming (OpenAI format only)
			if choice.Delta.ReasoningSummary != nil && choice.Delta.ReasoningSummary.Text != "" {
				streamChunk.Type = "reasoning"
				streamChunk.Reasoning = &ReasoningChunk{
					Summary: choice.Delta.ReasoningSummary.Text,
				}
			}

			// Handle tool call deltas
			if len(choice.Delta.ToolCalls) > 0 {
				for _, toolCallDelta := range choice.Delta.ToolCalls {
					streamChunk.Type = "tool_call_delta"
					streamChunk.ToolCallDelta = &ToolCallDelta{
						Index: toolCallDelta.Index,
						ID:    toolCallDelta.ID,
						Type:  toolCallDelta.Type,
					}

					if toolCallDelta.Function != nil {
						streamChunk.ToolCallDelta.FunctionName = toolCallDelta.Function.Name
						streamChunk.ToolCallDelta.ArgumentsDelta = toolCallDelta.Function.Arguments
					}

					// Return immediately for each tool call delta
					return streamChunk, nil
				}
			}

			if choice.FinishReason != "" {
				streamChunk.FinishReason = choice.FinishReason
			}

			// Only return if we have content, reasoning, or finish reason
			if streamChunk.Type != "" || streamChunk.FinishReason != "" {
				return streamChunk, nil
			}
		}
	}

	if err := r.scanner.Err(); err != nil {
		return nil, err
	}

	r.done = true
	return &StreamChunk{Done: true, Provider: r.provider}, nil
}

func (r *openaiStreamReader) Close() error {
	return r.resp.Body.Close()
}

// OpenAI API structures
// OpenAI API request/response structures
type openaiRequest struct {
	Model               string                `json:"model"`
	Messages            []openaiMessage       `json:"messages"`
	MaxTokens           *int                  `json:"max_tokens,omitempty"`
	MaxCompletionTokens *int                  `json:"max_completion_tokens,omitempty"`
	Temperature         *float64              `json:"temperature,omitempty"`
	Stream              bool                  `json:"stream,omitempty"`
	Tools               []openaiTool          `json:"tools,omitempty"`
	ToolChoice          any                   `json:"tool_choice,omitempty"`
	ResponseFormat      *openaiResponseFormat `json:"response_format,omitempty"`
	Reasoning           *openaiReasoning      `json:"reasoning,omitempty"`
}

type openaiReasoning struct {
	Effort  string `json:"effort,omitempty"`  // OpenAI format
	Summary string `json:"summary,omitempty"` // OpenAI format
}

type openaiResponseFormat struct {
	Type       string            `json:"type"` // "text", "json_object", "json_schema"
	JSONSchema *openaiJSONSchema `json:"json_schema,omitempty"`
}

type openaiJSONSchema struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Schema      any    `json:"schema"`
	Strict      *bool  `json:"strict,omitempty"`
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

// openaiToolCallDelta represents incremental tool call data from OpenAI
type openaiToolCallDelta struct {
	Index    int                      `json:"index"`
	ID       string                   `json:"id,omitempty"`
	Type     string                   `json:"type,omitempty"`
	Function *openaiToolCallFuncDelta `json:"function,omitempty"`
}

type openaiToolCallFuncDelta struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
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
	ToolCalls        []openaiToolCallDelta   `json:"tool_calls,omitempty"`
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
	Input     any                    `json:"input"`
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

// convertResponseFormat converts response format and ensures OpenAI compatibility
func (p *OpenAIProvider) convertResponseFormat(rf *ResponseFormat) *openaiResponseFormat {
	result := &openaiResponseFormat{Type: rf.Type}

	if rf.JSONSchema != nil {
		result.JSONSchema = &openaiJSONSchema{
			Name:        rf.JSONSchema.Name,
			Description: rf.JSONSchema.Description,
			Schema:      p.ensureStrictSchema(rf.JSONSchema.Schema),
			Strict:      rf.JSONSchema.Strict,
		}
	}

	return result
}

// ensureStrictSchema recursively adds additionalProperties: false to all objects for OpenAI strict mode
func (p *OpenAIProvider) ensureStrictSchema(schema interface{}) interface{} {
	switch s := schema.(type) {
	case map[string]interface{}:
		result := make(map[string]interface{}, len(s))
		for k, v := range s {
			result[k] = p.ensureStrictSchema(v)
		}
		if result["type"] == "object" {
			result["additionalProperties"] = false
		}
		return result
	case []interface{}:
		for i, v := range s {
			s[i] = p.ensureStrictSchema(v)
		}
		return s
	default:
		return schema
	}
}
