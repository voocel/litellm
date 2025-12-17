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
	RegisterBuiltin("openai", func(cfg ProviderConfig) Provider {
		return NewOpenAI(cfg)
	}, "https://api.openai.com")
}

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
		// GPT-5.1 family (flagship)
		{
			ID: "gpt-5.1", Provider: "openai", Name: "GPT-5.1", ContextWindow: 400000, MaxOutputTokens: 128000,
			Capabilities: []string{"chat", "function_call", "vision", "code", "reasoning"},
		},
		{
			ID: "gpt-5.1-mini", Provider: "openai", Name: "GPT-5.1 Mini", ContextWindow: 400000, MaxOutputTokens: 64000,
			Capabilities: []string{"chat", "function_call", "vision", "code", "reasoning"},
		},

		// GPT-5 family (prior generation)
		{
			ID: "gpt-5", Provider: "openai", Name: "GPT-5", ContextWindow: 400000, MaxOutputTokens: 128000,
			Capabilities: []string{"chat", "function_call", "vision", "code", "reasoning"},
		},
		{
			ID: "gpt-5-mini", Provider: "openai", Name: "GPT-5 Mini", ContextWindow: 400000, MaxOutputTokens: 64000,
			Capabilities: []string{"chat", "function_call", "vision", "code", "reasoning"},
		},
		{
			ID: "gpt-5-nano", Provider: "openai", Name: "GPT-5 Nano", ContextWindow: 400000, MaxOutputTokens: 32000,
			Capabilities: []string{"chat", "function_call", "vision", "code"},
		},

		// GPT-4.1 family (~1M context, ~32K max output)
		{
			ID: "gpt-4.1", Provider: "openai", Name: "GPT-4.1", ContextWindow: 1000000, MaxOutputTokens: 32768,
			Capabilities: []string{"chat", "function_call", "vision", "code"},
		},
		{
			ID: "gpt-4.1-mini", Provider: "openai", Name: "GPT-4.1 Mini", ContextWindow: 1000000, MaxOutputTokens: 32768,
			Capabilities: []string{"chat", "function_call", "vision", "code"},
		},
		{
			ID: "gpt-4.1-nano", Provider: "openai", Name: "GPT-4.1 Nano", ContextWindow: 1000000, MaxOutputTokens: 32768,
			Capabilities: []string{"chat", "function_call", "code"},
		},

		// GPT-4o family (128K context, 16K max output)
		{
			ID: "gpt-4o", Provider: "openai", Name: "GPT-4o", ContextWindow: 128000, MaxOutputTokens: 16384,
			Capabilities: []string{"chat", "function_call", "vision"},
		},
		{
			ID: "gpt-4o-mini", Provider: "openai", Name: "GPT-4o Mini", ContextWindow: 128000, MaxOutputTokens: 16384,
			Capabilities: []string{"chat", "function_call", "vision"},
		},

		// o-series reasoning（200K context, 100K max output）
		{
			ID: "o3-pro", Provider: "openai", Name: "OpenAI o3 Pro", ContextWindow: 200000, MaxOutputTokens: 100000,
			Capabilities: []string{"chat", "reasoning"},
		},
		{
			ID: "o3", Provider: "openai", Name: "OpenAI o3", ContextWindow: 200000, MaxOutputTokens: 100000,
			Capabilities: []string{"chat", "reasoning"},
		},
		{
			ID: "o3-mini", Provider: "openai", Name: "OpenAI o3 Mini", ContextWindow: 200000, MaxOutputTokens: 100000,
			Capabilities: []string{"chat", "reasoning"},
		},
	}
}

// needsMaxCompletionTokens checks if the model requires max_completion_tokens instead of max_tokens
func (p *OpenAIProvider) needsMaxCompletionTokens(model string) bool {
	modelLower := strings.ToLower(model)

	// o-series reasoning models (o1, o3, o4)
	if strings.HasPrefix(modelLower, "o1") ||
		strings.HasPrefix(modelLower, "o3") ||
		strings.HasPrefix(modelLower, "o4") {
		return true
	}

	// GPT-5 series
	if strings.HasPrefix(modelLower, "gpt-5") {
		return true
	}

	return false
}

// isReasoningModel checks if the model supports advanced reasoning (o-series and GPT-5)
func (p *OpenAIProvider) isReasoningModel(model string) bool {
	return p.needsMaxCompletionTokens(model)
}

// resolveTokenParams resolves token parameter conflicts based on model type
func (p *OpenAIProvider) resolveTokenParams(req *Request) (maxTokens *int, maxCompletionTokens *int) {
	if req.MaxTokens == nil {
		return nil, nil
	}

	if p.needsMaxCompletionTokens(req.Model) {
		// Reasoning models use max_completion_tokens
		return nil, req.MaxTokens
	}

	// Traditional models use max_tokens
	return req.MaxTokens, nil
}

// buildResponsesAPIRequest maps internal Request into the official Responses API payload
func (p *OpenAIProvider) buildResponsesAPIRequest(req *Request, modelName string) responsesAPIRequest {
	targetModel := modelName
	if rp := req.ResponsesParams; rp != nil && rp.ModelOverride != "" {
		targetModel = rp.ModelOverride
	}

	responsesReq := responsesAPIRequest{
		Model: targetModel,
	}

	if inputItems := convertMessagesToResponsesInput(req.Messages); len(inputItems) > 0 {
		responsesReq.Input = inputItems
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

	// Responses API still supports function/file_search and other tools
	if len(req.Tools) > 0 {
		responsesReq.Tools = make([]openaiTool, len(req.Tools))
		for i, tool := range req.Tools {
			var function *openaiToolFunction
			if tool.Type == "function" {
				function = &openaiToolFunction{
					Name:        tool.Function.Name,
					Description: tool.Function.Description,
					Parameters:  tool.Function.Parameters,
				}
			}
			responsesReq.Tools[i] = openaiTool{
				Type:     tool.Type,
				Function: function,
			}
		}
	}

	if rp := req.ResponsesParams; rp != nil && rp.ToolChoice != nil {
		responsesReq.ToolChoice = rp.ToolChoice
	} else if req.ToolChoice != nil {
		responsesReq.ToolChoice = req.ToolChoice
	}

	switch {
	case req.ResponsesParams != nil && req.ResponsesParams.ResponseFormat != nil:
		responsesReq.ResponseFormat = p.convertResponseFormat(req.ResponsesParams.ResponseFormat)
	case req.ResponseFormat != nil:
		responsesReq.ResponseFormat = p.convertResponseFormat(req.ResponseFormat)
	}

	if rp := req.ResponsesParams; rp != nil && rp.Temperature != nil {
		responsesReq.Temperature = rp.Temperature
	} else if req.Temperature != nil {
		responsesReq.Temperature = req.Temperature
	}

	if rp := req.ResponsesParams; rp != nil && rp.TopP != nil {
		responsesReq.TopP = rp.TopP
	} else if req.TopP != nil {
		responsesReq.TopP = req.TopP
	}

	// Token limits: prefer Responses-specific, otherwise fall back to generic MaxTokens
	if rp := req.ResponsesParams; rp != nil && rp.MaxOutputTokens != nil {
		responsesReq.MaxOutputTokens = rp.MaxOutputTokens
	} else if req.MaxTokens != nil {
		responsesReq.MaxOutputTokens = req.MaxTokens
	}
	if rp := req.ResponsesParams; rp != nil && rp.MaxInputTokens != nil {
		responsesReq.MaxInputTokens = rp.MaxInputTokens
	}

	if rp := req.ResponsesParams; rp != nil && rp.MaxToolCalls != nil {
		responsesReq.MaxToolCalls = rp.MaxToolCalls
	}
	if rp := req.ResponsesParams; rp != nil && rp.ParallelToolCalls != nil {
		responsesReq.ParallelToolCalls = rp.ParallelToolCalls
	} else if req.ParallelToolCalls != nil {
		responsesReq.ParallelToolCalls = req.ParallelToolCalls
	}

	if rp := req.ResponsesParams; rp != nil {
		if rp.Instructions != "" {
			responsesReq.Instructions = rp.Instructions
		}
		if rp.Conversation != "" {
			responsesReq.Conversation = rp.Conversation
		}
		if rp.PreviousResponseID != "" {
			responsesReq.PreviousResponseID = rp.PreviousResponseID
		}
		if len(rp.Metadata) > 0 {
			responsesReq.Metadata = rp.Metadata
		}
		if rp.Store != nil {
			responsesReq.Store = rp.Store
		}
		if len(rp.Include) > 0 {
			responsesReq.Include = rp.Include
		}
		if rp.SafetyIdentifier != "" {
			responsesReq.SafetyIdentifier = rp.SafetyIdentifier
		}
		if rp.ServiceTier != "" {
			responsesReq.ServiceTier = rp.ServiceTier
		}
		if rp.PromptCacheKey != "" {
			responsesReq.PromptCacheKey = rp.PromptCacheKey
		}
		if rp.PromptCacheRetention != "" {
			responsesReq.PromptCacheRetention = rp.PromptCacheRetention
		}
		if rp.Background != nil {
			responsesReq.Background = rp.Background
		}
		if len(rp.Prompt) > 0 {
			responsesReq.Prompt = rp.Prompt
		}
	}

	if responsesReq.Store == nil && req.Store != nil {
		responsesReq.Store = req.Store
	}
	if responsesReq.SafetyIdentifier == "" && req.SafetyIdentifier != "" {
		responsesReq.SafetyIdentifier = req.SafetyIdentifier
	}
	if responsesReq.ServiceTier == "" && req.ServiceTier != "" {
		responsesReq.ServiceTier = req.ServiceTier
	}

	return responsesReq
}

func (p *OpenAIProvider) Chat(ctx context.Context, req *Request) (*Response, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}

	// Validate request parameters using base provider validation
	// Note: Temperature validation is skipped for reasoning models as they don't support it
	if !p.isReasoningModel(req.Model) {
		if err := p.BaseProvider.ValidateRequest(req); err != nil {
			return nil, err
		}
	} else {
		// For reasoning models, only validate model and messages
		if req.Model == "" {
			return nil, fmt.Errorf("openai: model is required")
		}
		if len(req.Messages) == 0 {
			return nil, fmt.Errorf("openai: at least one message is required")
		}
		if req.MaxTokens != nil && *req.MaxTokens <= 0 {
			return nil, fmt.Errorf("openai: max_tokens must be positive, got %d", *req.MaxTokens)
		}
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

	openaiReq.TopP = req.TopP
	openaiReq.TopLogProbs = req.TopLogProbs
	if req.ServiceTier != "" {
		openaiReq.ServiceTier = req.ServiceTier
	}
	if req.Store != nil {
		openaiReq.Store = req.Store
	}
	if req.ParallelToolCalls != nil {
		openaiReq.ParallelToolCalls = req.ParallelToolCalls
	}
	if req.SafetyIdentifier != "" {
		openaiReq.SafetyIdentifier = req.SafetyIdentifier
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

	// Resolve token parameters based on model type
	maxTokens, maxCompletionTokens := p.resolveTokenParams(req)
	openaiReq.MaxTokens = maxTokens
	openaiReq.MaxCompletionTokens = maxCompletionTokens

	// Set temperature only for non-reasoning models
	if !p.isReasoningModel(modelName) {
		openaiReq.Temperature = req.Temperature
	}

	// Set stop sequences (up to 4 sequences)
	if len(req.Stop) > 0 {
		openaiReq.Stop = req.Stop
	}

	// Convert tool definitions
	if len(req.Tools) > 0 {
		openaiReq.Tools = ConvertTools(req.Tools)
	}

	openaiReq.Messages = ConvertMessagesToOpenAI(req.Messages)

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
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body))
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
		response.Contents = convertOpenAIContentToMessageContents(choice.Message.Content)
		response.Content = joinMessageContentsText(response.Contents)
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
	responsesReq := p.buildResponsesAPIRequest(req, modelName)

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
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body))
	}

	var responsesResp responsesAPIResponse
	if err := json.NewDecoder(resp.Body).Decode(&responsesResp); err != nil {
		return nil, fmt.Errorf("openai: decode responses response: %w", err)
	}

	responseContent, responseContents, toolCalls, reasoningData := p.extractResponsesOutput(&responsesResp)

	response := &Response{
		Content:   responseContent,
		Contents:  responseContents,
		ToolCalls: toolCalls,
		Usage: Usage{
			PromptTokens:     responsesResp.Usage.InputTokens,
			CompletionTokens: responsesResp.Usage.OutputTokens,
			TotalTokens:      responsesResp.Usage.TotalTokens,
		},
		Model:        responsesResp.Model,
		Provider:     "openai",
		Reasoning:    reasoningData,
		FinishReason: responsesResp.Status,
	}

	if responsesResp.Usage.OutputTokensDetails != nil {
		response.Usage.ReasoningTokens = responsesResp.Usage.OutputTokensDetails.ReasoningTokens
	}

	return response, nil
}

// extractResponsesOutput converts Responses API output into the common response format
func (p *OpenAIProvider) extractResponsesOutput(responsesResp *responsesAPIResponse) (string, []MessageContent, []ToolCall, *ReasoningData) {
	var (
		toolCalls       []ToolCall
		reasoning       *ReasoningData
		messageContents []MessageContent
	)

	for _, outputItem := range responsesResp.Output {
		switch outputItem.Type {
		case "message":
			messageContents = append(messageContents, convertResponsesContentToMessageContents(outputItem.Content)...)
		case "tool_call":
			if outputItem.Name == "" && outputItem.Arguments == "" {
				continue
			}
			callID := outputItem.CallID
			if callID == "" {
				callID = outputItem.ID
			}
			toolCalls = append(toolCalls, ToolCall{
				ID:   callID,
				Type: "function",
				Function: FunctionCall{
					Name:      outputItem.Name,
					Arguments: outputItem.Arguments,
				},
			})
		case "reasoning":
			if len(outputItem.Summary) == 0 {
				continue
			}
			var summaryBuilder strings.Builder
			for _, summary := range outputItem.Summary {
				if summary.Text == "" {
					continue
				}
				if summaryBuilder.Len() > 0 {
					summaryBuilder.WriteString("\n")
				}
				summaryBuilder.WriteString(summary.Text)
			}
			if summaryBuilder.Len() > 0 {
				reasoning = &ReasoningData{
					Summary: summaryBuilder.String(),
					Content: summaryBuilder.String(),
				}
			}
		}
	}

	content := responsesResp.OutputText
	if text := joinMessageContentsText(messageContents); text != "" {
		content = text
	}

	if reasoning != nil && responsesResp.Usage.OutputTokensDetails != nil {
		reasoning.TokensUsed = responsesResp.Usage.OutputTokensDetails.ReasoningTokens
	}

	return content, messageContents, toolCalls, reasoning
}

func convertResponsesUsage(apiUsage responsesAPIUsage) *Usage {
	if apiUsage.InputTokens == 0 && apiUsage.OutputTokens == 0 && apiUsage.TotalTokens == 0 {
		return nil
	}

	usage := &Usage{
		PromptTokens:     apiUsage.InputTokens,
		CompletionTokens: apiUsage.OutputTokens,
		TotalTokens:      apiUsage.TotalTokens,
	}

	if apiUsage.OutputTokensDetails != nil {
		usage.ReasoningTokens = apiUsage.OutputTokensDetails.ReasoningTokens
	}

	return usage
}

func (p *OpenAIProvider) Stream(ctx context.Context, req *Request) (StreamReader, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}

	modelName := req.Model

	// Check if should use Responses API (same logic as Chat)
	shouldUseResponsesAPI := req.UseResponsesAPI ||
		(p.isReasoningModel(modelName) && (req.ReasoningEffort != "" || req.ReasoningSummary != ""))

	if shouldUseResponsesAPI {
		return p.streamWithResponsesAPI(ctx, req, modelName)
	}

	openaiReq := openaiRequest{
		Model:    modelName,
		Messages: make([]openaiMessage, len(req.Messages)),
		Stream:   true,
		// Enable usage statistics in streaming responses
		StreamOptions: &openaiStreamOptions{
			IncludeUsage: true,
		},
	}

	// Convert response format
	if req.ResponseFormat != nil {
		openaiReq.ResponseFormat = p.convertResponseFormat(req.ResponseFormat)
	}

	// Set correct parameters based on model type
	if p.isReasoningModel(modelName) {
		// Reasoning models always use max_completion_tokens
		openaiReq.MaxCompletionTokens = req.MaxTokens

		// Set reasoning_effort for Chat Completions API (top-level parameter)
		// Note: reasoning_effort is supported in streaming mode for Chat Completions API
		// but reasoning summary is only available in Responses API
		if req.ReasoningEffort != "" {
			openaiReq.ReasoningEffort = req.ReasoningEffort
		}
	} else {
		// Traditional models use max_tokens and temperature
		openaiReq.MaxTokens = req.MaxTokens
		openaiReq.Temperature = req.Temperature
	}

	// Set stop sequences (up to 4 sequences)
	if len(req.Stop) > 0 {
		openaiReq.Stop = req.Stop
	}

	// Convert tool definitions
	if len(req.Tools) > 0 {
		openaiReq.Tools = ConvertTools(req.Tools)
	}

	// Set tool choice
	if req.ToolChoice != nil {
		openaiReq.ToolChoice = req.ToolChoice
	}

	// Convert messages
	openaiReq.Messages = ConvertMessagesToOpenAI(req.Messages)

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
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body))
	}

	scanner := bufio.NewScanner(resp.Body)
	// Increase buffer size to handle large tokens (default 64KB, max 1MB)
	buf := make([]byte, 0, 64*1024)
	scanner.Buffer(buf, 1024*1024)

	return &openaiStreamReader{
		resp:     resp,
		scanner:  scanner,
		provider: "openai",
	}, nil
}

// streamWithResponsesAPI handles streaming for the /responses endpoint
func (p *OpenAIProvider) streamWithResponsesAPI(ctx context.Context, req *Request, modelName string) (StreamReader, error) {
	responsesReq := p.buildResponsesAPIRequest(req, modelName)

	body, err := json.Marshal(responsesReq)
	if err != nil {
		return nil, fmt.Errorf("openai: marshal responses request: %w", err)
	}

	// Use Responses API with stream=true query param (or header, but usually query param for OpenAI)
	// Note: The /responses endpoint might use a different mechanism, but assuming standard OpenAI streaming
	// However, /responses uses SSE with specific event types.
	url := p.buildURL("/responses") + "?stream=true"
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("openai: create responses request: %w", err)
	}

	p.setHeaders(httpReq)
	// Ensure Accept header is set for SSE
	httpReq.Header.Set("Accept", "text/event-stream")

	resp, err := p.HTTPClient().Do(httpReq)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body))
	}

	scanner := bufio.NewScanner(resp.Body)
	buf := make([]byte, 0, 64*1024)
	scanner.Buffer(buf, 1024*1024)

	return &responsesAPIStreamReader{
		resp:     resp,
		scanner:  scanner,
		provider: "openai",
		model:    modelName,
	}, nil
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
			return nil, fmt.Errorf("openai: failed to parse stream chunk: %w", err)
		}

		// Handle usage chunk (comes with empty choices array)
		if chunk.Usage != nil && len(chunk.Choices) == 0 {
			streamChunk := &StreamChunk{
				Provider: r.provider,
				Model:    chunk.Model,
				Done:     true,
				Usage: &Usage{
					PromptTokens:     chunk.Usage.PromptTokens,
					CompletionTokens: chunk.Usage.CompletionTokens,
					TotalTokens:      chunk.Usage.TotalTokens,
				},
			}
			if chunk.Usage.CompletionTokensDetails != nil {
				streamChunk.Usage.ReasoningTokens = chunk.Usage.CompletionTokensDetails.ReasoningTokens
			}
			r.done = true
			return streamChunk, nil
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

			// Check if stream is finished (usage chunk will come separately after this)
			if choice.FinishReason != "" {
				streamChunk.FinishReason = choice.FinishReason
				// Don't mark as done yet, wait for usage chunk
				return streamChunk, nil
			}

			// Only return if we have content or reasoning
			if streamChunk.Type != "" {
				return streamChunk, nil
			}
		}
	}

	if err := r.scanner.Err(); err != nil {
		return nil, fmt.Errorf("openai: stream read error: %w", err)
	}

	r.done = true
	return &StreamChunk{Done: true, Provider: r.provider}, nil
}

func (r *openaiStreamReader) Close() error {
	return r.resp.Body.Close()
}

// responsesAPIStreamReader implements streaming for OpenAI /responses endpoint
type responsesAPIStreamReader struct {
	resp     *http.Response
	scanner  *bufio.Scanner
	provider string
	model    string
	done     bool
}

func (r *responsesAPIStreamReader) Next() (*StreamChunk, error) {
	if r.done {
		return &StreamChunk{Done: true, Provider: r.provider, Model: r.model}, nil
	}

	// /responses API uses SSE with specific event types
	// event: response.output_text.delta
	// data: {"delta": "..."}
	//
	// event: response.function_call_arguments.delta
	// data: {"call_id": "...", "delta": "..."}
	//
	// event: response.output_text.done
	// data: {}

	var currentEvent string

	for r.scanner.Scan() {
		line := r.scanner.Text()

		if strings.HasPrefix(line, "event: ") {
			currentEvent = strings.TrimPrefix(line, "event: ")
			continue
		}

		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimPrefix(line, "data: ")

		streamChunk := &StreamChunk{
			Provider: r.provider,
			Model:    r.model,
		}

		switch currentEvent {
		case "response.output_text.delta":
			var delta struct {
				Delta        string `json:"delta"`
				ItemID       string `json:"item_id"`
				OutputIndex  *int   `json:"output_index,omitempty"`
				ContentIndex *int   `json:"content_index,omitempty"`
			}
			if err := json.Unmarshal([]byte(data), &delta); err != nil {
				return nil, fmt.Errorf("openai: parse output delta: %w", err)
			}
			streamChunk.Type = "content"
			streamChunk.Content = delta.Delta
			streamChunk.ItemID = delta.ItemID
			streamChunk.OutputIndex = delta.OutputIndex
			streamChunk.ContentIndex = delta.ContentIndex
			return streamChunk, nil

		case "response.function_call_arguments.delta":
			var delta struct {
				CallID      string `json:"call_id"`
				Delta       string `json:"delta"`
				Index       *int   `json:"index,omitempty"`
				ItemID      string `json:"item_id"`
				OutputIndex *int   `json:"output_index,omitempty"`
			}
			if err := json.Unmarshal([]byte(data), &delta); err != nil {
				return nil, fmt.Errorf("openai: parse tool delta: %w", err)
			}
			streamChunk.Type = "tool_call_delta"
			streamChunk.ToolCallDelta = &ToolCallDelta{
				ID:             delta.CallID,
				Type:           "function",
				ArgumentsDelta: delta.Delta,
				OutputIndex:    delta.OutputIndex,
				ItemID:         delta.ItemID,
			}
			if delta.Index != nil {
				streamChunk.ToolCallDelta.Index = *delta.Index
			}
			streamChunk.ItemID = delta.ItemID
			streamChunk.OutputIndex = delta.OutputIndex
			// Note: /responses might not send index, so we might need to manage it if multiple tools are called
			// For now assuming single tool stream or ID matching is sufficient
			return streamChunk, nil

		case "response.completed":
			var completed responsesAPICompletedEvent
			if err := json.Unmarshal([]byte(data), &completed); err != nil {
				return nil, fmt.Errorf("openai: parse responses completed: %w", err)
			}
			if completed.Response.Model != "" {
				r.model = completed.Response.Model
				streamChunk.Model = completed.Response.Model
			}
			streamChunk.Done = true
			if usage := convertResponsesUsage(completed.Response.Usage); usage != nil {
				streamChunk.Usage = usage
			}
			r.done = true
			return streamChunk, nil

		case "response.error":
			var responseErr responsesAPIErrorEvent
			if err := json.Unmarshal([]byte(data), &responseErr); err != nil {
				return nil, fmt.Errorf("openai: parse responses error: %w", err)
			}
			return nil, fmt.Errorf("openai: responses stream error: %s", responseErr.Error.Message)

		case "response.output_text.done":
			// Text finished; wait for response.completed to carry usage
			continue
		}
	}

	if err := r.scanner.Err(); err != nil {
		return nil, fmt.Errorf("openai: stream read error: %w", err)
	}

	r.done = true
	return &StreamChunk{Done: true, Provider: r.provider, Model: r.model}, nil
}

func (r *responsesAPIStreamReader) Close() error {
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
	TopP                *float64              `json:"top_p,omitempty"`
	TopLogProbs         *int                  `json:"top_logprobs,omitempty"`
	Stream              bool                  `json:"stream,omitempty"`
	StreamOptions       *openaiStreamOptions  `json:"stream_options,omitempty"`
	Stop                []string              `json:"stop,omitempty"` // Up to 4 sequences where the API will stop generating
	Tools               []openaiTool          `json:"tools,omitempty"`
	ToolChoice          any                   `json:"tool_choice,omitempty"`
	ResponseFormat      *openaiResponseFormat `json:"response_format,omitempty"`
	ServiceTier         string                `json:"service_tier,omitempty"`
	Store               *bool                 `json:"store,omitempty"`
	ParallelToolCalls   *bool                 `json:"parallel_tool_calls,omitempty"`
	SafetyIdentifier    string                `json:"safety_identifier,omitempty"`

	// Chat Completions API: top-level reasoning_effort parameter
	ReasoningEffort string `json:"reasoning_effort,omitempty"` // minimal, low, medium, high

	// Responses API: nested reasoning object (used in responsesAPIRequest instead)
	Reasoning *openaiReasoning `json:"reasoning,omitempty"`
}

type openaiStreamOptions struct {
	IncludeUsage bool `json:"include_usage"`
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
	Content    interface{}      `json:"content,omitempty"`
	ToolCalls  []openaiToolCall `json:"tool_calls,omitempty"`
	ToolCallID string           `json:"tool_call_id,omitempty"`
}

type openaiContentPart struct {
	Type     string          `json:"type"`
	Text     string          `json:"text,omitempty"`
	ImageURL *openaiImageURL `json:"image_url,omitempty"`
}

type openaiImageURL struct {
	URL string `json:"url"`
}

type openaiTool struct {
	Type     string              `json:"type"`
	Function *openaiToolFunction `json:"function,omitempty"`
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
	Usage   *openaiUsage   `json:"usage,omitempty"`
}

// Responses API structures
type responsesAPIRequest struct {
	Model                string                  `json:"model"`
	Input                []responsesAPIInputItem `json:"input,omitempty"`
	Instructions         string                  `json:"instructions,omitempty"`
	Conversation         string                  `json:"conversation,omitempty"`
	PreviousResponseID   string                  `json:"previous_response_id,omitempty"`
	Metadata             map[string]string       `json:"metadata,omitempty"`
	Reasoning            *responsesAPIReasoning  `json:"reasoning,omitempty"`
	Tools                []openaiTool            `json:"tools,omitempty"`
	ToolChoice           any                     `json:"tool_choice,omitempty"`
	MaxOutputTokens      *int                    `json:"max_output_tokens,omitempty"`
	MaxInputTokens       *int                    `json:"max_input_tokens,omitempty"`
	MaxToolCalls         *int                    `json:"max_tool_calls,omitempty"`
	ParallelToolCalls    *bool                   `json:"parallel_tool_calls,omitempty"`
	Store                *bool                   `json:"store,omitempty"`
	Temperature          *float64                `json:"temperature,omitempty"`
	TopP                 *float64                `json:"top_p,omitempty"`
	ResponseFormat       *openaiResponseFormat   `json:"response_format,omitempty"`
	SafetyIdentifier     string                  `json:"safety_identifier,omitempty"`
	ServiceTier          string                  `json:"service_tier,omitempty"`
	Include              []string                `json:"include,omitempty"`
	PromptCacheKey       string                  `json:"prompt_cache_key,omitempty"`
	PromptCacheRetention string                  `json:"prompt_cache_retention,omitempty"`
	Background           *bool                   `json:"background,omitempty"`
	Prompt               map[string]interface{}  `json:"prompt,omitempty"`
}

type responsesAPIReasoning struct {
	Effort  string `json:"effort,omitempty"`
	Summary string `json:"summary,omitempty"`
}

type responsesAPIInputItem struct {
	Type    string                    `json:"type"`
	Role    string                    `json:"role,omitempty"`
	Content []responsesAPIContentItem `json:"content,omitempty"`
	ID      string                    `json:"id,omitempty"`
}

type responsesAPIResponse struct {
	ID         string                   `json:"id"`
	Object     string                   `json:"object"`
	Model      string                   `json:"model"`
	Status     string                   `json:"status,omitempty"`
	OutputText string                   `json:"output_text"`
	Output     []responsesAPIOutputItem `json:"output"`
	Usage      responsesAPIUsage        `json:"usage"`
}

type responsesAPIOutputItem struct {
	ID        string                    `json:"id"`
	Type      string                    `json:"type"`
	Role      string                    `json:"role,omitempty"`
	Status    string                    `json:"status,omitempty"`
	CallID    string                    `json:"call_id,omitempty"`
	Name      string                    `json:"name,omitempty"`
	Arguments string                    `json:"arguments,omitempty"`
	Content   []responsesAPIContentItem `json:"content,omitempty"`
	Summary   []responsesAPISummaryItem `json:"summary,omitempty"`
	Output    []responsesAPIToolOutput  `json:"output,omitempty"`
}

type responsesAPIContentItem struct {
	Type        string                   `json:"type"`
	Text        string                   `json:"text,omitempty"`
	Annotations []map[string]interface{} `json:"annotations,omitempty"`
	Logprobs    []map[string]interface{} `json:"logprobs,omitempty"`
	ImageURL    *responsesAPIImageURL    `json:"image_url,omitempty"`
}

type responsesAPIImageURL struct {
	URL string `json:"url"`
}

type responsesAPISummaryItem struct {
	Text string `json:"text"`
}

type responsesAPIToolOutput struct {
	Type    string `json:"type"`
	Content string `json:"content,omitempty"`
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

type responsesAPICompletedEvent struct {
	Response struct {
		Model string            `json:"model"`
		Usage responsesAPIUsage `json:"usage"`
	} `json:"response"`
}

type responsesAPIErrorEvent struct {
	Error struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
}

// convertResponseFormat converts response format and ensures OpenAI compatibility
func (p *OpenAIProvider) convertResponseFormat(rf *ResponseFormat) *openaiResponseFormat {
	if rf == nil {
		return nil
	}

	result := &openaiResponseFormat{Type: rf.Type}

	if rf.JSONSchema != nil {
		schema := rf.JSONSchema.Schema

		// Clean schema for OpenAI compatibility
		schema = p.cleanSchemaForOpenAI(schema)

		// Apply strict mode if requested
		if rf.JSONSchema.Strict != nil && *rf.JSONSchema.Strict {
			schema = p.ensureStrictSchema(schema)
		}

		result.JSONSchema = &openaiJSONSchema{
			Name:        rf.JSONSchema.Name,
			Description: rf.JSONSchema.Description,
			Schema:      schema,
			Strict:      rf.JSONSchema.Strict,
		}
	}

	return result
}

// cleanSchemaForOpenAI removes unsupported validation rules and ensures OpenAI compatibility
func (p *OpenAIProvider) cleanSchemaForOpenAI(schema interface{}) interface{} {
	switch s := schema.(type) {
	case map[string]interface{}:
		result := make(map[string]interface{}, len(s))
		for k, v := range s {
			// Skip unsupported OpenAI schema properties
			if k == "examples" || k == "default" || k == "const" {
				continue
			}
			result[k] = p.cleanSchemaForOpenAI(v)
		}

		// Add additionalProperties: false for objects in strict mode
		if result["type"] == "object" {
			if _, hasAdditionalProps := result["additionalProperties"]; !hasAdditionalProps {
				result["additionalProperties"] = false
			}
		}

		return result
	case []interface{}:
		cleaned := make([]interface{}, len(s))
		for i, v := range s {
			cleaned[i] = p.cleanSchemaForOpenAI(v)
		}
		return cleaned
	default:
		return schema
	}
}

// ensureStrictSchema recursively adds additionalProperties: false to all objects for OpenAI strict mode
func (p *OpenAIProvider) ensureStrictSchema(schema interface{}) interface{} {
	return p.cleanSchemaForOpenAI(schema)
}

// ==================== Common conversion helpers (used by OpenAI-compatible providers) ====================

// OpenAICompatMessage is a generic OpenAI-compatible message format
// Used by DeepSeek/GLM/Qwen etc.; Content stays interface{} to allow flexible formats
type OpenAICompatMessage struct {
	Role             string           `json:"role"`
	Content          interface{}      `json:"content,omitempty"` // string or array (needed by OpenRouter)
	ToolCalls        []openaiToolCall `json:"tool_calls,omitempty"`
	ToolCallID       string           `json:"tool_call_id,omitempty"`
	ReasoningContent string           `json:"reasoning_content,omitempty"` // used by GLM/Qwen
}

// ConvertMessagesToOpenAI converts generic Message into native OpenAI format (string content or parts)
// Used internally by OpenAI provider
func ConvertMessagesToOpenAI(messages []Message) []openaiMessage {
	result := make([]openaiMessage, len(messages))
	for i, msg := range messages {
		openaiMsg := openaiMessage{
			Role:       msg.Role,
			ToolCallID: msg.ToolCallID,
		}

		if parts := buildOpenAIContentParts(msg); len(parts) > 0 {
			openaiMsg.Content = parts
		} else if msg.Content != "" {
			openaiMsg.Content = msg.Content
		}

		// Convert tool calls
		if len(msg.ToolCalls) > 0 {
			openaiMsg.ToolCalls = make([]openaiToolCall, len(msg.ToolCalls))
			for j, tc := range msg.ToolCalls {
				openaiMsg.ToolCalls[j] = openaiToolCall{
					ID:   tc.ID,
					Type: tc.Type,
					Function: openaiToolCallFunc{
						Name:      tc.Function.Name,
						Arguments: tc.Function.Arguments,
					},
				}
			}
		}

		result[i] = openaiMsg
	}
	return result
}

// ConvertMessages converts generic Message into OpenAI-compatible format (Content as interface{})
// For DeepSeek/GLM/Qwen/OpenRouter etc.
func ConvertMessages(messages []Message) []OpenAICompatMessage {
	result := make([]OpenAICompatMessage, len(messages))
	for i, msg := range messages {
		compatMsg := OpenAICompatMessage{
			Role:       msg.Role,
			ToolCallID: msg.ToolCallID,
		}

		if parts := buildOpenAIContentParts(msg); len(parts) > 0 {
			compatMsg.Content = parts
		} else {
			compatMsg.Content = msg.Content
		}

		// Convert tool calls
		if len(msg.ToolCalls) > 0 {
			compatMsg.ToolCalls = make([]openaiToolCall, len(msg.ToolCalls))
			for j, tc := range msg.ToolCalls {
				compatMsg.ToolCalls[j] = openaiToolCall{
					ID:   tc.ID,
					Type: tc.Type,
					Function: openaiToolCallFunc{
						Name:      tc.Function.Name,
						Arguments: tc.Function.Arguments,
					},
				}
			}
		}

		result[i] = compatMsg
	}
	return result
}

// ConvertTools converts generic Tool into OpenAI format; shared by all OpenAI-compatible providers
func ConvertTools(tools []Tool) []openaiTool {
	result := make([]openaiTool, len(tools))
	for i, tool := range tools {
		var function *openaiToolFunction
		if tool.Type == "function" {
			function = &openaiToolFunction{
				Name:        tool.Function.Name,
				Description: tool.Function.Description,
				Parameters:  tool.Function.Parameters,
			}
		}
		result[i] = openaiTool{
			Type:     tool.Type,
			Function: function,
		}
	}
	return result
}

func buildOpenAIContentParts(msg Message) []openaiContentPart {
	contents := normalizeMessageContents(msg)
	if len(contents) == 0 {
		return nil
	}

	parts := make([]openaiContentPart, 0, len(contents))
	for _, content := range contents {
		switch strings.ToLower(content.Type) {
		case "", "text", "input_text":
			if content.Text == "" {
				continue
			}
			parts = append(parts, openaiContentPart{
				Type: "text",
				Text: content.Text,
			})
		case "image_url", "input_image":
			if content.ImageURL == nil || content.ImageURL.URL == "" {
				continue
			}
			parts = append(parts, openaiContentPart{
				Type: "image_url",
				ImageURL: &openaiImageURL{
					URL: content.ImageURL.URL,
				},
			})
		default:
			if content.Text != "" {
				parts = append(parts, openaiContentPart{
					Type: "text",
					Text: content.Text,
				})
			}
		}
	}

	return parts
}

func convertMessagesToResponsesInput(messages []Message) []responsesAPIInputItem {
	if len(messages) == 0 {
		return nil
	}

	items := make([]responsesAPIInputItem, 0, len(messages))
	for _, msg := range messages {
		contentItems := convertMessageContentsToResponsesContent(msg)
		if len(contentItems) == 0 {
			continue
		}
		items = append(items, responsesAPIInputItem{
			Type:    "message",
			Role:    msg.Role,
			Content: contentItems,
		})
	}
	return items
}

func convertMessageContentsToResponsesContent(msg Message) []responsesAPIContentItem {
	contents := normalizeMessageContents(msg)
	if len(contents) == 0 {
		return nil
	}

	result := make([]responsesAPIContentItem, 0, len(contents))
	for _, content := range contents {
		switch strings.ToLower(content.Type) {
		case "", "text", "input_text":
			if content.Text == "" {
				continue
			}
			result = append(result, responsesAPIContentItem{
				Type: "input_text",
				Text: content.Text,
			})
		case "image_url", "input_image":
			if content.ImageURL == nil || content.ImageURL.URL == "" {
				continue
			}
			result = append(result, responsesAPIContentItem{
				Type:     "input_image",
				ImageURL: &responsesAPIImageURL{URL: content.ImageURL.URL},
			})
		default:
			if content.Text != "" {
				result = append(result, responsesAPIContentItem{
					Type: "input_text",
					Text: content.Text,
				})
			}
		}
	}
	return result
}

func normalizeMessageContents(msg Message) []MessageContent {
	if len(msg.Contents) > 0 {
		return msg.Contents
	}
	if msg.Content != "" {
		return []MessageContent{{Type: "text", Text: msg.Content}}
	}
	return nil
}

func convertOpenAIContentToMessageContents(content interface{}) []MessageContent {
	switch val := content.(type) {
	case string:
		if val == "" {
			return nil
		}
		return []MessageContent{{Type: "text", Text: val}}
	case []interface{}:
		result := make([]MessageContent, 0, len(val))
		for _, item := range val {
			partMap, ok := item.(map[string]interface{})
			if !ok {
				continue
			}
			partType, _ := partMap["type"].(string)
			switch strings.ToLower(partType) {
			case "image_url":
				if imageData, ok := partMap["image_url"].(map[string]interface{}); ok {
					if urlValue, _ := imageData["url"].(string); urlValue != "" {
						result = append(result, MessageContent{
							Type: "image_url",
							ImageURL: &MessageImageURL{
								URL: urlValue,
							},
						})
					}
				}
			default:
				textValue, _ := partMap["text"].(string)
				if textValue == "" {
					continue
				}
				result = append(result, MessageContent{
					Type:        "text",
					Text:        textValue,
					Annotations: extractAnnotations(partMap),
					Logprobs:    extractLogprobs(partMap),
				})
			}
		}
		return result
	default:
		return nil
	}
}

func convertResponsesContentToMessageContents(items []responsesAPIContentItem) []MessageContent {
	if len(items) == 0 {
		return nil
	}

	result := make([]MessageContent, 0, len(items))
	for _, item := range items {
		switch strings.ToLower(item.Type) {
		case "output_image", "image", "image_url":
			if item.ImageURL == nil || item.ImageURL.URL == "" {
				continue
			}
			result = append(result, MessageContent{
				Type:     "image_url",
				ImageURL: &MessageImageURL{URL: item.ImageURL.URL},
			})
		default:
			if item.Text == "" {
				continue
			}
			result = append(result, MessageContent{
				Type:        "text",
				Text:        item.Text,
				Annotations: item.Annotations,
				Logprobs:    item.Logprobs,
			})
		}
	}
	return result
}

func joinMessageContentsText(contents []MessageContent) string {
	if len(contents) == 0 {
		return ""
	}
	var builder strings.Builder
	for _, content := range contents {
		if content.Type != "text" || content.Text == "" {
			continue
		}
		if builder.Len() > 0 {
			builder.WriteString("\n")
		}
		builder.WriteString(content.Text)
	}
	return builder.String()
}

func extractAnnotations(part map[string]interface{}) []map[string]interface{} {
	raw, ok := part["annotations"].([]interface{})
	if !ok || len(raw) == 0 {
		return nil
	}
	result := make([]map[string]interface{}, 0, len(raw))
	for _, entry := range raw {
		if data, ok := entry.(map[string]interface{}); ok {
			result = append(result, data)
		}
	}
	return result
}

func extractLogprobs(part map[string]interface{}) []map[string]interface{} {
	raw, ok := part["logprobs"].([]interface{})
	if !ok || len(raw) == 0 {
		return nil
	}
	result := make([]map[string]interface{}, 0, len(raw))
	for _, entry := range raw {
		if data, ok := entry.(map[string]interface{}); ok {
			result = append(result, data)
		}
	}
	return result
}
