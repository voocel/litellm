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

// ==================== Responses API Types ====================

// OpenAIResponsesRequest is a dedicated request type for OpenAI Responses API.
// It avoids polluting the generic Request with OpenAI-only fields.
type OpenAIResponsesRequest struct {
	Model string `json:"model"`

	// Use standard messages to build input items for Responses API.
	Messages []Message `json:"messages"`

	// Output configuration
	MaxOutputTokens *int `json:"max_output_tokens,omitempty"`

	// Sampling parameters
	Temperature *float64 `json:"temperature,omitempty"`
	TopP        *float64 `json:"top_p,omitempty"`

	// Text format configuration (replaces response_format in Responses API)
	ResponseFormat *ResponseFormat `json:"response_format,omitempty"`

	// Tool configuration
	Tools      []Tool `json:"tools,omitempty"`
	ToolChoice any    `json:"tool_choice,omitempty"`

	// Reasoning configuration (for gpt-5 and o-series models)
	ReasoningEffort  string `json:"reasoning_effort,omitempty"`  // none, low, medium, high
	ReasoningSummary string `json:"reasoning_summary,omitempty"` // auto, concise, detailed

	// Unified thinking control (optional). If set, it maps to reasoning fields.
	Thinking *ThinkingConfig `json:"thinking,omitempty"`

	// APIKey overrides the provider-level API key for this single request.
	APIKey string `json:"-"`

	// OnPayload is called with the serialized JSON body before sending
	// the HTTP request. Useful for debugging and logging API calls.
	OnPayload func(provider string, payload []byte) `json:"-"`
}

// responsesAPIRequest represents the request structure for OpenAI Responses API
// See: https://platform.openai.com/docs/api-reference/responses/create
type responsesAPIRequest struct {
	// Required
	Model string `json:"model"`

	// Input: can be string or array of input items
	// For simplicity, we use array format; simple string input should be wrapped
	Input []responsesAPIInputItem `json:"input,omitempty"`

	// System instructions
	Instructions string `json:"instructions,omitempty"`

	// Conversation management (mutually exclusive with PreviousResponseID)
	Conversation       any    `json:"conversation,omitempty"`         // string or object
	PreviousResponseID string `json:"previous_response_id,omitempty"` // for multi-turn without conversation

	// Output configuration
	MaxOutputTokens *int     `json:"max_output_tokens,omitempty"`
	MaxToolCalls    *int     `json:"max_tool_calls,omitempty"`
	Include         []string `json:"include,omitempty"` // e.g., "message.output_text.logprobs", "reasoning.encrypted_content"
	Stream          *bool    `json:"stream,omitempty"`

	// Sampling parameters
	Temperature *float64 `json:"temperature,omitempty"`
	TopP        *float64 `json:"top_p,omitempty"`

	// Text format configuration (replaces response_format in Responses API)
	Text *responsesAPITextConfig `json:"text,omitempty"`

	// Truncation strategy: "auto" or "disabled" (default)
	Truncation string `json:"truncation,omitempty"`

	// Tool configuration
	Tools             []openaiTool `json:"tools,omitempty"`
	ToolChoice        any          `json:"tool_choice,omitempty"`
	ParallelToolCalls *bool        `json:"parallel_tool_calls,omitempty"`

	// Reasoning configuration (for gpt-5 and o-series models)
	Reasoning *responsesAPIReasoning `json:"reasoning,omitempty"`

	// Caching
	PromptCacheKey       string `json:"prompt_cache_key,omitempty"`
	PromptCacheRetention string `json:"prompt_cache_retention,omitempty"` // e.g., "24h"

	// Metadata and identification
	Metadata         map[string]string `json:"metadata,omitempty"` // max 16 key-value pairs
	SafetyIdentifier string            `json:"safety_identifier,omitempty"`

	// Service configuration
	ServiceTier string `json:"service_tier,omitempty"` // auto, default, flex, priority
	Background  *bool  `json:"background,omitempty"`
	Store       *bool  `json:"store,omitempty"`

	// Prompt template reference
	Prompt map[string]interface{} `json:"prompt,omitempty"`
}

// responsesAPITextConfig configures text output format
// See: https://platform.openai.com/docs/api-reference/responses/create
type responsesAPITextConfig struct {
	Format *responsesAPITextFormat `json:"format,omitempty"`
}

// responsesAPITextFormat specifies the text format type
type responsesAPITextFormat struct {
	Type       string                      `json:"type"` // "text", "json_object", "json_schema"
	JSONSchema *responsesAPIJSONSchemaSpec `json:"json_schema,omitempty"`
}

// responsesAPIJSONSchemaSpec defines JSON schema for structured output
type responsesAPIJSONSchemaSpec struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Schema      any    `json:"schema"`
	Strict      *bool  `json:"strict,omitempty"`
}

// responsesAPIReasoning configures reasoning behavior for Responses API
// See: https://platform.openai.com/docs/api-reference/responses/create
type responsesAPIReasoning struct {
	Effort  string `json:"effort,omitempty"`  // none, low, medium, high
	Summary string `json:"summary,omitempty"` // auto, concise, detailed
}

// responsesAPIInputItem represents an input item in Responses API
type responsesAPIInputItem struct {
	Type    string                    `json:"type"`
	Role    string                    `json:"role,omitempty"`
	Content []responsesAPIContentItem `json:"content,omitempty"`
	ID      string                    `json:"id,omitempty"`
}

// responsesAPIResponse represents the response from Responses API
// See: https://platform.openai.com/docs/api-reference/responses/object
type responsesAPIResponse struct {
	ID                string                   `json:"id"`
	Object            string                   `json:"object"` // "response"
	CreatedAt         int64                    `json:"created_at,omitempty"`
	Model             string                   `json:"model"`
	Status            string                   `json:"status,omitempty"` // queued, in_progress, completed, failed, cancelled
	OutputText        string                   `json:"output_text"`
	Output            []responsesAPIOutputItem `json:"output"`
	Usage             responsesAPIUsage        `json:"usage"`
	Error             *responsesAPIError       `json:"error,omitempty"`
	IncompleteDetails *responsesAPIIncomplete  `json:"incomplete_details,omitempty"`
	Metadata          map[string]string        `json:"metadata,omitempty"`
}

// responsesAPIError represents error information in response
type responsesAPIError struct {
	Code    string `json:"code,omitempty"`
	Message string `json:"message,omitempty"`
}

// responsesAPIIncomplete represents incomplete response details
type responsesAPIIncomplete struct {
	Reason string `json:"reason,omitempty"`
}

// responsesAPIOutputItem represents an output item (message, reasoning, tool_call)
type responsesAPIOutputItem struct {
	ID        string                    `json:"id"`
	Type      string                    `json:"type"` // message, reasoning, tool_call
	Role      string                    `json:"role,omitempty"`
	Status    string                    `json:"status,omitempty"`
	CallID    string                    `json:"call_id,omitempty"`
	Name      string                    `json:"name,omitempty"`
	Arguments string                    `json:"arguments,omitempty"`
	Content   []responsesAPIContentItem `json:"content,omitempty"`
	Summary   []responsesAPISummaryItem `json:"summary,omitempty"`
	Output    []responsesAPIToolOutput  `json:"output,omitempty"`
}

// responsesAPIContentItem represents content within an output item
type responsesAPIContentItem struct {
	Type        string                   `json:"type"` // input_text, output_text, input_image, etc.
	Text        string                   `json:"text,omitempty"`
	Annotations []map[string]interface{} `json:"annotations,omitempty"`
	Logprobs    []map[string]interface{} `json:"logprobs,omitempty"`
	ImageURL    *responsesAPIImageURL    `json:"image_url,omitempty"`
}

// responsesAPIImageURL represents an image URL in Responses API
type responsesAPIImageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`
}

// responsesAPISummaryItem represents a reasoning summary item
type responsesAPISummaryItem struct {
	Text string `json:"text"`
}

// responsesAPIToolOutput represents tool output in Responses API
type responsesAPIToolOutput struct {
	Type    string `json:"type"`
	Content string `json:"content,omitempty"`
}

// responsesAPIUsage represents token usage in Responses API
type responsesAPIUsage struct {
	InputTokens         int                              `json:"input_tokens"`
	OutputTokens        int                              `json:"output_tokens"`
	TotalTokens         int                              `json:"total_tokens"`
	OutputTokensDetails *responsesAPIOutputTokensDetails `json:"output_tokens_details,omitempty"`
}

// responsesAPIOutputTokensDetails contains detailed output token breakdown
type responsesAPIOutputTokensDetails struct {
	ReasoningTokens int `json:"reasoning_tokens,omitempty"`
}

// responsesAPICompletedEvent represents the response.completed streaming event
type responsesAPICompletedEvent struct {
	Response struct {
		ID                string                   `json:"id,omitempty"`
		Object            string                   `json:"object,omitempty"`
		Status            string                   `json:"status,omitempty"`
		Model             string                   `json:"model"`
		Usage             responsesAPIUsage        `json:"usage"`
		IncompleteDetails *responsesAPIIncomplete  `json:"incomplete_details,omitempty"`
		Output            []responsesAPIOutputItem `json:"output,omitempty"`
	} `json:"response"`
	SequenceNumber int `json:"sequence_number,omitempty"`
}

// responsesAPIErrorEvent represents the response.error streaming event
type responsesAPIErrorEvent struct {
	Error struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
}

// ==================== Responses API Methods ====================

// Responses executes a Responses API request for OpenAI.
func (p *OpenAIProvider) Responses(ctx context.Context, req *OpenAIResponsesRequest) (*Response, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}
	if req == nil {
		return nil, fmt.Errorf("openai: responses request cannot be nil")
	}
	if req.Model == "" {
		return nil, fmt.Errorf("openai: model is required for responses request")
	}
	if len(req.Messages) == 0 {
		return nil, fmt.Errorf("openai: messages are required for responses request")
	}

	return p.completeWithResponsesAPI(ctx, req)
}

// ResponsesStream executes a streaming Responses API request for OpenAI.
func (p *OpenAIProvider) ResponsesStream(ctx context.Context, req *OpenAIResponsesRequest) (StreamReader, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}
	if req == nil {
		return nil, fmt.Errorf("openai: responses request cannot be nil")
	}
	if req.Model == "" {
		return nil, fmt.Errorf("openai: model is required for responses request")
	}
	if len(req.Messages) == 0 {
		return nil, fmt.Errorf("openai: messages are required for responses request")
	}

	return p.streamWithResponsesAPI(ctx, req)
}

// buildResponsesAPIRequest maps OpenAIResponsesRequest into the official Responses API payload.
func (p *OpenAIProvider) buildResponsesAPIRequest(req *OpenAIResponsesRequest) responsesAPIRequest {
	responsesReq := responsesAPIRequest{
		Model: req.Model,
	}

	instructions, filteredMessages := extractResponsesInstructions(req.Messages)
	if instructions != "" {
		responsesReq.Instructions = instructions
	}
	if inputItems := convertMessagesToResponsesInput(filteredMessages); len(inputItems) > 0 {
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
	if req.Thinking != nil {
		thinking := normalizeThinking(&Request{Thinking: req.Thinking})
		if thinking.Type == "disabled" {
			responsesReq.Reasoning = &responsesAPIReasoning{
				Effort: "none",
			}
		} else if responsesReq.Reasoning == nil {
			responsesReq.Reasoning = &responsesAPIReasoning{
				Summary: "auto",
			}
		}
	}

	// Responses API supports tools
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

	if req.ToolChoice != nil {
		responsesReq.ToolChoice = req.ToolChoice
	}

	// Convert response format to text.format structure for Responses API
	responsesReq.Text = p.convertToTextFormat(req)

	if req.Temperature != nil {
		responsesReq.Temperature = req.Temperature
	}
	if req.TopP != nil {
		responsesReq.TopP = req.TopP
	}

	if req.MaxOutputTokens != nil {
		responsesReq.MaxOutputTokens = req.MaxOutputTokens
	}

	return responsesReq
}

// completeWithResponsesAPI uses OpenAI Responses API.
func (p *OpenAIProvider) completeWithResponsesAPI(ctx context.Context, req *OpenAIResponsesRequest) (*Response, error) {
	responsesReq := p.buildResponsesAPIRequest(req)

	body, err := json.Marshal(responsesReq)
	if err != nil {
		return nil, fmt.Errorf("openai: marshal responses request: %w", err)
	}

	if req.OnPayload != nil {
		req.OnPayload("openai", body)
	}

	// Use Responses API
	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.buildURL("/responses"), bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("openai: create responses request: %w", err)
	}

	p.setHeaders(httpReq, &Request{APIKey: req.APIKey})

	resp, err := p.HTTPClient().Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, NewHTTPError("openai", resp.StatusCode, string(body))
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
		FinishReason: NormalizeFinishReason(responsesResp.Status),
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

// convertToTextFormat converts ResponseFormat to Responses API text.format structure.
// See: https://platform.openai.com/docs/api-reference/responses/create
func (p *OpenAIProvider) convertToTextFormat(req *OpenAIResponsesRequest) *responsesAPITextConfig {
	rf := req.ResponseFormat

	if rf == nil {
		return nil
	}

	textConfig := &responsesAPITextConfig{
		Format: &responsesAPITextFormat{
			Type: rf.Type,
		},
	}

	// Convert json_schema format
	if rf.Type == "json_schema" && rf.JSONSchema != nil {
		schema := rf.JSONSchema.Schema

		// Clean schema for OpenAI compatibility
		schema = p.cleanSchemaForOpenAI(schema)

		// Apply strict mode if requested
		if rf.JSONSchema.Strict != nil && *rf.JSONSchema.Strict {
			schema = p.ensureStrictSchema(schema)
		}

		textConfig.Format.JSONSchema = &responsesAPIJSONSchemaSpec{
			Name:        rf.JSONSchema.Name,
			Description: rf.JSONSchema.Description,
			Schema:      schema,
			Strict:      rf.JSONSchema.Strict,
		}
	}

	return textConfig
}

// convertResponsesUsage converts Responses API usage to common Usage format
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

// streamWithResponsesAPI handles streaming for the /responses endpoint.
func (p *OpenAIProvider) streamWithResponsesAPI(ctx context.Context, req *OpenAIResponsesRequest) (StreamReader, error) {
	responsesReq := p.buildResponsesAPIRequest(req)
	stream := true
	responsesReq.Stream = &stream

	body, err := json.Marshal(responsesReq)
	if err != nil {
		return nil, fmt.Errorf("openai: marshal responses request: %w", err)
	}

	if req.OnPayload != nil {
		req.OnPayload("openai", body)
	}

	// Use Responses API with stream=true in request body
	url := p.buildURL("/responses")
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("openai: create responses request: %w", err)
	}

	p.setHeaders(httpReq, &Request{APIKey: req.APIKey})
	// Ensure Accept header is set for SSE
	httpReq.Header.Set("Accept", "text/event-stream")

	resp, err := p.HTTPClient().Do(httpReq)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, NewHTTPError("openai", resp.StatusCode, string(body))
	}

	scanner := bufio.NewScanner(resp.Body)
	buf := make([]byte, 0, 64*1024)
	scanner.Buffer(buf, 1024*1024)

	return &responsesAPIStreamReader{
		resp:            resp,
		scanner:         scanner,
		provider:        "openai",
		model:           req.Model,
		toolCallSeenMap: make(map[string]bool),
	}, nil
}

// ==================== Responses API Stream Reader ====================

// responsesAPIStreamReader implements streaming for OpenAI /responses endpoint
type responsesAPIStreamReader struct {
	resp            *http.Response
	scanner         *bufio.Scanner
	provider        string
	model           string
	toolCallSeenMap map[string]bool
	lastSequence    int
	done            bool
}

func (r *responsesAPIStreamReader) Next() (*StreamChunk, error) {
	if r.done {
		return &StreamChunk{Done: true, Provider: r.provider, Model: r.model}, nil
	}

	// Responses API streaming events (SSE):
	// See: https://platform.openai.com/docs/api-reference/responses-streaming
	//
	// Lifecycle events:
	// - response.created, response.in_progress, response.completed, response.failed
	//
	// Content events:
	// - response.output_item.added, response.output_item.done
	// - response.content_part.added, response.content_part.done
	// - response.output_text.delta, response.output_text.done
	// - response.reasoning_text.delta, response.reasoning_text.done
	// - response.reasoning_summary_text.delta, response.reasoning_summary_text.done
	// - response.refusal.delta, response.refusal.done
	//
	// Tool events:
	// - response.function_call_arguments.delta, response.function_call_arguments.done
	// - response.file_search_call.*, response.code_interpreter_call.*

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
		// === Content Events ===
		case "response.output_text.delta":
			var delta struct {
				Delta          string `json:"delta"`
				ItemID         string `json:"item_id"`
				OutputIndex    *int   `json:"output_index,omitempty"`
				ContentIndex   *int   `json:"content_index,omitempty"`
				SequenceNumber int    `json:"sequence_number,omitempty"`
			}
			if err := json.Unmarshal([]byte(data), &delta); err != nil {
				return nil, fmt.Errorf("openai: parse output delta: %w", err)
			}
			if !r.shouldEmit(&delta.SequenceNumber) {
				continue
			}
			streamChunk.Type = "content"
			streamChunk.Content = delta.Delta
			streamChunk.ItemID = delta.ItemID
			streamChunk.OutputIndex = delta.OutputIndex
			streamChunk.ContentIndex = delta.ContentIndex
			return streamChunk, nil

		// === Refusal Events (model refuses to respond) ===
		case "response.refusal.delta":
			var delta struct {
				Delta          string `json:"delta"`
				ItemID         string `json:"item_id"`
				OutputIndex    *int   `json:"output_index,omitempty"`
				ContentIndex   *int   `json:"content_index,omitempty"`
				SequenceNumber int    `json:"sequence_number,omitempty"`
			}
			if err := json.Unmarshal([]byte(data), &delta); err != nil {
				return nil, fmt.Errorf("openai: parse refusal delta: %w", err)
			}
			if !r.shouldEmit(&delta.SequenceNumber) {
				continue
			}
			streamChunk.Type = "refusal"
			streamChunk.Content = delta.Delta
			streamChunk.ItemID = delta.ItemID
			streamChunk.OutputIndex = delta.OutputIndex
			streamChunk.ContentIndex = delta.ContentIndex
			return streamChunk, nil

		case "response.refusal.done":
			var done struct {
				Refusal        string `json:"refusal"`
				ItemID         string `json:"item_id"`
				OutputIndex    *int   `json:"output_index,omitempty"`
				ContentIndex   *int   `json:"content_index,omitempty"`
				SequenceNumber int    `json:"sequence_number,omitempty"`
			}
			if err := json.Unmarshal([]byte(data), &done); err != nil {
				return nil, fmt.Errorf("openai: parse refusal done: %w", err)
			}
			if done.Refusal == "" {
				continue
			}
			if !r.shouldEmit(&done.SequenceNumber) {
				continue
			}
			streamChunk.Type = "refusal"
			streamChunk.Content = done.Refusal
			streamChunk.ItemID = done.ItemID
			streamChunk.OutputIndex = done.OutputIndex
			streamChunk.ContentIndex = done.ContentIndex
			return streamChunk, nil

		// === Reasoning Events ===
		case "response.reasoning_text.delta":
			var delta struct {
				Delta          string `json:"delta"`
				ItemID         string `json:"item_id"`
				OutputIndex    *int   `json:"output_index,omitempty"`
				ContentIndex   *int   `json:"content_index,omitempty"`
				SequenceNumber int    `json:"sequence_number,omitempty"`
			}
			if err := json.Unmarshal([]byte(data), &delta); err != nil {
				return nil, fmt.Errorf("openai: parse reasoning delta: %w", err)
			}
			if !r.shouldEmit(&delta.SequenceNumber) {
				continue
			}
			streamChunk.Type = "reasoning"
			streamChunk.Reasoning = &ReasoningChunk{Content: delta.Delta}
			streamChunk.ItemID = delta.ItemID
			streamChunk.OutputIndex = delta.OutputIndex
			streamChunk.ContentIndex = delta.ContentIndex
			return streamChunk, nil

		case "response.reasoning_text.done":
			streamChunk.Type = "reasoning_done"
			streamChunk.Reasoning = &ReasoningChunk{Done: true}
			return streamChunk, nil

		case "response.reasoning_summary_text.delta":
			var delta struct {
				Delta          string `json:"delta"`
				ItemID         string `json:"item_id"`
				OutputIndex    *int   `json:"output_index,omitempty"`
				SummaryIndex   *int   `json:"summary_index,omitempty"`
				SequenceNumber int    `json:"sequence_number,omitempty"`
			}
			if err := json.Unmarshal([]byte(data), &delta); err != nil {
				return nil, fmt.Errorf("openai: parse reasoning summary delta: %w", err)
			}
			if !r.shouldEmit(&delta.SequenceNumber) {
				continue
			}
			streamChunk.Type = "reasoning"
			streamChunk.Reasoning = &ReasoningChunk{Summary: delta.Delta}
			streamChunk.ItemID = delta.ItemID
			streamChunk.OutputIndex = delta.OutputIndex
			return streamChunk, nil

		case "response.reasoning_summary_text.done":
			streamChunk.Type = "reasoning_summary_done"
			streamChunk.Reasoning = &ReasoningChunk{Done: true}
			return streamChunk, nil

		// === Tool Call Events ===
		case "response.function_call_arguments.delta":
			var delta struct {
				Delta          string `json:"delta"`
				ItemID         string `json:"item_id"`
				OutputIndex    *int   `json:"output_index,omitempty"`
				SequenceNumber int    `json:"sequence_number,omitempty"`
			}
			if err := json.Unmarshal([]byte(data), &delta); err != nil {
				return nil, fmt.Errorf("openai: parse tool delta: %w", err)
			}
			if !r.shouldEmit(&delta.SequenceNumber) {
				continue
			}
			streamChunk.Type = "tool_call_delta"
			streamChunk.ToolCallDelta = &ToolCallDelta{
				ID:             delta.ItemID,
				Type:           "function",
				ArgumentsDelta: delta.Delta,
				OutputIndex:    delta.OutputIndex,
				ItemID:         delta.ItemID,
			}
			streamChunk.ItemID = delta.ItemID
			streamChunk.OutputIndex = delta.OutputIndex
			if delta.ItemID != "" {
				r.toolCallSeenMap[delta.ItemID] = true
			}
			return streamChunk, nil

		case "response.function_call_arguments.done":
			var done struct {
				Name           string `json:"name"`
				Arguments      string `json:"arguments"`
				ItemID         string `json:"item_id"`
				OutputIndex    *int   `json:"output_index,omitempty"`
				SequenceNumber int    `json:"sequence_number,omitempty"`
			}
			if err := json.Unmarshal([]byte(data), &done); err != nil {
				return nil, fmt.Errorf("openai: parse tool done: %w", err)
			}
			if !r.shouldEmit(&done.SequenceNumber) {
				continue
			}
			seenDelta := done.ItemID != "" && r.toolCallSeenMap[done.ItemID]
			streamChunk.ToolCallDelta = &ToolCallDelta{
				ID:           done.ItemID,
				Type:         "function",
				FunctionName: done.Name,
				OutputIndex:  done.OutputIndex,
				ItemID:       done.ItemID,
			}
			if !seenDelta {
				streamChunk.Type = "tool_call_delta"
				streamChunk.ToolCallDelta.ArgumentsDelta = done.Arguments
			} else {
				streamChunk.Type = "tool_call_end"
			}
			streamChunk.ItemID = done.ItemID
			streamChunk.OutputIndex = done.OutputIndex
			if done.ItemID != "" {
				r.toolCallSeenMap[done.ItemID] = true
			}
			return streamChunk, nil

		// === Output Item Events ===
		case "response.output_item.added":
			var item struct {
				Item struct {
					ID     string `json:"id"`
					Type   string `json:"type"`
					Role   string `json:"role,omitempty"`
					Status string `json:"status,omitempty"`
				} `json:"item"`
				OutputIndex    *int `json:"output_index,omitempty"`
				SequenceNumber int  `json:"sequence_number,omitempty"`
			}
			if err := json.Unmarshal([]byte(data), &item); err != nil {
				return nil, fmt.Errorf("openai: parse output item added: %w", err)
			}
			if !r.shouldEmit(&item.SequenceNumber) {
				continue
			}
			streamChunk.Type = "output_item_added"
			streamChunk.ItemID = item.Item.ID
			streamChunk.OutputIndex = item.OutputIndex
			return streamChunk, nil

		case "response.output_item.done":
			// Output item complete, continue processing
			continue

		// === Lifecycle Events ===
		case "response.created", "response.in_progress", "response.queued":
			// Lifecycle events - continue processing
			continue

		case "response.completed":
			var completed responsesAPICompletedEvent
			if err := json.Unmarshal([]byte(data), &completed); err != nil {
				return nil, fmt.Errorf("openai: parse responses completed: %w", err)
			}
			if !r.shouldEmit(&completed.SequenceNumber) {
				continue
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

		case "response.incomplete":
			// Handle truncated output (e.g., max_output_tokens reached)
			var incomplete responsesAPICompletedEvent
			if err := json.Unmarshal([]byte(data), &incomplete); err != nil {
				return nil, fmt.Errorf("openai: parse responses incomplete: %w", err)
			}
			if !r.shouldEmit(&incomplete.SequenceNumber) {
				continue
			}
			if incomplete.Response.Model != "" {
				r.model = incomplete.Response.Model
				streamChunk.Model = incomplete.Response.Model
			}
			streamChunk.Done = true
			streamChunk.FinishReason = "length"
			if usage := convertResponsesUsage(incomplete.Response.Usage); usage != nil {
				streamChunk.Usage = usage
			}
			r.done = true
			return streamChunk, nil

		case "response.failed":
			var failed struct {
				Response struct {
					Error struct {
						Code    string `json:"code"`
						Message string `json:"message"`
					} `json:"error"`
				} `json:"response"`
				SequenceNumber int `json:"sequence_number,omitempty"`
			}
			if err := json.Unmarshal([]byte(data), &failed); err != nil {
				return nil, fmt.Errorf("openai: parse responses failed: %w", err)
			}
			if !r.shouldEmit(&failed.SequenceNumber) {
				continue
			}
			return nil, fmt.Errorf("openai: response failed: [%s] %s", failed.Response.Error.Code, failed.Response.Error.Message)

		// === Error Event ===
		case "error":
			var responseErr responsesAPIErrorEvent
			if err := json.Unmarshal([]byte(data), &responseErr); err != nil {
				return nil, fmt.Errorf("openai: parse responses error: %w", err)
			}
			return nil, fmt.Errorf("openai: stream error: %s", responseErr.Error.Message)

		// === Content Part Events ===
		case "response.content_part.added", "response.content_part.done":
			// Content part lifecycle, continue processing
			continue

		case "response.output_text.done":
			// Text finished; wait for response.completed to carry usage
			continue

		// === File Search Events ===
		case "response.file_search_call.in_progress", "response.file_search_call.searching", "response.file_search_call.completed":
			// File search events - continue processing
			continue

		// === Code Interpreter Events ===
		case "response.code_interpreter_call.in_progress", "response.code_interpreter_call.interpreting", "response.code_interpreter_call.completed":
			// Code interpreter events - continue processing
			continue

		case "response.code_interpreter_call.code.delta":
			var delta struct {
				Delta          string `json:"delta"`
				ItemID         string `json:"item_id"`
				OutputIndex    *int   `json:"output_index,omitempty"`
				SequenceNumber int    `json:"sequence_number,omitempty"`
			}
			if err := json.Unmarshal([]byte(data), &delta); err != nil {
				return nil, fmt.Errorf("openai: parse code interpreter delta: %w", err)
			}
			if !r.shouldEmit(&delta.SequenceNumber) {
				continue
			}
			streamChunk.Type = "code_interpreter_delta"
			streamChunk.Content = delta.Delta
			streamChunk.ItemID = delta.ItemID
			streamChunk.OutputIndex = delta.OutputIndex
			return streamChunk, nil

		case "response.code_interpreter_call.code.done":
			// Code interpreter code complete
			continue

		// === Web Search Events ===
		case "response.web_search_call.in_progress", "response.web_search_call.searching", "response.web_search_call.completed":
			continue

		// === Image Generation Events ===
		case "response.image_generation_call.in_progress", "response.image_generation_call.generating",
			"response.image_generation_call.partial_image", "response.image_generation_call.completed":
			continue

		// === MCP (Model Context Protocol) Events ===
		case "response.mcp_call.in_progress", "response.mcp_call.completed", "response.mcp_call.failed",
			"response.mcp_call_arguments.delta", "response.mcp_call_arguments.done",
			"response.mcp_list_tools.in_progress", "response.mcp_list_tools.completed", "response.mcp_list_tools.failed":
			continue

		// === Reasoning Summary Part Events ===
		case "response.reasoning_summary_part.added", "response.reasoning_summary_part.done":
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

func (r *responsesAPIStreamReader) shouldEmit(sequenceNumber *int) bool {
	if sequenceNumber == nil || *sequenceNumber == 0 {
		return true
	}
	if *sequenceNumber <= r.lastSequence {
		return false
	}
	r.lastSequence = *sequenceNumber
	return true
}

// ==================== Responses API Conversion ====================

// extractResponsesInstructions extracts system/developer messages into instructions and returns remaining messages.
func extractResponsesInstructions(messages []Message) (string, []Message) {
	if len(messages) == 0 {
		return "", messages
	}

	var instructionParts []string
	filtered := make([]Message, 0, len(messages))
	for _, msg := range messages {
		role := strings.ToLower(strings.TrimSpace(msg.Role))
		if role == "system" || role == "developer" {
			text := msg.Content
			if text == "" {
				text = joinMessageContentsText(normalizeMessageContents(msg))
			}
			if strings.TrimSpace(text) != "" {
				instructionParts = append(instructionParts, text)
			}
			continue
		}
		filtered = append(filtered, msg)
	}

	instructions := strings.TrimSpace(strings.Join(instructionParts, "\n"))
	return instructions, filtered
}

// convertMessagesToResponsesInput converts messages to Responses API input format
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

// convertMessageContentsToResponsesContent converts message contents to Responses API format
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
				Type: "input_image",
				ImageURL: &responsesAPIImageURL{
					URL:    content.ImageURL.URL,
					Detail: content.ImageURL.Detail,
				},
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

// convertResponsesContentToMessageContents converts Responses API content to common format
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
				Type: "image_url",
				ImageURL: &MessageImageURL{
					URL:    item.ImageURL.URL,
					Detail: item.ImageURL.Detail,
				},
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
