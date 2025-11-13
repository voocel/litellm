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
	"sync/atomic"
	"time"
)

// GeminiProvider implements Google Gemini API integration
type GeminiProvider struct {
	*BaseProvider
}

// Package-level counter for generating unique tool call IDs
var geminiToolCallCounter atomic.Uint64

// NewGemini creates a new Gemini provider
func NewGemini(config ProviderConfig) *GeminiProvider {
	baseProvider := NewBaseProvider("gemini", config)

	return &GeminiProvider{
		BaseProvider: baseProvider,
	}
}

func (p *GeminiProvider) SupportsModel(model string) bool {
	for _, m := range p.Models() {
		if m.ID == model {
			return true
		}
	}
	return false
}

func (p *GeminiProvider) Models() []ModelInfo {
	return []ModelInfo{
		{
			ID: "gemini-2.5-pro", Provider: "gemini", Name: "Gemini 2.5 Pro", MaxTokens: 2000000,
			Capabilities: []string{"chat", "vision", "code", "function_call", "thinking"},
		},
		{
			ID: "gemini-2.5-flash", Provider: "gemini", Name: "Gemini 2.5 Flash", MaxTokens: 1000000,
			Capabilities: []string{"chat", "vision", "function_call", "thinking"},
		},
		{
			ID: "gemini-2.0-flash", Provider: "gemini", Name: "Gemini 2.0 Flash", MaxTokens: 1000000,
			Capabilities: []string{"chat", "vision", "function_call", "tool_use"},
		},
		{
			ID: "gemini-2.0-flash-lite", Provider: "gemini", Name: "Gemini 2.0 Flash Lite", MaxTokens: 1000000,
			Capabilities: []string{"chat", "function_call"},
		},
		{
			ID: "gemini-2.0-pro", Provider: "gemini", Name: "Gemini 2.0 Pro Experimental", MaxTokens: 2000000,
			Capabilities: []string{"chat", "vision", "code", "function_call", "tool_use"},
		},
	}
}

// supportsThinking checks if the model supports thinking capability
func (p *GeminiProvider) supportsThinking(model string) bool {
	for _, m := range p.Models() {
		if m.ID == model {
			for _, cap := range m.Capabilities {
				if cap == "thinking" {
					return true
				}
			}
		}
	}
	return false
}

func (p *GeminiProvider) Chat(ctx context.Context, req *Request) (*Response, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}

	modelName := req.Model

	geminiReq := geminiRequest{
		Contents: make([]geminiContent, 0),
	}

	// Set generation configuration
	if req.Temperature != nil || req.MaxTokens != nil || req.ResponseFormat != nil || len(req.Stop) > 0 || p.supportsThinking(modelName) {
		geminiReq.GenerationConfig = &geminiGenerationConfig{
			Temperature:     req.Temperature,
			MaxOutputTokens: req.MaxTokens,
			StopSequences:   req.Stop, // Map Stop to stopSequences for Gemini
		}

		// Enable thinking for models that support it
		if p.supportsThinking(modelName) {
			geminiReq.GenerationConfig.ThinkingConfig = &geminiThinkingConfig{
				ThinkingBudget:  intPtr(-1), // Dynamic thinking budget
				IncludeThoughts: boolPtr(true),
			}
		}

		// Handle response format
		if req.ResponseFormat != nil {
			switch req.ResponseFormat.Type {
			case "json_object":
				geminiReq.GenerationConfig.ResponseMimeType = "application/json"
			case "json_schema":
				geminiReq.GenerationConfig.ResponseMimeType = "application/json"
				if req.ResponseFormat.JSONSchema != nil && req.ResponseFormat.JSONSchema.Schema != nil {
					if schema, ok := req.ResponseFormat.JSONSchema.Schema.(map[string]any); ok {
						geminiReq.GenerationConfig.ResponseSchema = &geminiResponseSchema{
							Type:        "object",
							Description: req.ResponseFormat.JSONSchema.Description,
						}
						if props, ok := schema["properties"].(map[string]any); ok {
							geminiReq.GenerationConfig.ResponseSchema.Properties = props
						}
						if required, ok := schema["required"].([]any); ok {
							reqFields := make([]string, len(required))
							for i, field := range required {
								if fieldStr, ok := field.(string); ok {
									reqFields[i] = fieldStr
								}
							}
							geminiReq.GenerationConfig.ResponseSchema.Required = reqFields
						}
					}
				}
			}
		}
	}

	// Handle system messages and regular messages
	var systemMessage string
	for _, msg := range req.Messages {
		if msg.Role == "system" {
			// Gemini uses systemInstruction for system messages
			systemMessage = msg.Content
			continue
		}

		content := geminiContent{
			Role:  p.convertRole(msg.Role),
			Parts: []geminiPart{},
		}

		// Add cache control if specified for this message
		if msg.CacheControl != nil {
			content.CacheControl = &geminiCacheControl{
				Type: msg.CacheControl.Type,
			}
		}

		if msg.Content != "" {
			content.Parts = append(content.Parts, geminiPart{Text: msg.Content})
		}

		// Handle tool calls
		if len(msg.ToolCalls) > 0 {
			for _, toolCall := range msg.ToolCalls {
				var args map[string]any
				if err := json.Unmarshal([]byte(toolCall.Function.Arguments), &args); err != nil {
					args = map[string]any{"input": toolCall.Function.Arguments}
				}
				content.Parts = append(content.Parts, geminiPart{
					FunctionCall: &geminiFunctionCall{
						Name: toolCall.Function.Name,
						Args: args,
					},
				})
			}
		}

		// Handle tool responses
		if msg.ToolCallID != "" {
			var response map[string]any
			if err := json.Unmarshal([]byte(msg.Content), &response); err != nil {
				response = map[string]any{"result": msg.Content}
			}

			// Get tool name from context or generate one
			toolName := "function_result"
			if req.Extra != nil {
				if name, ok := req.Extra["tool_name"].(string); ok {
					toolName = name
				}
			}

			content.Parts = []geminiPart{{
				FunctionResponse: &geminiFunctionResponse{
					Name:     toolName,
					Response: response,
				},
			}}
		}

		if len(content.Parts) > 0 {
			geminiReq.Contents = append(geminiReq.Contents, content)
		}
	}

	if systemMessage != "" {
		geminiReq.SystemInstruction = &geminiContent{
			Parts: []geminiPart{{Text: systemMessage}},
		}
	}

	// Convert tool definitions
	if len(req.Tools) > 0 {
		tool := geminiTool{
			FunctionDeclarations: make([]geminiFunctionDeclaration, len(req.Tools)),
		}
		for i, t := range req.Tools {
			var params map[string]any
			if p, ok := t.Function.Parameters.(map[string]any); ok {
				params = p
			} else {
				if jsonBytes, err := json.Marshal(t.Function.Parameters); err == nil {
					if err := json.Unmarshal(jsonBytes, &params); err != nil {
						// Skip tools with invalid parameters
						continue
					}
				}
			}

			tool.FunctionDeclarations[i] = geminiFunctionDeclaration{
				Name:        t.Function.Name,
				Description: t.Function.Description,
				Parameters:  params,
			}
		}
		geminiReq.Tools = []geminiTool{tool}

		// Configure tool calling mode based on ToolChoice
		if req.ToolChoice != nil {
			geminiReq.ToolConfig = p.convertToolChoice(req.ToolChoice)
		}
	}

	// Send request
	body, err := json.Marshal(geminiReq)
	if err != nil {
		return nil, fmt.Errorf("gemini: marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/v1beta/models/%s:generateContent?key=%s",
		p.Config().BaseURL, modelName, p.Config().APIKey)

	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("gemini: create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := p.HTTPClient().Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("gemini: request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, err := io.ReadAll(resp.Body)
		if err != nil {
			return nil, fmt.Errorf("gemini: failed to read error response: %w", err)
		}
		return nil, fmt.Errorf("gemini: API error %d: %s", resp.StatusCode, string(body))
	}

	var geminiResp geminiResponse
	if err := json.NewDecoder(resp.Body).Decode(&geminiResp); err != nil {
		return nil, fmt.Errorf("gemini: decode response: %w", err)
	}

	response := &Response{
		Model:    modelName,
		Provider: "gemini",
	}

	if geminiResp.UsageMetadata != nil {
		response.Usage = Usage{
			PromptTokens:         geminiResp.UsageMetadata.PromptTokenCount,
			CompletionTokens:     geminiResp.UsageMetadata.CandidatesTokenCount,
			ReasoningTokens:      geminiResp.UsageMetadata.ThoughtsTokenCount,
			TotalTokens:          geminiResp.UsageMetadata.TotalTokenCount,
			CacheReadInputTokens: geminiResp.UsageMetadata.CachedContentTokenCount,
		}
	}

	if len(geminiResp.Candidates) > 0 {
		candidate := geminiResp.Candidates[0]
		response.FinishReason = candidate.FinishReason

		// Extract content, thinking, and tool calls
		var thinkingContent string
		for _, part := range candidate.Content.Parts {
			if part.Text != "" {
				// Check if this part is thinking content
				if part.Thought != nil && *part.Thought {
					thinkingContent += part.Text
				} else {
					response.Content += part.Text
				}
			}

			if part.FunctionCall != nil {
				args, _ := json.Marshal(part.FunctionCall.Args)
				response.ToolCalls = append(response.ToolCalls, ToolCall{
					ID:   p.generateToolCallID(),
					Type: "function",
					Function: FunctionCall{
						Name:      part.FunctionCall.Name,
						Arguments: string(args),
					},
				})
			}
		}

		// Populate reasoning data if thinking content exists
		if thinkingContent != "" {
			response.Reasoning = &ReasoningData{
				Content:    thinkingContent,
				Summary:    "Gemini thinking process",
				TokensUsed: geminiResp.UsageMetadata.ThoughtsTokenCount,
			}
		}
	}

	return response, nil
}

func (p *GeminiProvider) Stream(ctx context.Context, req *Request) (StreamReader, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}

	modelName := req.Model
	geminiReq := geminiRequest{
		Contents: make([]geminiContent, 0),
	}

	if req.Temperature != nil || req.MaxTokens != nil || len(req.Stop) > 0 || p.supportsThinking(modelName) {
		geminiReq.GenerationConfig = &geminiGenerationConfig{
			Temperature:     req.Temperature,
			MaxOutputTokens: req.MaxTokens,
			StopSequences:   req.Stop, // Map Stop to stopSequences for Gemini
		}

		// Enable thinking for models that support it
		if p.supportsThinking(modelName) {
			geminiReq.GenerationConfig.ThinkingConfig = &geminiThinkingConfig{
				ThinkingBudget:  intPtr(-1), // Dynamic thinking budget for complete thoughts
				IncludeThoughts: boolPtr(true),
			}
		}
	}

	// Handle messages (simplified version for streaming)
	var systemMessage string
	for _, msg := range req.Messages {
		if msg.Role == "system" {
			systemMessage = msg.Content
			continue
		}

		if msg.Content != "" {
			content := geminiContent{
				Role:  p.convertRole(msg.Role),
				Parts: []geminiPart{{Text: msg.Content}},
			}

			// Add cache control if specified for this message
			if msg.CacheControl != nil {
				content.CacheControl = &geminiCacheControl{
					Type: msg.CacheControl.Type,
				}
			}

			geminiReq.Contents = append(geminiReq.Contents, content)
		}
	}

	if systemMessage != "" {
		geminiReq.SystemInstruction = &geminiContent{
			Parts: []geminiPart{{Text: systemMessage}},
		}
	}

	body, err := json.Marshal(geminiReq)
	if err != nil {
		return nil, fmt.Errorf("gemini: marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/v1beta/models/%s:streamGenerateContent?key=%s",
		p.Config().BaseURL, modelName, p.Config().APIKey)

	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("gemini: create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := p.HTTPClient().Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("gemini: request failed: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		body, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			return nil, fmt.Errorf("gemini: failed to read stream error response: %w", err)
		}
		return nil, fmt.Errorf("gemini: API error %d: %s", resp.StatusCode, string(body))
	}

	return &geminiStreamReader{
		resp:     resp,
		scanner:  bufio.NewScanner(resp.Body),
		provider: "gemini",
	}, nil
}

// convertRole converts standard roles to Gemini roles
func (p *GeminiProvider) convertRole(role string) string {
	switch role {
	case "assistant":
		return "model"
	case "user":
		return "user"
	case "tool":
		return "function"
	default:
		return "user" // Default to user
	}
}

// generateToolCallID generates a unique tool call ID
func (p *GeminiProvider) generateToolCallID() string {
	timestamp := time.Now().Unix()
	counter := geminiToolCallCounter.Add(1)
	return fmt.Sprintf("call_%d_%d", timestamp, counter)
}

// convertToolChoice converts the unified ToolChoice to Gemini's toolConfig format
// According to Gemini API docs: https://ai.google.dev/gemini-api/docs/function-calling
// Supported modes: AUTO (default), ANY (force function call), NONE (disable function calling)
func (p *GeminiProvider) convertToolChoice(toolChoice any) *geminiToolConfig {
	if toolChoice == nil {
		return nil
	}

	// Handle string format: "auto", "any", "none"
	if mode, ok := toolChoice.(string); ok {
		modeUpper := strings.ToUpper(mode)
		if modeUpper == "AUTO" || modeUpper == "ANY" || modeUpper == "NONE" {
			return &geminiToolConfig{
				FunctionCallingConfig: &geminiFunctionCallingConfig{
					Mode: modeUpper,
				},
			}
		}
		// If it's "required", map to "ANY" (force function call)
		if mode == "required" {
			return &geminiToolConfig{
				FunctionCallingConfig: &geminiFunctionCallingConfig{
					Mode: "ANY",
				},
			}
		}
		return nil
	}

	// Handle map format: {"type": "...", "name": "..."}
	if choiceMap, ok := toolChoice.(map[string]any); ok {
		typeVal, hasType := choiceMap["type"].(string)
		if !hasType {
			return nil
		}

		switch typeVal {
		case "auto":
			return &geminiToolConfig{
				FunctionCallingConfig: &geminiFunctionCallingConfig{
					Mode: "AUTO",
				},
			}
		case "any", "required":
			return &geminiToolConfig{
				FunctionCallingConfig: &geminiFunctionCallingConfig{
					Mode: "ANY",
				},
			}
		case "none":
			return &geminiToolConfig{
				FunctionCallingConfig: &geminiFunctionCallingConfig{
					Mode: "NONE",
				},
			}
		case "function", "tool":
			// Specific function: {"type": "function", "name": "function_name"}
			if funcName, ok := choiceMap["name"].(string); ok {
				return &geminiToolConfig{
					FunctionCallingConfig: &geminiFunctionCallingConfig{
						Mode:                 "ANY",
						AllowedFunctionNames: []string{funcName},
					},
				}
			}
		}
	}

	return nil
}

func boolPtr(b bool) *bool {
	return &b
}

func intPtr(i int) *int {
	return &i
}

// geminiStreamReader implements streaming for Gemini
type geminiStreamReader struct {
	resp      *http.Response
	scanner   *bufio.Scanner
	provider  string
	done      bool
	buffer    strings.Builder
	responses []geminiStreamResponse
	index     int
}

func (r *geminiStreamReader) Next() (*StreamChunk, error) {
	if r.done {
		return &StreamChunk{Done: true, Provider: r.provider}, nil
	}

	if r.index < len(r.responses) {
		return r.processResponse(r.responses[r.index])
	}

	if len(r.responses) == 0 {
		if err := r.readCompleteJSON(); err != nil {
			return nil, err
		}
	}

	if r.index < len(r.responses) {
		return r.processResponse(r.responses[r.index])
	}

	r.done = true
	return &StreamChunk{Done: true, Provider: r.provider}, nil
}

func (r *geminiStreamReader) readCompleteJSON() error {
	for r.scanner.Scan() {
		line := r.scanner.Text()
		r.buffer.WriteString(line)
	}

	if err := r.scanner.Err(); err != nil {
		return err
	}

	jsonStr := r.buffer.String()

	var responses []geminiStreamResponse
	if err := json.Unmarshal([]byte(jsonStr), &responses); err != nil {
		return fmt.Errorf("gemini: failed to parse stream response: %w", err)
	}

	r.responses = responses
	return nil
}

func (r *geminiStreamReader) processResponse(streamResp geminiStreamResponse) (*StreamChunk, error) {
	defer func() { r.index++ }()

	if len(streamResp.Candidates) > 0 {
		candidate := streamResp.Candidates[0]
		streamChunk := &StreamChunk{
			Provider: r.provider,
		}

		// Extract text content and thinking content
		for _, part := range candidate.Content.Parts {
			// Check if this is thinking content
			if part.Thought != nil && *part.Thought && part.Text != "" {
				streamChunk.Type = "reasoning"
				streamChunk.Reasoning = &ReasoningChunk{
					Content: part.Text,
					Summary: "Gemini thinking process",
				}
				return streamChunk, nil
			} else if part.Text != "" {
				// Regular content
				streamChunk.Type = "content"
				streamChunk.Content = part.Text
				return streamChunk, nil
			}

			// Handle tool calls
			if part.FunctionCall != nil {
				streamChunk.Type = "tool_call_delta"
				args, _ := json.Marshal(part.FunctionCall.Args)
				streamChunk.ToolCallDelta = &ToolCallDelta{
					ID:             fmt.Sprintf("call_%d", time.Now().UnixNano()),
					Type:           "function",
					FunctionName:   part.FunctionCall.Name,
					ArgumentsDelta: string(args),
				}
				return streamChunk, nil
			}
		}

		if candidate.FinishReason != "" {
			streamChunk.FinishReason = candidate.FinishReason
			streamChunk.Done = true
			r.done = true
			return streamChunk, nil
		}

		return r.Next()
	}

	return r.Next()
}

func (r *geminiStreamReader) Close() error {
	return r.resp.Body.Close()
}

// Gemini API request/response structures
type geminiRequest struct {
	Contents          []geminiContent         `json:"contents"`
	GenerationConfig  *geminiGenerationConfig `json:"generationConfig,omitempty"`
	Tools             []geminiTool            `json:"tools,omitempty"`
	ToolConfig        *geminiToolConfig       `json:"toolConfig,omitempty"` // Tool configuration for function calling
	SystemInstruction *geminiContent          `json:"systemInstruction,omitempty"`
}

type geminiContent struct {
	Role         string              `json:"role,omitempty"`
	Parts        []geminiPart        `json:"parts"`
	CacheControl *geminiCacheControl `json:"cache_control,omitempty"`
}

// geminiCacheControl represents Gemini's cache control structure
type geminiCacheControl struct {
	Type string `json:"type"`
}

type geminiPart struct {
	Text             string                  `json:"text,omitempty"`
	Thought          *bool                   `json:"thought,omitempty"`
	FunctionCall     *geminiFunctionCall     `json:"functionCall,omitempty"`
	FunctionResponse *geminiFunctionResponse `json:"functionResponse,omitempty"`
}

type geminiFunctionCall struct {
	Name string         `json:"name"`
	Args map[string]any `json:"args"`
}

type geminiFunctionResponse struct {
	Name     string         `json:"name"`
	Response map[string]any `json:"response"`
}

type geminiGenerationConfig struct {
	Temperature      *float64              `json:"temperature,omitempty"`
	MaxOutputTokens  *int                  `json:"maxOutputTokens,omitempty"`
	TopP             *float64              `json:"topP,omitempty"`
	TopK             *int                  `json:"topK,omitempty"`
	StopSequences    []string              `json:"stopSequences,omitempty"`
	ResponseMimeType string                `json:"responseMimeType,omitempty"`
	ResponseSchema   *geminiResponseSchema `json:"responseSchema,omitempty"`
	ThinkingConfig   *geminiThinkingConfig `json:"thinkingConfig,omitempty"`
}

type geminiResponseSchema struct {
	Type        string         `json:"type"`
	Description string         `json:"description,omitempty"`
	Properties  map[string]any `json:"properties,omitempty"`
	Required    []string       `json:"required,omitempty"`
}

type geminiThinkingConfig struct {
	ThinkingBudget  *int  `json:"thinkingBudget,omitempty"`
	IncludeThoughts *bool `json:"includeThoughts,omitempty"`
}

type geminiTool struct {
	FunctionDeclarations []geminiFunctionDeclaration `json:"functionDeclarations"`
}

type geminiFunctionDeclaration struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters"`
}

// geminiToolConfig configures how the model uses the provided tools
type geminiToolConfig struct {
	FunctionCallingConfig *geminiFunctionCallingConfig `json:"functionCallingConfig,omitempty"`
}

// geminiFunctionCallingConfig controls function calling behavior
type geminiFunctionCallingConfig struct {
	Mode                 string   `json:"mode,omitempty"`                 // AUTO, ANY, NONE
	AllowedFunctionNames []string `json:"allowedFunctionNames,omitempty"` // Optional: restrict to specific functions
}

type geminiResponse struct {
	Candidates    []geminiCandidate    `json:"candidates"`
	UsageMetadata *geminiUsageMetadata `json:"usageMetadata,omitempty"`
}

type geminiCandidate struct {
	Content      geminiContent `json:"content"`
	FinishReason string        `json:"finishReason,omitempty"`
	Index        int           `json:"index,omitempty"`
}

type geminiUsageMetadata struct {
	PromptTokenCount        int `json:"promptTokenCount"`
	CandidatesTokenCount    int `json:"candidatesTokenCount"`
	ThoughtsTokenCount      int `json:"thoughtsTokenCount,omitempty"`
	TotalTokenCount         int `json:"totalTokenCount"`
	CachedContentTokenCount int `json:"cachedContentTokenCount,omitempty"`
}

// Streaming structures
type geminiStreamResponse struct {
	Candidates    []geminiCandidate    `json:"candidates,omitempty"`
	UsageMetadata *geminiUsageMetadata `json:"usageMetadata,omitempty"`
}
