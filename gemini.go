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
	"time"
)

// GeminiProvider implements the Provider interface for Google Gemini
type GeminiProvider struct {
	*BaseProvider
}

// NewGeminiProvider creates a new Gemini provider
func NewGeminiProvider(config ProviderConfig) Provider {
	if config.BaseURL == "" {
		config.BaseURL = "https://generativelanguage.googleapis.com"
	}
	return &GeminiProvider{
		BaseProvider: NewBaseProvider("gemini", config),
	}
}

// Models returns the list of supported models
func (p *GeminiProvider) Models() []ModelInfo {
	return []ModelInfo{
		{
			ID: "gemini-2.5-pro", Provider: "gemini", Name: "Gemini 2.5 Pro", MaxTokens: 2000000,
			Capabilities: []ModelCapability{CapabilityChat, CapabilityVision, CapabilityCode, CapabilityFunctionCall},
		},
		{
			ID: "gemini-2.5-flash", Provider: "gemini", Name: "Gemini 2.5 Flash", MaxTokens: 1000000,
			Capabilities: []ModelCapability{CapabilityChat, CapabilityVision, CapabilityFunctionCall},
		},
	}
}

// Gemini API request/response structures
type geminiRequest struct {
	Contents          []geminiContent         `json:"contents"`
	GenerationConfig  *geminiGenerationConfig `json:"generationConfig,omitempty"`
	Tools             []geminiTool            `json:"tools,omitempty"`
	SystemInstruction *geminiContent          `json:"systemInstruction,omitempty"`
}

type geminiContent struct {
	Role  string       `json:"role,omitempty"`
	Parts []geminiPart `json:"parts"`
}

type geminiPart struct {
	Text             string                  `json:"text,omitempty"`
	FunctionCall     *geminiFunctionCall     `json:"functionCall,omitempty"`
	FunctionResponse *geminiFunctionResponse `json:"functionResponse,omitempty"`
}

type geminiFunctionCall struct {
	Name string                 `json:"name"`
	Args map[string]interface{} `json:"args"`
}

type geminiFunctionResponse struct {
	Name     string                 `json:"name"`
	Response map[string]interface{} `json:"response"`
}

type geminiGenerationConfig struct {
	Temperature     *float64 `json:"temperature,omitempty"`
	MaxOutputTokens *int     `json:"maxOutputTokens,omitempty"`
	TopP            *float64 `json:"topP,omitempty"`
	TopK            *int     `json:"topK,omitempty"`
	StopSequences   []string `json:"stopSequences,omitempty"`
}

type geminiTool struct {
	FunctionDeclarations []geminiFunctionDeclaration `json:"functionDeclarations"`
}

type geminiFunctionDeclaration struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Parameters  map[string]interface{} `json:"parameters"`
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
	PromptTokenCount     int `json:"promptTokenCount"`
	CandidatesTokenCount int `json:"candidatesTokenCount"`
	TotalTokenCount      int `json:"totalTokenCount"`
}

// Streaming structures
type geminiStreamResponse struct {
	Candidates    []geminiCandidate    `json:"candidates,omitempty"`
	UsageMetadata *geminiUsageMetadata `json:"usageMetadata,omitempty"`
}

func (p *GeminiProvider) Complete(ctx context.Context, req *Request) (*Response, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}

	modelName := req.Model

	// Build Gemini request
	geminiReq := geminiRequest{
		Contents: make([]geminiContent, 0),
	}

	// Set generation configuration
	if req.Temperature != nil || req.MaxTokens != nil {
		geminiReq.GenerationConfig = &geminiGenerationConfig{
			Temperature:     req.Temperature,
			MaxOutputTokens: req.MaxTokens,
		}
	}

	// Handle system messages and regular messages
	var systemMessage string
	for _, msg := range req.Messages {
		if msg.Role == "system" {
			// Gemini uses systemInstruction to handle system messages
			systemMessage = msg.Content
			continue
		}

		content := geminiContent{
			Role:  p.convertRole(msg.Role),
			Parts: []geminiPart{},
		}

		// Add text content
		if msg.Content != "" {
			content.Parts = append(content.Parts, geminiPart{Text: msg.Content})
		}

		// Handle tool calls
		if len(msg.ToolCalls) > 0 {
			for _, toolCall := range msg.ToolCalls {
				var args map[string]interface{}
				if err := json.Unmarshal([]byte(toolCall.Function.Arguments), &args); err != nil {
					// If parsing fails, try direct use
					args = map[string]interface{}{"input": toolCall.Function.Arguments}
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
			var response map[string]interface{}
			if err := json.Unmarshal([]byte(msg.Content), &response); err != nil {
				// If parsing fails, use raw text
				response = map[string]interface{}{"result": msg.Content}
			}

			// Get tool name from Extra field, or use default if not available
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

	// Set system instruction
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
			var params map[string]interface{}
			if p, ok := t.Function.Parameters.(map[string]interface{}); ok {
				params = p
			} else {
				if jsonBytes, err := json.Marshal(t.Function.Parameters); err == nil {
					json.Unmarshal(jsonBytes, &params)
				}
			}

			tool.FunctionDeclarations[i] = geminiFunctionDeclaration{
				Name:        t.Function.Name,
				Description: t.Function.Description,
				Parameters:  params,
			}
		}
		geminiReq.Tools = []geminiTool{tool}
	}

	// Send request
	body, err := json.Marshal(geminiReq)
	if err != nil {
		return nil, fmt.Errorf("gemini: marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/v1beta/models/%s:generateContent?key=%s",
		p.config.BaseURL, modelName, p.config.APIKey)

	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("gemini: create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("gemini: request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("gemini: API error %d: %s", resp.StatusCode, string(body))
	}

	var geminiResp geminiResponse
	if err := json.NewDecoder(resp.Body).Decode(&geminiResp); err != nil {
		return nil, fmt.Errorf("gemini: decode response: %w", err)
	}

	// Build response
	response := &Response{
		Model:    modelName,
		Provider: "gemini",
	}

	if geminiResp.UsageMetadata != nil {
		response.Usage = Usage{
			PromptTokens:     geminiResp.UsageMetadata.PromptTokenCount,
			CompletionTokens: geminiResp.UsageMetadata.CandidatesTokenCount,
			TotalTokens:      geminiResp.UsageMetadata.TotalTokenCount,
		}
	}

	if len(geminiResp.Candidates) > 0 {
		candidate := geminiResp.Candidates[0]
		response.FinishReason = candidate.FinishReason

		// Extract content and tool calls
		for _, part := range candidate.Content.Parts {
			if part.Text != "" {
				response.Content += part.Text
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
	}

	return response, nil
}

func (p *GeminiProvider) Stream(ctx context.Context, req *Request) (StreamReader, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}

	modelName := req.Model

	// Build request (similar to Complete method)
	geminiReq := geminiRequest{
		Contents: make([]geminiContent, 0),
	}

	// Set generation configuration
	if req.Temperature != nil || req.MaxTokens != nil {
		geminiReq.GenerationConfig = &geminiGenerationConfig{
			Temperature:     req.Temperature,
			MaxOutputTokens: req.MaxTokens,
		}
	}

	// Handle messages (simplified version, mainly process text)
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
			geminiReq.Contents = append(geminiReq.Contents, content)
		}
	}

	// Set system instruction
	if systemMessage != "" {
		geminiReq.SystemInstruction = &geminiContent{
			Parts: []geminiPart{{Text: systemMessage}},
		}
	}

	// Send streaming request
	body, err := json.Marshal(geminiReq)
	if err != nil {
		return nil, fmt.Errorf("gemini: marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/v1beta/models/%s:streamGenerateContent?key=%s",
		p.config.BaseURL, modelName, p.config.APIKey)

	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("gemini: create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("gemini: request failed: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
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
	return fmt.Sprintf("call_%d", time.Now().UnixNano())
}

// geminiStreamReader implements StreamReader for Gemini
type geminiStreamReader struct {
	resp     *http.Response
	scanner  *bufio.Scanner
	provider string
	err      error
	done     bool
}

func (r *geminiStreamReader) Read() (*StreamChunk, error) {
	if r.done {
		return &StreamChunk{Done: true, Provider: r.provider}, nil
	}

	for r.scanner.Scan() {
		line := r.scanner.Text()
		line = strings.TrimSpace(line)

		// Gemini streaming response format might be different, handle JSON lines here
		if line == "" || line == "data: [DONE]" {
			continue
		}

		// Remove possible "data: " prefix
		if strings.HasPrefix(line, "data: ") {
			line = strings.TrimPrefix(line, "data: ")
		}

		// Skip empty lines and non-JSON lines
		if line == "" || (!strings.HasPrefix(line, "{") && !strings.HasPrefix(line, "[")) {
			continue
		}

		var streamResp geminiStreamResponse
		if err := json.Unmarshal([]byte(line), &streamResp); err != nil {
			// If parsing fails, continue to next line
			continue
		}

		if len(streamResp.Candidates) > 0 {
			candidate := streamResp.Candidates[0]
			streamChunk := &StreamChunk{
				Provider: r.provider,
			}

			// Extract text content
			for _, part := range candidate.Content.Parts {
				if part.Text != "" {
					streamChunk.Type = ChunkTypeContent
					streamChunk.Content = part.Text
					return streamChunk, nil
				}

				// Handle tool calls
				if part.FunctionCall != nil {
					streamChunk.Type = ChunkTypeToolCall
					args, _ := json.Marshal(part.FunctionCall.Args)
					streamChunk.ToolCalls = []ToolCall{{
						ID:   fmt.Sprintf("call_%d", time.Now().UnixNano()),
						Type: "function",
						Function: FunctionCall{
							Name:      part.FunctionCall.Name,
							Arguments: string(args),
						},
					}}
					return streamChunk, nil
				}
			}

			if candidate.FinishReason != "" {
				streamChunk.FinishReason = candidate.FinishReason
				streamChunk.Done = true
				r.done = true
				return streamChunk, nil
			}
		}

		// Handle usage statistics
		if streamResp.UsageMetadata != nil {
			return &StreamChunk{
				Type:     ChunkTypeUsage,
				Provider: r.provider,
				Usage: &Usage{
					PromptTokens:     streamResp.UsageMetadata.PromptTokenCount,
					CompletionTokens: streamResp.UsageMetadata.CandidatesTokenCount,
					TotalTokens:      streamResp.UsageMetadata.TotalTokenCount,
				},
			}, nil
		}
	}

	if err := r.scanner.Err(); err != nil {
		r.err = err
		return nil, err
	}

	r.done = true
	return &StreamChunk{Done: true, Provider: r.provider}, nil
}

func (r *geminiStreamReader) Close() error {
	return r.resp.Body.Close()
}

func (r *geminiStreamReader) Err() error {
	return r.err
}

func init() {
	RegisterProvider("gemini", NewGeminiProvider)
}
