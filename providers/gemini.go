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

func init() {
	RegisterBuiltin("gemini", func(cfg ProviderConfig) Provider {
		return NewGemini(cfg)
	}, "https://generativelanguage.googleapis.com")
}

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

func (p *GeminiProvider) validateExtra(req *Request) error {
	if err := p.BaseProvider.ValidateExtra(req.Extra, []string{"tool_name"}); err != nil {
		return err
	}
	if req.Extra != nil {
		if value, ok := req.Extra["tool_name"]; ok {
			if _, ok := value.(string); !ok {
				return fmt.Errorf("gemini: extra parameter 'tool_name' must be a string")
			}
		}
	}
	return nil
}

func (p *GeminiProvider) Chat(ctx context.Context, req *Request) (*Response, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}
	if err := p.validateExtra(req); err != nil {
		return nil, err
	}

	if err := p.BaseProvider.ValidateRequest(req); err != nil {
		return nil, err
	}

	modelName := req.Model

	geminiReq := geminiRequest{
		Contents: make([]geminiContent, 0),
	}

	generationConfig, err := p.buildGenerationConfig(req)
	if err != nil {
		return nil, err
	}
	if generationConfig != nil {
		geminiReq.GenerationConfig = generationConfig
	}

	contents, systemMessage, err := p.buildContents(req)
	if err != nil {
		return nil, err
	}
	geminiReq.Contents = contents

	if systemMessage != "" {
		geminiReq.SystemInstruction = &geminiContent{
			Parts: []geminiPart{{Text: systemMessage}},
		}
	}

	// Convert tool definitions
	if len(req.Tools) > 0 {
		warnIgnoredStrictTools(req, "gemini", req.Tools)
		tools, err := p.convertTools(req.Tools)
		if err != nil {
			return nil, err
		}
		geminiReq.Tools = tools

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

	p.NotifyPayload(req, body)

	url := fmt.Sprintf("%s/v1beta/models/%s:generateContent?key=%s",
		p.Config().BaseURL, modelName, p.ResolveAPIKey(req))

	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("gemini: create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := p.HTTPClient().Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, NewHTTPError("gemini", resp.StatusCode, string(body))
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
		response.FinishReason = NormalizeFinishReason(candidate.FinishReason)

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
				id := part.FunctionCall.ID
				if id == "" {
					id = p.generateToolCallID()
				}
				response.ToolCalls = append(response.ToolCalls, ToolCall{
					ID:   id,
					Type: "function",
					Function: FunctionCall{
						Name:      part.FunctionCall.Name,
						Arguments: string(args),
					},
				})
			}
		}

		// Populate reasoning data if thinking content exists
		if !isThinkingDisabled(req) && thinkingContent != "" {
			response.ReasoningContent = thinkingContent
		}
	}

	return response, nil
}

func (p *GeminiProvider) Stream(ctx context.Context, req *Request) (StreamReader, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}
	if err := p.validateExtra(req); err != nil {
		return nil, err
	}
	if err := p.BaseProvider.ValidateRequest(req); err != nil {
		return nil, err
	}

	geminiReq := geminiRequest{
		Contents: make([]geminiContent, 0),
	}

	includeReasoning := !isThinkingDisabled(req)

	generationConfig, err := p.buildGenerationConfig(req)
	if err != nil {
		return nil, err
	}
	if generationConfig != nil {
		geminiReq.GenerationConfig = generationConfig
	}

	contents, systemMessage, err := p.buildContents(req)
	if err != nil {
		return nil, err
	}
	geminiReq.Contents = contents

	if systemMessage != "" {
		geminiReq.SystemInstruction = &geminiContent{
			Parts: []geminiPart{{Text: systemMessage}},
		}
	}

	if len(req.Tools) > 0 {
		warnIgnoredStrictTools(req, "gemini", req.Tools)
		tools, err := p.convertTools(req.Tools)
		if err != nil {
			return nil, err
		}
		geminiReq.Tools = tools
		if req.ToolChoice != nil {
			geminiReq.ToolConfig = p.convertToolChoice(req.ToolChoice)
		}
	}

	body, err := json.Marshal(geminiReq)
	if err != nil {
		return nil, fmt.Errorf("gemini: marshal request: %w", err)
	}

	p.NotifyPayload(req, body)

	url := fmt.Sprintf("%s/v1beta/models/%s:streamGenerateContent?alt=sse&key=%s",
		p.Config().BaseURL, req.Model, p.ResolveAPIKey(req))

	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("gemini: create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/event-stream")

	resp, err := p.HTTPClient().Do(httpReq)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, NewHTTPError("gemini", resp.StatusCode, string(body))
	}

	return &geminiStreamReader{
		resp:             resp,
		scanner:          newStreamScanner(resp.Body),
		provider:         "gemini",
		model:            req.Model,
		includeReasoning: includeReasoning,
		toolCallIndexes:  make(map[string]int),
	}, nil
}

// ListModels returns available models for Gemini.
func (p *GeminiProvider) ListModels(ctx context.Context) ([]ModelInfo, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}

	baseURL := strings.TrimSuffix(p.Config().BaseURL, "/")
	if baseURL == "" {
		baseURL = "https://generativelanguage.googleapis.com"
	}
	url := fmt.Sprintf("%s/v1beta/models?key=%s", baseURL, p.ResolveAPIKey(nil))
	httpReq, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("gemini: create models request: %w", err)
	}

	resp, err := p.HTTPClient().Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, NewHTTPError("gemini", resp.StatusCode, string(body))
	}

	var payload geminiModelList
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return nil, fmt.Errorf("gemini: decode models response: %w", err)
	}

	models := make([]ModelInfo, 0, len(payload.Models))
	for _, item := range payload.Models {
		id := strings.TrimPrefix(item.Name, "models/")
		name := item.DisplayName
		if name == "" {
			name = id
		}
		models = append(models, ModelInfo{
			ID:               id,
			Name:             name,
			Provider:         "gemini",
			Description:      item.Description,
			InputTokenLimit:  item.InputTokenLimit,
			OutputTokenLimit: item.OutputTokenLimit,
		})
	}

	return models, nil
}

// buildContents translates canonical messages into Gemini's Contents array
// plus a system-instruction string. Used by both Chat and Stream so the two
// code paths cannot drift on tool-call / tool-response handling.
//
// Gemini uses the same request body schema for generateContent and
// streamGenerateContent — see https://ai.google.dev/api/rest/v1beta/models/streamGenerateContent
func (p *GeminiProvider) buildContents(req *Request) ([]geminiContent, string, error) {
	prepared, err := PrepareMessages(req.Messages)
	if err != nil {
		return nil, "", fmt.Errorf("gemini: %w", err)
	}

	contents := make([]geminiContent, 0, len(prepared))
	// callNames binds a tool call id to its function name so tool-result
	// messages can emit the correct functionResponse.name (Gemini docs
	// recommend echoing both id and name). We learn the mapping from the
	// assistant tool_calls we emit earlier in this same message stream.
	callNames := make(map[string]string)
	var systemMessage string

	for _, msg := range prepared {
		if msg.Role == "system" {
			systemMessage = msg.Content
			continue
		}

		content := geminiContent{
			Role:  p.convertRole(msg.Role),
			Parts: []geminiPart{},
		}

		// Multi-content (text + images) take precedence over Content — but not
		// for tool results, which must emit a functionResponse part instead.
		if len(msg.Contents) > 0 && msg.ToolCallID == "" {
			for _, c := range msg.Contents {
				switch c.Type {
				case "text", "", "input_text":
					if c.Text != "" {
						content.Parts = append(content.Parts, geminiPart{Text: c.Text})
					}
				case "image_url", "input_image":
					if c.ImageURL == nil || c.ImageURL.URL == "" {
						continue
					}
					if mime, data, ok := parseDataURL(c.ImageURL.URL); ok {
						content.Parts = append(content.Parts, geminiPart{
							InlineData: &geminiInlineData{MimeType: mime, Data: data},
						})
					} else {
						content.Parts = append(content.Parts, geminiPart{
							FileData: &geminiFileData{MimeType: inferMimeType(c.ImageURL.URL), FileURI: c.ImageURL.URL},
						})
					}
				}
			}
		} else if msg.Content != "" && msg.ToolCallID == "" {
			content.Parts = append(content.Parts, geminiPart{Text: msg.Content})
		}

		// Assistant tool_calls → functionCall parts (id echoed, name required)
		if len(msg.ToolCalls) > 0 {
			for _, toolCall := range msg.ToolCalls {
				var args map[string]any
				if err := json.Unmarshal([]byte(toolCall.Function.Arguments), &args); err != nil {
					return nil, "", fmt.Errorf("gemini: invalid tool call arguments for '%s': %w", toolCall.Function.Name, err)
				}
				callNames[toolCall.ID] = toolCall.Function.Name
				content.Parts = append(content.Parts, geminiPart{
					FunctionCall: &geminiFunctionCall{
						ID:   toolCall.ID,
						Name: toolCall.Function.Name,
						Args: args,
					},
				})
			}
		}

		// Tool role → functionResponse part: echo id + name for correlation.
		if msg.ToolCallID != "" {
			response := decodeGeminiToolResponse(msg)

			toolName := callNames[msg.ToolCallID]
			if toolName == "" {
				// Fallback only when id→name is unknown (e.g. tool result sent
				// without its matching assistant turn in the same request).
				if req.Extra != nil {
					if name, ok := req.Extra["tool_name"].(string); ok && name != "" {
						toolName = name
					}
				}
				if toolName == "" {
					toolName = "function_result"
				}
			}

			content.Parts = []geminiPart{{
				FunctionResponse: &geminiFunctionResponse{
					ID:       msg.ToolCallID,
					Name:     toolName,
					Response: response,
				},
			}}
		}

		if len(content.Parts) > 0 {
			contents = append(contents, content)
		}
	}

	return contents, systemMessage, nil
}

// convertRole converts standard roles to Gemini roles
// Gemini only supports "user" and "model" roles
// Tool/function responses should use "user" role with functionResponse part
func (p *GeminiProvider) convertRole(role string) string {
	switch role {
	case "assistant":
		return "model"
	case "tool":
		// Tool responses use "user" role in Gemini (with functionResponse part)
		return "user"
	default:
		return "user"
	}
}

// decodeGeminiToolResponse converts a canonical tool-result message into the
// map form Gemini's functionResponse expects. Gemini requires a JSON object;
// when Content is not valid JSON (e.g. a plain string from OpenAI-style tool
// results, or the synthetic error produced by PrepareMessages for orphaned
// tool calls) we wrap it under a single key so the request still validates.
func decodeGeminiToolResponse(msg Message) map[string]any {
	if msg.Content == "" {
		if msg.IsError {
			return map[string]any{"error": "tool execution failed"}
		}
		return map[string]any{}
	}
	var obj map[string]any
	if err := json.Unmarshal([]byte(msg.Content), &obj); err == nil && obj != nil {
		return obj
	}
	key := "result"
	if msg.IsError {
		key = "error"
	}
	return map[string]any{key: msg.Content}
}

// generateToolCallID generates a unique tool call ID
func (p *GeminiProvider) generateToolCallID() string {
	timestamp := time.Now().Unix()
	counter := geminiToolCallCounter.Add(1)
	return fmt.Sprintf("call_%d_%d", timestamp, counter)
}

func (p *GeminiProvider) buildGenerationConfig(req *Request) (*geminiGenerationConfig, error) {
	thinking := normalizeThinking(req)
	if thinking != nil && thinking.Type != "enabled" && thinking.Type != "disabled" {
		return nil, fmt.Errorf("gemini: thinking type must be enabled or disabled")
	}
	if req.Temperature == nil && req.MaxTokens == nil && req.TopP == nil && req.ResponseFormat == nil && len(req.Stop) == 0 && thinking == nil {
		return nil, nil
	}

	cfg := &geminiGenerationConfig{
		Temperature:     req.Temperature,
		MaxOutputTokens: req.MaxTokens,
		TopP:            req.TopP,
		StopSequences:   req.Stop,
	}
	if thinking != nil {
		includeThoughts := thinking.Type != "disabled"
		cfg.ThinkingConfig = &geminiThinkingConfig{
			ThinkingLevel:   thinking.Level,
			IncludeThoughts: &includeThoughts,
			ThinkingBudget:  thinking.BudgetTokens,
		}
	}
	if err := p.applyResponseFormat(cfg, req.ResponseFormat); err != nil {
		return nil, err
	}
	return cfg, nil
}

func (p *GeminiProvider) applyResponseFormat(cfg *geminiGenerationConfig, format *ResponseFormat) error {
	if format == nil {
		return nil
	}

	switch format.Type {
	case "json_object":
		cfg.ResponseMimeType = "application/json"
	case "json_schema":
		if format.JSONSchema == nil || format.JSONSchema.Schema == nil {
			return fmt.Errorf("gemini: response_format json_schema requires json_schema.schema")
		}
		cfg.ResponseMimeType = "application/json"
		cfg.ResponseJsonSchema = format.JSONSchema.Schema
	default:
		return fmt.Errorf("gemini: unsupported response_format type %q", format.Type)
	}
	return nil
}

func (p *GeminiProvider) convertTools(tools []Tool) ([]geminiTool, error) {
	tool := geminiTool{
		FunctionDeclarations: make([]geminiFunctionDeclaration, 0, len(tools)),
	}
	for _, t := range tools {
		params, err := geminiToolParameters(t)
		if err != nil {
			return nil, err
		}
		tool.FunctionDeclarations = append(tool.FunctionDeclarations, geminiFunctionDeclaration{
			Name:        t.Function.Name,
			Description: t.Function.Description,
			Parameters:  params,
		})
	}
	return []geminiTool{tool}, nil
}

func geminiToolParameters(tool Tool) (map[string]any, error) {
	if tool.Function.Parameters == nil {
		return nil, nil
	}
	if params, ok := tool.Function.Parameters.(map[string]any); ok {
		return params, nil
	}
	jsonBytes, err := json.Marshal(tool.Function.Parameters)
	if err != nil {
		return nil, fmt.Errorf("gemini: failed to marshal tool parameters for %q: %w", tool.Function.Name, err)
	}
	var params map[string]any
	if err := json.Unmarshal(jsonBytes, &params); err != nil {
		return nil, fmt.Errorf("gemini: invalid tool parameters for %q: %w", tool.Function.Name, err)
	}
	return params, nil
}

func warnIgnoredStrictTools(req *Request, provider string, tools []Tool) {
	for _, tool := range tools {
		if tool.Type == "function" && tool.Function.Strict != nil && *tool.Function.Strict {
			notifyWarning(req, provider, "tool %q requested strict=true, but strict tool calling is not supported by this provider; strict was omitted", tool.Function.Name)
		}
	}
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

// geminiStreamReader implements streaming for Gemini
// Gemini uses newline-delimited JSON format (each line is a complete JSON object)
type geminiStreamReader struct {
	resp             *http.Response
	scanner          *bufio.Scanner
	provider         string
	model            string
	includeReasoning bool
	done             bool
	pending          string
	queue            []geminiStreamResponse
	usage            *Usage // Accumulate usage from streaming response
	toolCallIndex    int    // next index for unseen tool calls
	toolCallIndexes  map[string]int
}

func (r *geminiStreamReader) Next() (*StreamChunk, error) {
	if r.done {
		return &StreamChunk{Done: true, Provider: r.provider, Model: r.model, Usage: r.usage}, nil
	}
	for len(r.queue) > 0 {
		next := r.queue[0]
		r.queue = r.queue[1:]
		if chunk, err := r.processResponse(next); chunk != nil || err != nil {
			return chunk, err
		}
	}

	// Read next line from the stream
	for r.scanner.Scan() {
		line := strings.TrimSpace(r.scanner.Text())
		if line == "" {
			continue
		}
		if data, found := strings.CutPrefix(line, "data:"); found {
			line = strings.TrimSpace(data)
			if line == "" {
				continue
			}
		}
		if line == "[DONE]" {
			r.done = true
			return &StreamChunk{Done: true, Provider: r.provider, Model: r.model, Usage: r.usage}, nil
		}

		r.pending += line

		var streamResp geminiStreamResponse
		objErr := json.Unmarshal([]byte(r.pending), &streamResp)
		if objErr == nil {
			r.pending = ""
			if chunk, err := r.processResponse(streamResp); chunk != nil || err != nil {
				return chunk, err
			}
			continue
		}

		var streamArray []geminiStreamResponse
		arrErr := json.Unmarshal([]byte(r.pending), &streamArray)
		if arrErr == nil {
			r.pending = ""
			if len(streamArray) == 0 {
				continue
			}
			r.queue = append(r.queue, streamArray[1:]...)
			if chunk, err := r.processResponse(streamArray[0]); chunk != nil || err != nil {
				return chunk, err
			}
			continue
		}

		if isIncompleteJSON(objErr) || isIncompleteJSON(arrErr) {
			continue
		}
		return nil, fmt.Errorf("gemini: failed to parse stream chunk: %w", objErr)
	}

	if err := r.scanner.Err(); err != nil {
		return nil, NewNetworkError(r.provider, "stream read error: "+err.Error(), err)
	}
	if strings.TrimSpace(r.pending) != "" {
		return nil, fmt.Errorf("gemini: incomplete JSON stream")
	}

	r.done = true
	return &StreamChunk{Done: true, Provider: r.provider, Model: r.model, Usage: r.usage}, nil
}

func (r *geminiStreamReader) processResponse(streamResp geminiStreamResponse) (*StreamChunk, error) {
	// Update usage metadata if present (usually in the final chunk)
	if streamResp.UsageMetadata != nil {
		r.usage = &Usage{
			PromptTokens:         streamResp.UsageMetadata.PromptTokenCount,
			CompletionTokens:     streamResp.UsageMetadata.CandidatesTokenCount,
			ReasoningTokens:      streamResp.UsageMetadata.ThoughtsTokenCount,
			TotalTokens:          streamResp.UsageMetadata.TotalTokenCount,
			CacheReadInputTokens: streamResp.UsageMetadata.CachedContentTokenCount,
		}
	}

	if len(streamResp.Candidates) > 0 {
		candidate := streamResp.Candidates[0]
		streamChunk := &StreamChunk{
			Provider: r.provider,
			Model:    r.model,
		}

		// Extract text content and thinking content
		for _, part := range candidate.Content.Parts {
			// Check if this is thinking content
			if part.Thought != nil && *part.Thought && part.Text != "" {
				if !r.includeReasoning {
					continue
				}
				streamChunk.Type = "reasoning"
				streamChunk.ReasoningContent = part.Text
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
				id := part.FunctionCall.ID
				index := r.toolCallIndex
				if id != "" {
					if existing, ok := r.toolCallIndexes[id]; ok {
						index = existing
					} else {
						r.toolCallIndexes[id] = index
						r.toolCallIndex++
					}
				} else {
					id = fmt.Sprintf("call_%d", time.Now().UnixNano())
					r.toolCallIndex++
				}
				streamChunk.ToolCallDelta = &ToolCallDelta{
					Index:          index,
					ID:             id,
					Type:           "function",
					FunctionName:   part.FunctionCall.Name,
					ArgumentsDelta: string(args),
				}
				return streamChunk, nil
			}
		}

		if candidate.FinishReason != "" {
			streamChunk.FinishReason = NormalizeFinishReason(candidate.FinishReason)
			streamChunk.Done = true
			streamChunk.Usage = r.usage
			r.done = true
			return streamChunk, nil
		}

		return nil, nil // no emittable content, let caller continue
	}

	return nil, nil // no candidates, let caller continue
}

func (r *geminiStreamReader) Close() error {
	return r.resp.Body.Close()
}

func isIncompleteJSON(err error) bool {
	if err == nil {
		return false
	}
	msg := err.Error()
	return strings.Contains(msg, "unexpected end of JSON input") || strings.Contains(msg, "unexpected EOF")
}

func newStreamScanner(r io.Reader) *bufio.Scanner {
	scanner := bufio.NewScanner(r)
	buf := make([]byte, 0, 64*1024)
	scanner.Buffer(buf, 1024*1024)
	return scanner
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
	Role  string       `json:"role,omitempty"`
	Parts []geminiPart `json:"parts"`
}

type geminiPart struct {
	Text             string                  `json:"text,omitempty"`
	Thought          *bool                   `json:"thought,omitempty"`
	InlineData       *geminiInlineData       `json:"inlineData,omitempty"`
	FileData         *geminiFileData         `json:"fileData,omitempty"`
	FunctionCall     *geminiFunctionCall     `json:"functionCall,omitempty"`
	FunctionResponse *geminiFunctionResponse `json:"functionResponse,omitempty"`
}

type geminiInlineData struct {
	MimeType string `json:"mimeType"`
	Data     string `json:"data"`
}

type geminiFileData struct {
	MimeType string `json:"mimeType"`
	FileURI  string `json:"fileUri"`
}

// geminiFunctionCall carries the model's tool invocation. Gemini 3+ emits a
// unique per-call id; earlier models omit it. See:
// https://ai.google.dev/gemini-api/docs/function-calling
type geminiFunctionCall struct {
	ID   string         `json:"id,omitempty"`
	Name string         `json:"name"`
	Args map[string]any `json:"args"`
}

// geminiFunctionResponse returns a tool result to the model. When the original
// functionCall carried an id, the response must echo it so multi-tool parallel
// calls can be correlated.
type geminiFunctionResponse struct {
	ID       string         `json:"id,omitempty"`
	Name     string         `json:"name"`
	Response map[string]any `json:"response"`
}

type geminiGenerationConfig struct {
	Temperature        *float64              `json:"temperature,omitempty"`
	MaxOutputTokens    *int                  `json:"maxOutputTokens,omitempty"`
	TopP               *float64              `json:"topP,omitempty"`
	TopK               *int                  `json:"topK,omitempty"`
	StopSequences      []string              `json:"stopSequences,omitempty"`
	ResponseMimeType   string                `json:"responseMimeType,omitempty"`
	ResponseSchema     *geminiResponseSchema `json:"responseSchema,omitempty"`
	ResponseJsonSchema any                   `json:"responseJsonSchema,omitempty"`
	ThinkingConfig     *geminiThinkingConfig `json:"thinkingConfig,omitempty"`
}

type geminiResponseSchema struct {
	Type        string         `json:"type"`
	Description string         `json:"description,omitempty"`
	Properties  map[string]any `json:"properties,omitempty"`
	Required    []string       `json:"required,omitempty"`
}

type geminiThinkingConfig struct {
	ThinkingLevel   string `json:"thinkingLevel,omitempty"`   // For Gemini 3: "minimal", "low", "medium", "high"
	ThinkingBudget  *int   `json:"thinkingBudget,omitempty"`  // For Gemini 2.5: token budget
	IncludeThoughts *bool  `json:"includeThoughts,omitempty"` // Enable thought summaries
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

type geminiModelList struct {
	Models []geminiModelInfo `json:"models"`
}

type geminiModelInfo struct {
	Name             string `json:"name"`
	DisplayName      string `json:"displayName,omitempty"`
	Description      string `json:"description,omitempty"`
	InputTokenLimit  int    `json:"inputTokenLimit,omitempty"`
	OutputTokenLimit int    `json:"outputTokenLimit,omitempty"`
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
