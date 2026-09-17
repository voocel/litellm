package openai

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"math"
	"strings"

	"github.com/voocel/litellm"
)

func (p *Provider) buildRequest(req *litellm.Request, stream bool) (*chatRequest, error) {
	if err := validateSampling(req.Temperature, req.TopP); err != nil {
		return nil, err
	}
	choice, err := convertToolChoice(req.ToolChoice)
	if err != nil {
		return nil, err
	}
	out := &chatRequest{
		Model:       req.Model,
		Stream:      stream,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		Stop:        append([]string(nil), req.Stop...),
		ToolChoice:  choice,
	}
	if stream {
		out.StreamOptions = &streamOptions{IncludeUsage: true}
	}
	if err := req.Thinking.Validate(); err != nil {
		return nil, fmt.Errorf("openai: %w", err)
	}
	if req.Thinking != nil && req.Thinking.Mode != litellm.ThinkingUnspecified && !p.isReasoningModel(req.Model) {
		return nil, fmt.Errorf("openai: thinking is only supported for reasoning chat models")
	}
	if p.isReasoningModel(req.Model) {
		out.MaxCompletionTokens = req.MaxTokens
		if req.Thinking != nil && req.Thinking.Mode == litellm.ThinkingDisabled {
			out.ReasoningEffort = "none"
		}
		if req.Thinking != nil && req.Thinking.Mode == litellm.ThinkingEnabled {
			effort := reasoningEffort(req.Thinking)
			if effort == "" {
				effort = "medium"
			}
			if !supportsOpenAIReasoningEffort(effort) {
				return nil, fmt.Errorf("openai: unsupported reasoning_effort %q; use %s", effort, strings.Join(openAIReasoningEfforts(), ", "))
			}
			out.ReasoningEffort = effort
		}
	} else {
		out.MaxTokens = req.MaxTokens
	}
	if req.ResponseFormat != nil {
		converted, err := convertResponseFormat(req.ResponseFormat)
		if err != nil {
			return nil, err
		}
		out.ResponseFormat = converted
	}
	if len(req.ProviderOptions) > 0 {
		options, err := req.ProviderOptions.Decode()
		if err != nil {
			return nil, err
		}
		if err := applyProviderOptions(out, options); err != nil {
			return nil, err
		}
	}
	if len(req.Tools) > 0 {
		tools, err := convertTools(req.Tools)
		if err != nil {
			return nil, err
		}
		out.Tools = tools
	}
	messages, err := convertMessages(req.Messages)
	if err != nil {
		return nil, err
	}
	out.Messages = messages
	return out, nil
}

func (p *Provider) isReasoningModel(model string) bool {
	model = openAIModelName(model)
	if strings.Contains(model, "chat") {
		return false
	}
	var major int
	_, err := fmt.Sscanf(model, "gpt-%d", &major)
	return err == nil && major >= 5
}

func reasoningEffort(thinking *litellm.Thinking) string {
	if thinking == nil {
		return ""
	}
	return thinking.Effort
}

func openAIModelName(model string) string {
	model = strings.ToLower(strings.TrimSpace(model))
	if _, after, ok := strings.Cut(model, "/"); ok {
		return after
	}
	return model
}

func openAIReasoningEfforts() []string {
	return []string{"low", "medium", "high", "xhigh", "max"}
}

func supportsOpenAIReasoningEffort(effort string) bool {
	for _, supported := range openAIReasoningEfforts() {
		if effort == supported {
			return true
		}
	}
	return false
}

func supportsOpenAIReasoningEffortValue(effort string) bool {
	if effort == "none" {
		return true
	}
	return supportsOpenAIReasoningEffort(effort)
}

func convertMessages(messages []litellm.Message) ([]chatMessage, error) {
	out := make([]chatMessage, 0, len(messages))
	for i, msg := range messages {
		converted := chatMessage{Role: string(msg.Role)}
		switch msg.Role {
		case litellm.RoleSystem, litellm.RoleUser, litellm.RoleAssistant:
			content, toolCalls, reasoningContent, err := convertMessageBlocks(msg.Blocks)
			if err != nil {
				return nil, fmt.Errorf("openai: messages[%d]: %w", i, err)
			}
			converted.Content = content
			converted.ToolCalls = toolCalls
			converted.ReasoningContent = reasoningContent
		case litellm.RoleTool:
			toolMessages, err := convertToolMessage(msg.Blocks)
			if err != nil {
				return nil, fmt.Errorf("openai: messages[%d]: %w", i, err)
			}
			out = append(out, toolMessages...)
			continue
		default:
			return nil, fmt.Errorf("openai: unsupported role %q", msg.Role)
		}
		out = append(out, converted)
	}
	return out, nil
}

func convertMessageBlocks(blocks []litellm.Block) (any, []toolCall, string, error) {
	parts := make([]contentPart, 0, len(blocks))
	var text strings.Builder
	var toolCalls []toolCall
	var reasoning strings.Builder
	for _, block := range blocks {
		switch b := block.(type) {
		case litellm.TextBlock:
			if b.Text == "" {
				if b.Cache != nil {
					return nil, nil, "", fmt.Errorf("openai: cache breakpoint requires a non-empty text block")
				}
				continue
			}
			breakpoint, err := convertPromptCacheBreakpoint(b.Cache)
			if err != nil {
				return nil, nil, "", err
			}
			parts = append(parts, contentPart{Type: "text", Text: b.Text, PromptCacheBreakpoint: breakpoint})
			if text.Len() > 0 {
				text.WriteString("\n")
			}
			text.WriteString(b.Text)
		case litellm.ImageBlock:
			url, err := imageURLValue(b)
			if err != nil {
				return nil, nil, "", err
			}
			breakpoint, err := convertPromptCacheBreakpoint(b.Cache)
			if err != nil {
				return nil, nil, "", err
			}
			parts = append(parts, contentPart{
				Type: "image_url",
				ImageURL: &imageURL{
					URL:    url,
					Detail: b.Detail,
				},
				PromptCacheBreakpoint: breakpoint,
			})
		case litellm.ToolUseBlock:
			if b.Cache != nil {
				return nil, nil, "", fmt.Errorf("OpenAI Chat does not support cache breakpoints on tool use blocks")
			}
			toolCalls = append(toolCalls, toolCall{
				ID:   b.ID,
				Type: "function",
				Function: toolCallFunc{
					Name:      b.Name,
					Arguments: string(b.Arguments),
				},
			})
		case litellm.ReasoningBlock:
			if b.Cache != nil {
				return nil, nil, "", fmt.Errorf("OpenAI Chat does not support cache breakpoints on reasoning blocks")
			}
			if b.Signature != "" || len(b.Redacted) > 0 || len(b.Extra) > 0 {
				return nil, nil, "", fmt.Errorf("OpenAI Chat does not accept signed, redacted, or provider-extra reasoning blocks in message history")
			}
			if b.Text != "" {
				if reasoning.Len() > 0 {
					reasoning.WriteString("\n")
				}
				reasoning.WriteString(b.Text)
			}
		default:
			return nil, nil, "", fmt.Errorf("unsupported block %T", block)
		}
	}
	reasoningText := reasoning.String()
	if len(parts) == 0 {
		return nil, toolCalls, reasoningText, nil
	}
	if len(parts) == 1 && parts[0].Type == "text" && parts[0].PromptCacheBreakpoint == nil {
		return text.String(), toolCalls, reasoningText, nil
	}
	return parts, toolCalls, reasoningText, nil
}

func convertToolMessage(blocks []litellm.Block) ([]chatMessage, error) {
	out := make([]chatMessage, 0, len(blocks))
	for _, block := range blocks {
		result, ok := block.(litellm.ToolResultBlock)
		if !ok {
			return nil, fmt.Errorf("tool role only supports ToolResultBlock, got %T", block)
		}
		text, err := toolResultText(result.Content)
		if err != nil {
			return nil, err
		}
		var content any = text
		if result.Cache != nil {
			breakpoint, err := convertPromptCacheBreakpoint(result.Cache)
			if err != nil {
				return nil, err
			}
			content = []contentPart{{Type: "text", Text: text, PromptCacheBreakpoint: breakpoint}}
		}
		out = append(out, chatMessage{
			Role:       string(litellm.RoleTool),
			ToolCallID: result.ToolUseID,
			Content:    content,
		})
	}
	return out, nil
}

func convertPromptCacheBreakpoint(cache *litellm.CacheControl) (*promptCacheBreakpoint, error) {
	if cache == nil {
		return nil, nil
	}
	if cache.Type != "" && cache.Type != litellm.CacheTypeEphemeral {
		return nil, fmt.Errorf("openai: cache breakpoint type must be %q", litellm.CacheTypeEphemeral)
	}
	if cache.TTL != "" {
		return nil, fmt.Errorf("openai: cache breakpoint TTL must be set with prompt_cache_options.ttl")
	}
	return &promptCacheBreakpoint{Mode: "explicit"}, nil
}

func toolResultText(blocks []litellm.Block) (string, error) {
	var text strings.Builder
	for _, block := range blocks {
		switch b := block.(type) {
		case litellm.TextBlock:
			if b.Cache != nil {
				return "", fmt.Errorf("OpenAI Chat tool result content cache must be set on ToolResultBlock")
			}
			if text.Len() > 0 {
				text.WriteString("\n")
			}
			text.WriteString(b.Text)
		default:
			return "", fmt.Errorf("OpenAI Chat tool results only support text content, got %T", block)
		}
	}
	return text.String(), nil
}

func imageURLValue(block litellm.ImageBlock) (string, error) {
	switch {
	case block.URL != "":
		return block.URL, nil
	case len(block.Data) > 0:
		if block.MIME == "" {
			return "", fmt.Errorf("inline image MIME is required")
		}
		return "data:" + block.MIME + ";base64," + base64.StdEncoding.EncodeToString(block.Data), nil
	case block.FileURI != "":
		return block.FileURI, nil
	default:
		return "", fmt.Errorf("image requires URL, data, or file URI")
	}
}

func convertTools(tools []litellm.Tool) ([]tool, error) {
	out := make([]tool, 0, len(tools))
	for _, t := range tools {
		if t.Name == "" {
			return nil, fmt.Errorf("openai: tool name is required")
		}
		var params any = map[string]any{"type": "object"}
		if len(t.Parameters) > 0 {
			var decoded any
			if err := json.Unmarshal(t.Parameters, &decoded); err != nil {
				return nil, fmt.Errorf("openai: tool %q parameters must be valid JSON: %w", t.Name, err)
			}
			params = decoded
		}
		var strict *bool
		if t.Strict == litellm.StrictEnabled {
			normalised, err := normalizeStrictSchema(params)
			if err != nil {
				return nil, fmt.Errorf("openai: tool %q strict schema invalid: %w", t.Name, err)
			}
			params = normalised
			strict = litellm.Bool(true)
		} else if t.Strict == litellm.StrictDisabled {
			strict = litellm.Bool(false)
		}
		out = append(out, tool{
			Type: "function",
			Function: &toolFunction{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  params,
				Strict:      strict,
			},
		})
	}
	return out, nil
}

func convertResponseFormat(format *litellm.ResponseFormat) (*responseFormat, error) {
	out := &responseFormat{Type: string(format.Type)}
	if format.Type != litellm.ResponseFormatJSONSchema {
		return out, nil
	}
	if format.JSONSchema == nil {
		return nil, fmt.Errorf("openai: json schema response format requires schema")
	}
	var schema any
	if len(format.JSONSchema.Schema) > 0 {
		if err := json.Unmarshal(format.JSONSchema.Schema, &schema); err != nil {
			return nil, fmt.Errorf("openai: response schema must be valid JSON: %w", err)
		}
	}
	var strict *bool
	switch format.JSONSchema.Strict {
	case litellm.StrictEnabled:
		normalised, err := normalizeStrictSchema(schema)
		if err != nil {
			return nil, fmt.Errorf("openai: response strict schema invalid: %w", err)
		}
		schema = normalised
		strict = litellm.Bool(true)
	case litellm.StrictDisabled:
		strict = litellm.Bool(false)
	}
	out.JSONSchema = &jsonSchema{
		Name:        format.JSONSchema.Name,
		Description: format.JSONSchema.Description,
		Schema:      schema,
		Strict:      strict,
	}
	return out, nil
}

func convertToolChoice(choice *litellm.ToolChoice) (any, error) {
	if err := choice.Validate(); err != nil {
		return nil, err
	}
	if choice == nil {
		return nil, nil
	}
	if choice.Name != "" {
		return map[string]any{"type": "function", "function": map[string]any{"name": choice.Name}}, nil
	}
	return string(choice.Mode), nil
}

func validateSampling(temperature, topP *float64) error {
	if temperature != nil && (math.IsNaN(*temperature) || *temperature < 0 || *temperature > 2) {
		return fmt.Errorf("openai: temperature must be between 0 and 2")
	}
	if topP != nil && (math.IsNaN(*topP) || *topP < 0 || *topP > 1) {
		return fmt.Errorf("openai: top_p must be between 0 and 1")
	}
	return nil
}
