package openai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"strings"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/internal/wire"
	"github.com/voocel/litellm/provider/internal/openaicompat"
)

func (p *Provider) responses(ctx context.Context, req *litellm.Request) (*litellm.Response, error) {
	body, err := buildResponsesRequest(req, false)
	if err != nil {
		return nil, litellm.WrapError(p.Name(), litellm.ErrorTypeValidation, err)
	}
	resp, err := p.post(ctx, body, false)
	if err != nil {
		return nil, err
	}
	data, err := p.readResponse(resp)
	if err != nil {
		return nil, err
	}
	var parsed responsesResponse
	if err := json.Unmarshal(data, &parsed); err != nil {
		return nil, litellm.NewError(p.Name(), litellm.ErrorTypeProvider, "decode response", err)
	}
	if parsed.Error != nil {
		return nil, wire.StreamError(p.Name(), parsed.Error.Code, parsed.Error.Message)
	}
	out := convertResponsesResponse(&parsed, req.Model)
	out.Raw = data
	return out, nil
}

func (p *Provider) responsesStream(ctx context.Context, req *litellm.Request) (litellm.Stream, error) {
	body, err := buildResponsesRequest(req, true)
	if err != nil {
		return nil, litellm.WrapError(p.Name(), litellm.ErrorTypeValidation, err)
	}
	resp, err := p.post(ctx, body, true)
	if err != nil {
		return nil, err
	}
	return newResponsesStream(resp, req.Model), nil
}

func buildResponsesRequest(req *litellm.Request, stream bool) ([]byte, error) {
	if len(req.Stop) > 0 {
		return nil, errors.New("stop is not supported by the responses API")
	}
	opts, err := req.ProviderOptions.Decode()
	if err != nil {
		return nil, err
	}
	if err := wire.CheckOptions(opts, responsesOptions); err != nil {
		return nil, err
	}
	body := map[string]any{"model": req.Model}
	if stream {
		body["stream"] = true
	}
	if req.MaxTokens != nil {
		body["max_output_tokens"] = *req.MaxTokens
	}
	if req.Temperature != nil {
		body["temperature"] = *req.Temperature
	}
	if req.TopP != nil {
		body["top_p"] = *req.TopP
	}
	instructions, input, err := responsesInput(req.Messages)
	if err != nil {
		return nil, err
	}
	if instructions != "" {
		body["instructions"] = instructions
	}
	if len(input) > 0 {
		body["input"] = input
	}
	if len(req.Tools) > 0 {
		body["tools"] = responsesTools(req.Tools)
	}
	if choice := req.ToolChoice; choice != nil {
		if choice.Name != "" {
			body["tool_choice"] = map[string]any{"type": "function", "name": choice.Name}
		} else {
			body["tool_choice"] = string(choice.Mode)
		}
	}
	if format := responsesFormat(req.ResponseFormat); format != nil {
		body["text"] = map[string]any{"format": format}
	}
	if req.Thinking != nil {
		reasoning, err := responsesReasoning(req.Thinking)
		if err != nil {
			return nil, err
		}
		if len(reasoning) > 0 {
			body["reasoning"] = reasoning
		}
	}
	if err := wire.ApplyOptions(body, opts); err != nil {
		return nil, err
	}
	return json.Marshal(body)
}

func responsesReasoning(thinking *litellm.Thinking) (map[string]any, error) {
	if thinking.BudgetTokens != nil {
		return nil, errors.New("thinking budget_tokens is not supported; use effort")
	}
	if thinking.Mode == litellm.ThinkingDisabled {
		return map[string]any{"effort": "none"}, nil
	}
	out := map[string]any{}
	if thinking.Effort != "" {
		out["effort"] = thinking.Effort
	}
	if thinking.IncludeOutput {
		out["summary"] = "auto"
	}
	return out, nil
}

func responsesFormat(format *litellm.ResponseFormat) map[string]any {
	if format == nil || format.Type == "" {
		return nil
	}
	out := map[string]any{"type": string(format.Type)}
	if format.Type != litellm.ResponseFormatJSONSchema {
		return out
	}
	out["name"] = format.JSONSchema.Name
	if format.JSONSchema.Description != "" {
		out["description"] = format.JSONSchema.Description
	}
	if len(format.JSONSchema.Schema) > 0 {
		out["schema"] = json.RawMessage(format.JSONSchema.Schema)
	}
	if strict, ok := format.JSONSchema.Strict.Value(); ok {
		out["strict"] = strict
	}
	return out
}

func responsesTools(tools []litellm.Tool) []any {
	out := make([]any, 0, len(tools))
	for _, tool := range tools {
		fn := map[string]any{"type": "function", "name": tool.Name, "parameters": map[string]any{"type": "object"}}
		if tool.Description != "" {
			fn["description"] = tool.Description
		}
		if len(tool.Parameters) > 0 {
			fn["parameters"] = json.RawMessage(tool.Parameters)
		}
		if strict, ok := tool.Strict.Value(); ok {
			fn["strict"] = strict
		}
		out = append(out, fn)
	}
	return out
}

// responsesInput moves system text into instructions. A system message with a
// cache breakpoint stays in the input as a developer message, since
// instructions cannot carry one.
func responsesInput(messages []litellm.Message) (string, []any, error) {
	var instructions []string
	items := make([]any, 0, len(messages))
	for i, msg := range messages {
		var err error
		switch msg.Role {
		case litellm.RoleSystem:
			if text, cached, ok := textContent(msg.Blocks); ok && !cached {
				instructions = append(instructions, text)
				continue
			}
			items, err = appendMessage(items, "developer", "input_text", msg.Blocks)
		case litellm.RoleUser:
			items, err = appendMessage(items, "user", "input_text", msg.Blocks)
		case litellm.RoleAssistant:
			items, err = appendAssistant(items, msg.Blocks)
		case litellm.RoleTool:
			items, err = appendToolResults(items, msg.Blocks)
		}
		if err != nil {
			return "", nil, fmt.Errorf("messages[%d]: %w", i, err)
		}
	}
	return strings.Join(instructions, "\n"), items, nil
}

// textContent joins text-only blocks and reports whether any carries a cache
// breakpoint.
func textContent(blocks []litellm.Block) (text string, cached, ok bool) {
	parts := make([]string, 0, len(blocks))
	for _, block := range blocks {
		b, isText := block.(litellm.TextBlock)
		if !isText {
			return "", false, false
		}
		parts = append(parts, b.Text)
		cached = cached || b.Cache != nil
	}
	return strings.Join(parts, "\n"), cached, true
}

func appendMessage(items []any, role, textType string, blocks []litellm.Block) ([]any, error) {
	content := make([]any, 0, len(blocks))
	for _, block := range blocks {
		var part map[string]any
		var cache *litellm.CacheControl
		switch b := block.(type) {
		case litellm.TextBlock:
			if b.Text == "" {
				continue
			}
			part, cache = map[string]any{"type": textType, "text": b.Text}, b.Cache
		case litellm.ImageBlock:
			part, cache = map[string]any{"type": "input_image"}, b.Cache
			if b.FileURI != "" {
				part["file_id"] = b.FileURI
			} else {
				url, err := openaicompat.ImageURL(b)
				if err != nil {
					return nil, err
				}
				part["image_url"] = url
			}
			if b.Detail != "" {
				part["detail"] = b.Detail
			}
		default:
			return nil, fmt.Errorf("unsupported block %T", block)
		}
		// Only input content has a breakpoint slot.
		if cache != nil && textType == "input_text" {
			fields, err := promptCacheBreakpoint(cache)
			if err != nil {
				return nil, err
			}
			maps.Copy(part, fields)
		}
		content = append(content, part)
	}
	if len(content) == 0 {
		return items, nil
	}
	return append(items, map[string]any{"type": "message", "role": role, "content": content}), nil
}

func appendAssistant(items []any, blocks []litellm.Block) ([]any, error) {
	for _, block := range blocks {
		var err error
		switch b := block.(type) {
		case litellm.TextBlock:
			items, err = appendMessage(items, "assistant", "output_text", []litellm.Block{b})
		case litellm.ReasoningBlock:
			items = appendReasoning(items, b)
		case litellm.ToolUseBlock:
			items = append(items, map[string]any{"type": "function_call", "call_id": b.ID, "name": b.Name, "arguments": string(b.Arguments)})
		default:
			err = fmt.Errorf("unsupported block %T", block)
		}
		if err != nil {
			return nil, err
		}
	}
	return items, nil
}

// appendReasoning replays the reasoning item kept in Extra. Reasoning from
// other providers is not sent: an input reasoning item requires the id the
// API assigned to it.
func appendReasoning(items []any, block litellm.ReasoningBlock) []any {
	var item struct {
		Type string `json:"type"`
	}
	if json.Unmarshal(block.Extra, &item) != nil || item.Type != "reasoning" {
		return items
	}
	return append(items, json.RawMessage(block.Extra))
}

func appendToolResults(items []any, blocks []litellm.Block) ([]any, error) {
	for _, block := range blocks {
		result, ok := block.(litellm.ToolResultBlock)
		if !ok {
			return nil, fmt.Errorf("tool role only supports ToolResultBlock, got %T", block)
		}
		output, _, ok := textContent(result.Content)
		if !ok {
			return nil, fmt.Errorf("tool result %q only supports text content", result.ToolUseID)
		}
		items = append(items, map[string]any{"type": "function_call_output", "call_id": result.ToolUseID, "output": output})
	}
	return items, nil
}
