package openai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"slices"
	"strings"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/internal/wire"
	"github.com/voocel/litellm/provider/internal/openaicompat"
)

func (p *Provider) responses(ctx context.Context, req *litellm.Request) (*litellm.Response, error) {
	body, err := buildResponsesRequest(req, p.Name(), false)
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
	if parsed.Status == "queued" || parsed.Status == "in_progress" {
		return nil, litellm.NewError(p.Name(), litellm.ErrorTypeProvider, "response is not complete: "+parsed.Status, nil)
	}
	out := convertResponsesResponse(&parsed, p.Name(), req.Model)
	out.Raw = data
	return out, nil
}

func (p *Provider) responsesStream(ctx context.Context, req *litellm.Request) (litellm.Stream, error) {
	body, err := buildResponsesRequest(req, p.Name(), true)
	if err != nil {
		return nil, litellm.WrapError(p.Name(), litellm.ErrorTypeValidation, err)
	}
	resp, err := p.post(ctx, body, true)
	if err != nil {
		return nil, err
	}
	return newResponsesStream(resp, p.Name(), req.Model), nil
}

func buildResponsesRequest(req *litellm.Request, provider string, stream bool) ([]byte, error) {
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
	// Chat returns a completed reply; it has no asynchronous job handle or
	// polling API. A background stream still delivers the terminal result.
	if !stream && opts[ProviderOptionBackground] == true {
		return nil, errors.New("background=true requires Stream; Chat does not support background jobs")
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
	instructions, input, err := responsesInput(req.Messages, provider, req.Model)
	if err != nil {
		return nil, err
	}
	if instructions != "" {
		body["instructions"] = instructions
	}
	if len(input) > 0 {
		body["input"] = input
	}
	if offered := req.OfferedTools(); len(offered) > 0 {
		body["tools"] = responsesTools(offered)
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
	if thinking.Disabled {
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
	if strict := format.JSONSchema.Strict; strict != nil {
		out["strict"] = *strict
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
		if tool.Strict != nil {
			fn["strict"] = *tool.Strict
		}
		out = append(out, fn)
	}
	return out
}

// responsesInput moves leading system text into instructions. Later system
// messages stay in place as developer messages, where changing them keeps the
// cached prefix valid; so does one with a cache breakpoint, which instructions
// cannot carry.
func responsesInput(messages []litellm.Message, provider, model string) (string, []any, error) {
	var instructions []string
	items := make([]any, 0, len(messages))
	for i, msg := range messages {
		var err error
		switch msg.Role {
		case litellm.RoleSystem:
			if text, cached, ok := textContent(msg.Blocks); ok && !cached && len(items) == 0 {
				instructions = append(instructions, text)
				continue
			}
			items, err = appendMessage(items, "developer", msg.Blocks)
		case litellm.RoleUser:
			items, err = appendMessage(items, "user", msg.Blocks)
		case litellm.RoleAssistant:
			items, err = appendAssistant(items, msg.Blocks, provider, model)
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

func appendMessage(items []any, role string, blocks []litellm.Block) ([]any, error) {
	content, err := inputContent(blocks)
	if err != nil || len(content) == 0 {
		return items, err
	}
	return append(items, map[string]any{"type": "message", "role": role, "content": content}), nil
}

// inputContent maps blocks to input content parts. Tool references, which
// have no input part, are sent as text.
func inputContent(blocks []litellm.Block) ([]map[string]any, error) {
	content := make([]map[string]any, 0, len(blocks))
	for _, block := range blocks {
		var part map[string]any
		var cache *litellm.CacheControl
		switch b := block.(type) {
		case litellm.TextBlock:
			if b.Text == "" {
				continue
			}
			part, cache = map[string]any{"type": "input_text", "text": b.Text}, b.Cache
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
		case litellm.ToolReferenceBlock:
			part, cache = map[string]any{"type": "input_text", "text": wire.ToolReferenceText(b)}, b.Cache
		default:
			return nil, fmt.Errorf("unsupported block %T", block)
		}
		if cache != nil {
			maps.Copy(part, promptCacheBreakpoint)
		}
		content = append(content, part)
	}
	return content, nil
}

// appendAssistant maps an assistant message. The API pairs a reasoning item
// with the ids of the items after it, and only the model that produced the
// reasoning accepts it. So a message holding reasoning from the requested
// model is replayed whole, reasoning items and ids included; any other is sent
// as plain content, without reasoning or ids. Reasoning from other providers
// is never sent: an input reasoning item needs the id the API assigned to it.
func appendAssistant(items []any, blocks []litellm.Block, provider, model string) ([]any, error) {
	replay := slices.ContainsFunc(blocks, func(block litellm.Block) bool {
		b, ok := block.(litellm.ReasoningBlock)
		return ok && b.State != nil && b.State.Provider == provider && b.State.Model == model
	})
	for _, block := range blocks {
		switch b := block.(type) {
		case litellm.TextBlock:
			if b.Text == "" {
				continue
			}
			state, _ := wire.ReadState[itemState](b.State, provider)
			if !replay {
				state.ID = ""
			}
			items = appendOutputText(items, b.Text, state)
		case litellm.ReasoningBlock:
			if item, ok := wire.ReadState[json.RawMessage](b.State, provider); ok && replay {
				items = append(items, item)
			}
		case litellm.ToolUseBlock:
			call := map[string]any{"type": "function_call", "call_id": b.ID, "name": b.Name, "arguments": b.Arguments}
			if state, ok := wire.ReadState[itemState](b.State, provider); ok && replay && state.ID != "" {
				call["id"] = state.ID
			}
			items = append(items, call)
		default:
			return nil, fmt.Errorf("unsupported block %T", block)
		}
	}
	return items, nil
}

// appendOutputText adds assistant text, joining the parts of one output
// message. Phase is always kept, as the API asks.
func appendOutputText(items []any, text string, state itemState) []any {
	part := map[string]any{"type": "output_text", "text": text}
	if n := len(items); n > 0 && state.ID != "" {
		if last, ok := items[n-1].(map[string]any); ok && last["type"] == "message" && last["id"] == state.ID {
			last["content"] = append(last["content"].([]any), part)
			return items
		}
	}
	msg := map[string]any{"type": "message", "role": "assistant", "content": []any{part}}
	if state.ID != "" {
		msg["id"] = state.ID
	}
	if state.Phase != "" {
		msg["phase"] = state.Phase
	}
	return append(items, msg)
}

// appendToolResults sends text-only output as a string and any other as a
// content list, which can hold images and cache breakpoints.
func appendToolResults(items []any, blocks []litellm.Block) ([]any, error) {
	for _, block := range blocks {
		result, ok := block.(litellm.ToolResultBlock)
		if !ok {
			return nil, fmt.Errorf("tool role only supports ToolResultBlock, got %T", block)
		}
		var output any
		if text, cached, ok := textContent(result.Content); ok && !cached && result.Cache == nil {
			output = text
		} else {
			parts, err := inputContent(result.Content)
			if err != nil {
				return nil, fmt.Errorf("tool result %q: %w", result.ToolUseID, err)
			}
			if result.Cache != nil {
				if len(parts) == 0 {
					parts = append(parts, map[string]any{"type": "input_text", "text": ""})
				}
				maps.Copy(parts[len(parts)-1], promptCacheBreakpoint)
			}
			output = parts
		}
		items = append(items, map[string]any{"type": "function_call_output", "call_id": result.ToolUseID, "output": output})
	}
	return items, nil
}
