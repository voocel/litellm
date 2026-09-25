package openaicompat

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"slices"
	"strings"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/internal/wire"
)

func (p *Provider) buildRequest(req *litellm.Request, stream bool) ([]byte, error) {
	opts, err := req.ProviderOptions.Decode()
	if err != nil {
		return nil, err
	}
	if !p.cfg.AllowUnknownProviderOptions {
		if err := wire.CheckOptions(opts, p.spec.Options); err != nil {
			return nil, err
		}
	}
	if n, ok := opts["n"]; ok && !isOne(n) {
		return nil, errors.New(`provider option "n" must be 1; litellm.Response holds a single output`)
	}
	body := maps.Clone(p.spec.Fields)
	if body == nil {
		body = make(map[string]any)
	}
	body["model"] = req.Model
	messages, err := p.convertMessages(req.Messages)
	if err != nil {
		return nil, err
	}
	body["messages"] = messages
	if stream {
		body["stream"] = true
		if !p.spec.OmitStreamOptions {
			body["stream_options"] = map[string]any{"include_usage": true}
		}
	}
	if req.MaxTokens != nil {
		body[p.spec.maxTokensField()] = *req.MaxTokens
	}
	if req.Temperature != nil {
		body["temperature"] = *req.Temperature
	}
	if req.TopP != nil {
		body["top_p"] = *req.TopP
	}
	if len(req.Stop) > 0 {
		body["stop"] = req.Stop
	}
	if len(req.Tools) > 0 {
		tools, err := convertTools(req.Tools)
		if err != nil {
			return nil, err
		}
		body["tools"] = tools
	}
	if req.ToolChoice != nil {
		body["tool_choice"] = convertToolChoice(req.ToolChoice)
	}
	if req.ResponseFormat != nil {
		format, err := convertResponseFormat(req.ResponseFormat)
		if err != nil {
			return nil, err
		}
		if format != nil {
			body["response_format"] = format
		}
	}
	if req.Thinking != nil {
		fields, err := p.convertThinking(req.Thinking)
		if err != nil {
			return nil, err
		}
		maps.Copy(body, fields)
	}
	// Marshal normalizes the generated body to JSON types first, so options
	// merge into any generated object or array alike.
	return wire.MarshalBody(body, opts)
}

func (p *Provider) convertThinking(thinking *litellm.Thinking) (map[string]any, error) {
	if thinking.Mode == litellm.ThinkingDisabled && p.spec.ThinkingAlwaysOn {
		return nil, errors.New("thinking cannot be disabled")
	}
	if p.spec.Thinking != nil {
		return p.spec.Thinking(thinking)
	}
	return reasoningEffort(thinking)
}

func isOne(value any) bool {
	number, ok := value.(json.Number)
	if !ok {
		return false
	}
	n, err := number.Float64()
	return err == nil && n == 1
}

func (p *Provider) convertMessages(messages []litellm.Message) ([]map[string]any, error) {
	out := make([]map[string]any, 0, len(messages))
	for i, msg := range messages {
		if msg.Role == litellm.RoleTool {
			results, err := p.convertToolResults(msg.Blocks)
			if err != nil {
				return nil, fmt.Errorf("messages[%d]: %w", i, err)
			}
			out = append(out, results...)
			continue
		}
		converted, err := p.convertMessage(msg)
		if err != nil {
			return nil, fmt.Errorf("messages[%d]: %w", i, err)
		}
		if len(converted) == 1 {
			continue // only the role is left, e.g. after reasoning with no field to go in
		}
		out = append(out, converted)
	}
	return out, nil
}

func (p *Provider) convertMessage(msg litellm.Message) (map[string]any, error) {
	out := map[string]any{"role": string(msg.Role)}
	parts := make([]map[string]any, 0, len(msg.Blocks))
	var toolCalls []map[string]any
	for _, block := range msg.Blocks {
		switch b := block.(type) {
		case litellm.TextBlock:
			if b.Text == "" {
				continue
			}
			part, err := p.withCache(map[string]any{"type": "text", "text": b.Text}, b.Cache)
			if err != nil {
				return nil, err
			}
			parts = append(parts, part)
		case litellm.ImageBlock:
			url, err := ImageURL(b)
			if err != nil {
				return nil, err
			}
			image := map[string]any{"url": url}
			if b.Detail != "" {
				image["detail"] = b.Detail
			}
			part, err := p.withCache(map[string]any{"type": "image_url", "image_url": image}, b.Cache)
			if err != nil {
				return nil, err
			}
			parts = append(parts, part)
		case litellm.ToolUseBlock:
			toolCalls = append(toolCalls, map[string]any{
				"id":       b.ID,
				"type":     "function",
				"function": map[string]any{"name": b.Name, "arguments": string(b.Arguments)},
			})
		case litellm.ReasoningBlock:
			p.putReasoning(out, b)
		default:
			return nil, fmt.Errorf("unsupported block %T", block)
		}
	}
	switch {
	case len(parts) == 1 && len(parts[0]) == 2 && parts[0]["type"] == "text":
		out["content"] = parts[0]["text"]
	case len(parts) > 0:
		out["content"] = parts
	case len(toolCalls) > 0 && p.spec.EmptyToolCallContent:
		out["content"] = ""
	}
	if len(toolCalls) > 0 {
		out["tool_calls"] = toolCalls
	}
	return out, nil
}

func (p *Provider) convertToolResults(blocks []litellm.Block) ([]map[string]any, error) {
	out := make([]map[string]any, 0, len(blocks))
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
		if result.Cache != nil && p.spec.Cache != nil {
			part, err := p.withCache(map[string]any{"type": "text", "text": text}, result.Cache)
			if err != nil {
				return nil, err
			}
			content = []map[string]any{part}
		}
		out = append(out, map[string]any{"role": "tool", "tool_call_id": result.ToolUseID, "content": content})
	}
	return out, nil
}

func (p *Provider) withCache(part map[string]any, cache *litellm.CacheControl) (map[string]any, error) {
	if cache == nil || p.spec.Cache == nil {
		return part, nil
	}
	fields, err := p.spec.Cache(cache)
	if err != nil {
		return nil, err
	}
	maps.Copy(part, fields)
	return part, nil
}

// putReasoning replays a ReasoningBlock. reasoning_details this provider
// produced are appended verbatim and supersede the text field; otherwise Text,
// whatever its origin, goes to the first other field the spec names. Blocks
// with neither are dropped.
func (p *Provider) putReasoning(message map[string]any, block litellm.ReasoningBlock) {
	var field string
	for _, name := range p.spec.ReasoningFields {
		if name != "reasoning_details" {
			field = name
			break
		}
	}
	if items, ok := wire.ReadState[[]any](block.State, p.spec.Name); ok && items != nil && slices.Contains(p.spec.ReasoningFields, "reasoning_details") {
		current, _ := message["reasoning_details"].([]any)
		message["reasoning_details"] = append(current, items...)
		if field != "" {
			delete(message, field)
		}
		return
	}
	if block.Text == "" || field == "" || message["reasoning_details"] != nil {
		return
	}
	if current, _ := message[field].(string); current != "" {
		message[field] = current + "\n\n" + block.Text
	} else {
		message[field] = block.Text
	}
}

// ImageURL returns the image_url value for block: its URL, a data URL for
// inline data, or its file URI.
func ImageURL(block litellm.ImageBlock) (string, error) {
	switch {
	case block.URL != "":
		return block.URL, nil
	case len(block.Data) > 0:
		if block.MIME == "" {
			return "", errors.New("inline image requires MIME")
		}
		return "data:" + block.MIME + ";base64," + base64.StdEncoding.EncodeToString(block.Data), nil
	case block.FileURI != "":
		return block.FileURI, nil
	default:
		return "", errors.New("image requires URL, data or file URI")
	}
}

func toolResultText(blocks []litellm.Block) (string, error) {
	var out strings.Builder
	for _, block := range blocks {
		text, ok := block.(litellm.TextBlock)
		if !ok {
			return "", fmt.Errorf("tool results only support text content, got %T", block)
		}
		if out.Len() > 0 {
			out.WriteString("\n")
		}
		out.WriteString(text.Text)
	}
	return out.String(), nil
}

func convertTools(tools []litellm.Tool) ([]any, error) {
	out := make([]any, 0, len(tools))
	for _, tool := range tools {
		fn := map[string]any{"name": tool.Name, "parameters": map[string]any{"type": "object"}}
		if tool.Description != "" {
			fn["description"] = tool.Description
		}
		if len(tool.Parameters) > 0 {
			fn["parameters"] = json.RawMessage(tool.Parameters)
		}
		if strict, ok := tool.Strict.Value(); ok {
			fn["strict"] = strict
		}
		out = append(out, map[string]any{"type": "function", "function": fn})
	}
	return out, nil
}

func convertResponseFormat(format *litellm.ResponseFormat) (any, error) {
	switch format.Type {
	case "", litellm.ResponseFormatText:
		return nil, nil
	case litellm.ResponseFormatJSONObject:
		return map[string]any{"type": "json_object"}, nil
	case litellm.ResponseFormatJSONSchema:
		schema := map[string]any{"name": format.JSONSchema.Name}
		if format.JSONSchema.Description != "" {
			schema["description"] = format.JSONSchema.Description
		}
		if len(format.JSONSchema.Schema) > 0 {
			schema["schema"] = json.RawMessage(format.JSONSchema.Schema)
		}
		if strict, ok := format.JSONSchema.Strict.Value(); ok {
			schema["strict"] = strict
		}
		return map[string]any{"type": "json_schema", "json_schema": schema}, nil
	default:
		return nil, fmt.Errorf("unsupported response format %q", format.Type)
	}
}

func convertToolChoice(choice *litellm.ToolChoice) any {
	if choice.Name != "" {
		return map[string]any{"type": "function", "function": map[string]any{"name": choice.Name}}
	}
	return string(choice.Mode)
}
