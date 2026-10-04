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
	input, responseFormat := req.Messages, req.ResponseFormat
	if p.spec.usesSchemaPrompt(responseFormat) {
		input = withSchemaPrompt(input, responseFormat.JSONSchema)
		responseFormat = &litellm.ResponseFormat{Type: p.spec.SchemaFallback}
	}
	messages, err := p.convertMessages(input)
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
	if offered := req.OfferedTools(); len(offered) > 0 {
		tools, err := convertTools(offered)
		if err != nil {
			return nil, err
		}
		body["tools"] = tools
	}
	if req.ToolChoice != nil {
		body["tool_choice"] = convertToolChoice(req.ToolChoice)
	}
	if responseFormat != nil {
		format, err := convertResponseFormat(responseFormat)
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
	if thinking.Disabled && p.spec.ThinkingAlwaysOn {
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
	// Tool messages carry text only: the images of a turn's results follow
	// its tool messages in a user message.
	var images []map[string]any
	for i, msg := range messages {
		if msg.Role == litellm.RoleTool {
			results, parts, err := p.convertToolResults(msg.Blocks)
			if err != nil {
				return nil, fmt.Errorf("messages[%d]: %w", i, err)
			}
			out = append(out, results...)
			images = append(images, parts...)
			if len(images) > 0 && (i+1 == len(messages) || messages[i+1].Role != litellm.RoleTool) {
				out = append(out, map[string]any{"role": "user", "content": images})
				images = nil
			}
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
			parts = append(parts, p.withCache(map[string]any{"type": "text", "text": b.Text}, b.Cache))
		case litellm.ImageBlock:
			if slices.Contains(p.spec.StringContentRoles, msg.Role) {
				return nil, fmt.Errorf("%s messages do not support images", msg.Role)
			}
			part, err := p.imagePart(b)
			if err != nil {
				return nil, err
			}
			parts = append(parts, part)
		case litellm.ToolUseBlock:
			toolCalls = append(toolCalls, map[string]any{
				"id":       b.ID,
				"type":     "function",
				"function": map[string]any{"name": b.Name, "arguments": b.Arguments},
			})
		case litellm.ReasoningBlock:
			p.putReasoning(out, b)
		default:
			return nil, fmt.Errorf("unsupported block %T", block)
		}
	}
	switch {
	case slices.Contains(p.spec.StringContentRoles, msg.Role):
		var text strings.Builder
		for _, part := range parts {
			text.WriteString(part["text"].(string))
		}
		out["content"] = text.String()
	case len(parts) == 1 && len(parts[0]) == 2 && parts[0]["type"] == "text":
		out["content"] = parts[0]["text"]
	case len(parts) > 0:
		out["content"] = parts
	}
	if len(toolCalls) > 0 {
		out["tool_calls"] = toolCalls
	}
	return out, nil
}

// convertToolResults converts tool results to tool messages, and returns
// the user message parts of their images.
func (p *Provider) convertToolResults(blocks []litellm.Block) ([]map[string]any, []map[string]any, error) {
	out := make([]map[string]any, 0, len(blocks))
	var images []map[string]any
	for _, block := range blocks {
		result, ok := block.(litellm.ToolResultBlock)
		if !ok {
			return nil, nil, fmt.Errorf("tool role only supports ToolResultBlock, got %T", block)
		}
		var texts []string
		var parts []map[string]any
		for _, content := range result.Content {
			switch c := content.(type) {
			case litellm.TextBlock:
				texts = append(texts, c.Text)
			case litellm.ToolReferenceBlock:
				texts = append(texts, wire.ToolReferenceText(c))
			case litellm.ImageBlock:
				part, err := p.imagePart(c)
				if err != nil {
					return nil, nil, err
				}
				parts = append(parts, part)
			default:
				return nil, nil, fmt.Errorf("tool results do not support %T", content)
			}
		}
		text := strings.Join(texts, "\n")
		if len(parts) > 0 {
			if text == "" {
				text = "The result is the image in the next message."
			}
			images = append(images, map[string]any{"type": "text", "text": "The image of tool call " + result.ToolUseID + ":"})
			images = append(images, parts...)
		}
		var content any = text
		if result.Cache != nil && p.spec.Cache != nil {
			content = []map[string]any{p.withCache(map[string]any{"type": "text", "text": text}, result.Cache)}
		}
		out = append(out, map[string]any{"role": "tool", "tool_call_id": result.ToolUseID, "content": content})
	}
	return out, images, nil
}

// imagePart is the content part of an image.
func (p *Provider) imagePart(b litellm.ImageBlock) (map[string]any, error) {
	if b.FileURI != "" && p.spec.ImageFileID {
		return map[string]any{"type": "file", "file_id": b.FileURI}, nil
	}
	url, err := ImageURL(b)
	if err != nil {
		return nil, err
	}
	image := map[string]any{"url": url}
	if b.Detail != "" {
		image["detail"] = b.Detail
	}
	return p.withCache(map[string]any{"type": "image_url", "image_url": image}, b.Cache), nil
}

func (p *Provider) withCache(part map[string]any, cache *litellm.CacheControl) map[string]any {
	if cache != nil {
		maps.Copy(part, p.spec.Cache)
	}
	return part
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
	if items, ok := wire.ReadState[[]any](block.State, p.Name()); ok && items != nil && slices.Contains(p.spec.ReasoningFields, "reasoning_details") {
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
		if tool.Strict != nil {
			fn["strict"] = *tool.Strict
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
		if format.JSONSchema.Strict != nil {
			schema["strict"] = *format.JSONSchema.Strict
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
