package compat

import (
	"encoding/json"
	"fmt"

	"github.com/voocel/litellm"
)

func (p *Provider) convertResponse(resp *chatResponse, req *litellm.Request) (*litellm.Response, error) {
	out := &litellm.Response{
		Provider: p.Name(),
		Model:    req.Model,
		Usage:    convertUsage(resp.Usage, p.spec, p.Name(), req.Model),
	}
	if p.spec.Response.ModelFromResponse && resp.Model != "" {
		out.Model = resp.Model
		out.Usage.Model = resp.Model
	}
	if len(resp.Choices) == 0 {
		return out, nil
	}
	choice := resp.Choices[0]
	out.FinishReason = litellm.NormalizeFinishReason(choice.FinishReason)
	if reasoning := findReasoning(messageReasoningMap(choice.Message), p.reasoningFields(false)); reasoning != "" {
		extra, err := rawReasoningExtra(choice.Message)
		if err != nil {
			return nil, fmt.Errorf("%s: convert reasoning details: %w", p.Name(), err)
		}
		out.Blocks = append(out.Blocks, litellm.ReasoningBlock{Text: reasoning, Extra: extra})
	}
	blocks, err := contentBlocks(choice.Message.Content)
	if err != nil {
		return nil, fmt.Errorf("%s: convert response content: %w", p.Name(), err)
	}
	out.Blocks = append(out.Blocks, blocks...)
	for _, call := range choice.Message.ToolCalls {
		if !json.Valid([]byte(call.Function.Arguments)) {
			return nil, fmt.Errorf("%s: tool call %q arguments are not valid JSON", p.Name(), call.ID)
		}
		out.Blocks = append(out.Blocks, litellm.ToolUseBlock{
			ID:        call.ID,
			Name:      call.Function.Name,
			Arguments: json.RawMessage(call.Function.Arguments),
		})
	}
	return out, nil
}

func convertUsage(u usage, spec Spec, provider, model string) litellm.Usage {
	out := litellm.Usage{
		InputTokens:  u.PromptTokens,
		OutputTokens: u.CompletionTokens,
		TotalTokens:  u.TotalTokens,
		Provider:     provider,
		Model:        model,
	}
	if spec.Response.HasCompletionTokenDetails && u.CompletionTokensDetails != nil {
		out.ReasoningTokens = u.CompletionTokensDetails.ReasoningTokens
	}
	if spec.Response.HasCacheTokens {
		out.CacheReadTokens = u.PromptCacheHitTokens
	}
	return out
}

func contentBlocks(raw json.RawMessage) ([]litellm.Block, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		if text == "" {
			return nil, nil
		}
		return []litellm.Block{litellm.TextBlock{Text: text}}, nil
	}
	var parts []map[string]any
	if err := json.Unmarshal(raw, &parts); err != nil {
		return nil, fmt.Errorf("unsupported content payload: %w", err)
	}
	blocks := make([]litellm.Block, 0, len(parts))
	for _, part := range parts {
		partType, _ := part["type"].(string)
		switch partType {
		case "text":
			if text, _ := part["text"].(string); text != "" {
				blocks = append(blocks, litellm.TextBlock{Text: text})
			}
		default:
			return nil, fmt.Errorf("unsupported content part type %q", partType)
		}
	}
	return blocks, nil
}

func rawReasoningExtra(msg message) (json.RawMessage, error) {
	if msg.ReasoningDetails == nil {
		return nil, nil
	}
	data, err := json.Marshal(msg.ReasoningDetails)
	if err != nil {
		return nil, err
	}
	return data, nil
}

func messageReasoningMap(msg message) map[string]any {
	return map[string]any{
		"reasoning_summary": msg.ReasoningSummary,
		"reasoning_details": msg.ReasoningDetails,
		"reasoning_content": msg.ReasoningContent,
		"reasoning":         msg.Reasoning,
		"reasoning_text":    msg.ReasoningText,
	}
}

func findReasoning(m map[string]any, fields []string) string {
	for _, field := range fields {
		value := m[field]
		switch v := value.(type) {
		case string:
			if v != "" {
				return v
			}
		case map[string]any:
			if text, _ := v["text"].(string); text != "" {
				return text
			}
		case []any:
			if text := reasoningDetailsText(v); text != "" {
				return text
			}
		}
	}
	return ""
}

func reasoningDetailsText(details []any) string {
	var out string
	for _, item := range details {
		var text string
		switch v := item.(type) {
		case string:
			text = v
		case map[string]any:
			text, _ = v["text"].(string)
		}
		if text == "" {
			continue
		}
		if out != "" {
			out += "\n\n"
		}
		out += text
	}
	return out
}

func (p *Provider) reasoningFields(stream bool) []string {
	fields := p.spec.Response.ReasoningFields
	if stream && len(p.spec.Stream.ReasoningFields) > 0 {
		fields = p.spec.Stream.ReasoningFields
	}
	if len(fields) > 0 {
		return fields
	}
	return []string{"reasoning_summary", "reasoning_details", "reasoning_content", "reasoning", "reasoning_text"}
}
