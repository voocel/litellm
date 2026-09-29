package openaicompat

import (
	"cmp"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/internal/wire"
)

func (p *Provider) convertResponse(resp *chatResponse, req *litellm.Request) (*litellm.Response, error) {
	out := &litellm.Response{
		Provider: p.Name(),
		Model:    req.Model,
		Usage:    convertUsage(resp.Usage),
	}
	if resp.Model != "" {
		out.Model = resp.Model
	}
	if len(resp.Choices) == 0 {
		return out, nil
	}
	choice := resp.Choices[0]
	out.FinishReason = wire.FinishReason(choice.FinishReason)
	out.FinishReasonRaw = choice.FinishReason
	if block, ok := p.reasoningBlock(choice.Message.Fields, req.Model); ok {
		out.Blocks = append(out.Blocks, block)
	}
	blocks, refused, err := contentBlocks(choice.Message.Content)
	if err != nil {
		return nil, fmt.Errorf("convert response content: %w", err)
	}
	out.Blocks = append(out.Blocks, blocks...)
	if choice.Message.Refusal != "" {
		out.Blocks = append(out.Blocks, litellm.TextBlock{Text: choice.Message.Refusal})
		refused = true
	}
	if refused {
		out.FinishReason = litellm.FinishReasonSafety
	}
	// Chat Completions attaches metadata to the message and choice, not to
	// content parts. Keep part-level metadata for compatible vendors too.
	for i, block := range out.Blocks {
		if text, ok := block.(litellm.TextBlock); ok {
			text.Annotations = append(text.Annotations, Annotations(choice.Message.Annotations)...)
			if len(choice.Logprobs) > 0 && string(choice.Logprobs) != "null" {
				text.Logprobs = choice.Logprobs
			}
			out.Blocks[i] = text
			break
		}
	}
	for _, call := range choice.Message.ToolCalls {
		out.Blocks = append(out.Blocks, litellm.ToolUseBlock{
			ID:        call.ID,
			Name:      call.Function.Name,
			Arguments: json.RawMessage(cmp.Or(call.Function.Arguments, "{}")), // as streams deliver an argument-less call
		})
	}
	return out, nil
}

// reasoningBlock reads the first non-empty reasoning field the spec names.
// reasoning_details is also kept verbatim as the State.
func (p *Provider) reasoningBlock(message map[string]json.RawMessage, model string) (litellm.ReasoningBlock, bool) {
	var block litellm.ReasoningBlock
	for _, field := range p.spec.ReasoningFields {
		raw := message[field]
		if field == "reasoning_details" && len(raw) > 0 && string(raw) != "null" {
			block.State = wire.NewState(p.spec.Name, model, append(json.RawMessage(nil), raw...))
		}
		if block.Text == "" {
			block.Text = reasoningText(raw)
		}
	}
	return block, block.Text != "" || block.State != nil
}

// reasoningText accepts a string, an object with text or summary, or an array
// of either, as vendors use all three.
func reasoningText(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var text string
	if json.Unmarshal(raw, &text) == nil {
		return text
	}
	var item struct {
		Text    string `json:"text"`
		Summary string `json:"summary"`
	}
	if json.Unmarshal(raw, &item) == nil {
		if item.Text != "" {
			return item.Text
		}
		return item.Summary
	}
	var items []json.RawMessage
	if json.Unmarshal(raw, &items) != nil {
		return ""
	}
	parts := make([]string, 0, len(items))
	for _, item := range items {
		if text := reasoningText(item); text != "" {
			parts = append(parts, text)
		}
	}
	return strings.Join(parts, "\n\n")
}

func convertUsage(u usage) litellm.Usage {
	out := litellm.Usage{
		InputTokens:  u.PromptTokens,
		OutputTokens: u.CompletionTokens,
		TotalTokens:  u.TotalTokens,
	}
	if u.PromptTokensDetails != nil {
		out.CacheReadTokens = u.PromptTokensDetails.CachedTokens
		out.CacheWriteTokens = u.PromptTokensDetails.CacheWriteTokens
	}
	if out.CacheReadTokens == nil {
		out.CacheReadTokens = u.PromptCacheHitTokens
	}
	// DeepSeek splits the prompt into cache hits and misses, billing misses as
	// input: nothing is written to a separately priced cache.
	if out.CacheWriteTokens == nil && u.PromptCacheMissTokens != nil {
		out.CacheWriteTokens = new(0)
	}
	if u.CompletionTokensDetails != nil {
		out.ReasoningTokens = u.CompletionTokensDetails.ReasoningTokens
	}
	return out
}

// contentBlocks converts message content, a string or an array of parts. It
// reports whether a refusal part was present.
func contentBlocks(raw json.RawMessage) ([]litellm.Block, bool, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, false, nil
	}
	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		if text == "" {
			return nil, false, nil
		}
		return []litellm.Block{litellm.TextBlock{Text: text}}, false, nil
	}
	var parts []contentPart
	if err := json.Unmarshal(raw, &parts); err != nil {
		return nil, false, fmt.Errorf("unsupported content payload: %w", err)
	}
	blocks := make([]litellm.Block, 0, len(parts))
	var refused bool
	for _, part := range parts {
		switch part.Type {
		case "text":
			if part.Text != "" {
				blocks = append(blocks, litellm.TextBlock{Text: part.Text, Annotations: Annotations(part.Annotations), Logprobs: part.Logprobs})
			}
		case "refusal":
			if part.Refusal != "" {
				blocks = append(blocks, litellm.TextBlock{Text: part.Refusal})
				refused = true
			}
		default:
			return nil, false, fmt.Errorf("unsupported content part type %q", part.Type)
		}
	}
	return blocks, refused, nil
}

// Annotations converts OpenAI content annotations, keeping each verbatim in
// Extra.
func Annotations(raw []json.RawMessage) []litellm.Annotation {
	if len(raw) == 0 {
		return nil
	}
	out := make([]litellm.Annotation, 0, len(raw))
	for _, entry := range raw {
		var fields struct {
			Type        string `json:"type"`
			Text        string `json:"text"`
			URL         string `json:"url"`
			URLCitation *struct {
				URL   string `json:"url"`
				Title string `json:"title"`
			} `json:"url_citation"`
		}
		_ = json.Unmarshal(entry, &fields)
		if fields.URLCitation != nil {
			fields.URL, fields.Text = fields.URLCitation.URL, fields.URLCitation.Title
		}
		out = append(out, litellm.Annotation{Type: fields.Type, Text: fields.Text, URL: fields.URL, Extra: entry})
	}
	return out
}
