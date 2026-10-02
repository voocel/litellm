package anthropic

import (
	"encoding/json"
	"fmt"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/internal/wire"
)

type response struct {
	Content    []content `json:"content"`
	Usage      usage     `json:"usage"`
	Model      string    `json:"model"`
	StopReason string    `json:"stop_reason"`
}

// usage keeps omitted counts distinct from zero: a stream's message_delta
// updates only the counts it carries.
type usage struct {
	InputTokens              *int `json:"input_tokens"`
	OutputTokens             *int `json:"output_tokens"`
	CacheCreationInputTokens *int `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     *int `json:"cache_read_input_tokens"`
}

func convertResponse(resp *response, provider, model string) *litellm.Response {
	out := &litellm.Response{
		Model:           model,
		Provider:        provider,
		FinishReason:    wire.FinishReason(resp.StopReason),
		FinishReasonRaw: resp.StopReason,
		Usage:           convertUsage(resp.Usage),
	}
	if resp.Model != "" {
		out.Model = resp.Model
	}
	for _, c := range resp.Content {
		if block, ok := convertContent(c, provider, model); ok {
			out.Blocks = append(out.Blocks, block)
		} else {
			out.Warnings = append(out.Warnings, unsupportedBlock(provider, c.Type))
		}
	}
	return out
}

// convertContent maps a response content block for the requested model.
// Blocks litellm does not model, such as server tool calls, are reported by
// unsupportedBlock.
func convertContent(c content, provider, model string) (litellm.Block, bool) {
	switch c.Type {
	case "text":
		return litellm.TextBlock{Text: c.Text, Annotations: annotations(c.Citations)}, true
	case "thinking", "redacted_thinking":
		var text string
		if c.Thinking != nil {
			text = *c.Thinking
		}
		return litellm.ReasoningBlock{Text: text, State: reasoningState(provider, model, c.Type, c.Signature, c.Data)}, true
	case "tool_use":
		return litellm.ToolUseBlock{ID: c.ID, Name: c.Name, Arguments: string(c.Input)}, true
	}
	return nil, false
}

// reasoningState returns the State of a thinking block, nil until it has the
// signature or redacted data that makes it replayable: a stream cut short
// before the signature leaves thinking that cannot be sent back.
func reasoningState(provider, model, blockType, signature, data string) *litellm.ProviderState {
	if signature == "" && data == "" {
		return nil
	}
	return wire.NewState(provider, model, thinkingState{Type: blockType, Signature: signature, Data: data})
}

// annotations maps citations, keeping each verbatim in Extra.
func annotations(citations []json.RawMessage) []litellm.Annotation {
	if len(citations) == 0 {
		return nil
	}
	out := make([]litellm.Annotation, 0, len(citations))
	for _, raw := range citations {
		var c struct {
			Type      string `json:"type"`
			CitedText string `json:"cited_text"`
			URL       string `json:"url"`
		}
		_ = json.Unmarshal(raw, &c)
		out = append(out, litellm.Annotation{Type: c.Type, Text: c.CitedText, URL: c.URL, Extra: raw})
	}
	return out
}

func unsupportedBlock(provider, blockType string) litellm.Warning {
	return litellm.Warning{Code: "anthropic.unsupported_block", Provider: provider, Message: fmt.Sprintf("dropped content block %q, which litellm does not model", blockType)}
}

// convertUsage reports input as the total: Anthropic counts cache reads and
// writes separately from uncached input.
func convertUsage(u usage) litellm.Usage {
	read, write := count(u.CacheReadInputTokens), count(u.CacheCreationInputTokens)
	return litellm.Usage{
		InputTokens:      count(u.InputTokens) + read + write,
		OutputTokens:     count(u.OutputTokens),
		CacheReadTokens:  read,
		CacheWriteTokens: write,
	}
}

// count reads an omitted count as zero.
func count(n *int) int {
	if n == nil {
		return 0
	}
	return *n
}
