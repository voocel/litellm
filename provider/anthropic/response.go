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

type usage struct {
	InputTokens              *int `json:"input_tokens"`
	OutputTokens             *int `json:"output_tokens"`
	CacheCreationInputTokens *int `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     *int `json:"cache_read_input_tokens"`
}

func convertResponse(resp *response, model string) *litellm.Response {
	out := &litellm.Response{
		Model:           model,
		Provider:        "anthropic",
		FinishReason:    wire.FinishReason(resp.StopReason),
		FinishReasonRaw: resp.StopReason,
		Usage:           convertUsage(resp.Usage),
	}
	if resp.Model != "" {
		out.Model = resp.Model
	}
	for _, c := range resp.Content {
		if block, ok := convertContent(c); ok {
			out.Blocks = append(out.Blocks, block)
		} else {
			out.Warnings = append(out.Warnings, unsupportedBlock(c.Type))
		}
	}
	return out
}

// convertContent maps a response content block. Blocks litellm does not model,
// such as server tool calls, are reported by unsupportedBlock.
func convertContent(c content) (litellm.Block, bool) {
	switch c.Type {
	case "text":
		return litellm.TextBlock{Text: c.Text, Annotations: annotations(c.Citations)}, true
	case "thinking":
		var text string
		if c.Thinking != nil {
			text = *c.Thinking
		}
		return litellm.ReasoningBlock{Text: text, Signature: c.Signature}, true
	case "redacted_thinking":
		return litellm.ReasoningBlock{Redacted: []byte(c.Data)}, true
	case "tool_use":
		return litellm.ToolUseBlock{ID: c.ID, Name: c.Name, Arguments: c.Input}, true
	}
	return nil, false
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

func unsupportedBlock(blockType string) litellm.Warning {
	return litellm.Warning{Code: "anthropic.unsupported_block", Provider: "anthropic", Message: fmt.Sprintf("dropped content block %q, which litellm does not model", blockType)}
}

// convertUsage reports input as the total: Anthropic counts cache reads and
// writes separately from uncached input.
func convertUsage(u usage) litellm.Usage {
	input := wire.AddTokenDetails(u.InputTokens, u.CacheReadInputTokens, u.CacheCreationInputTokens)
	return litellm.Usage{
		InputTokens:      input,
		OutputTokens:     u.OutputTokens,
		TotalTokens:      wire.SumTokens(input, u.OutputTokens),
		CacheReadTokens:  u.CacheReadInputTokens,
		CacheWriteTokens: u.CacheCreationInputTokens,
	}
}
