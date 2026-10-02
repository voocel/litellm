package bedrock

import (
	"github.com/voocel/litellm"
	"github.com/voocel/litellm/internal/wire"
)

func convertResponse(resp *response, provider, model string) *litellm.Response {
	out := &litellm.Response{
		Model:           model,
		Provider:        provider,
		FinishReason:    wire.FinishReason(resp.StopReason),
		FinishReasonRaw: resp.StopReason,
		Usage:           convertUsage(resp.Usage),
	}
	for _, c := range resp.Output.Message.Content {
		switch {
		case c.Text != "":
			out.Blocks = append(out.Blocks, litellm.TextBlock{Text: c.Text})
		case c.ReasoningContent != nil:
			out.Blocks = append(out.Blocks, convertReasoning(c.ReasoningContent, provider, model))
		case c.ToolUse != nil:
			out.Blocks = append(out.Blocks, litellm.ToolUseBlock{ID: c.ToolUse.ToolUseID, Name: c.ToolUse.Name, Arguments: string(c.ToolUse.Input)})
		}
	}
	return out
}

func convertReasoning(r *reasoningContent, provider, model string) litellm.ReasoningBlock {
	state := reasoningState{RedactedContent: r.RedactedContent}
	var text string
	if r.ReasoningText != nil {
		text, state.Signature = r.ReasoningText.Text, r.ReasoningText.Signature
	}
	return litellm.ReasoningBlock{Text: text, State: wire.NewState(provider, model, state)}
}

// convertUsage reports input as the total: Bedrock counts cache reads and
// writes separately from uncached input.
func convertUsage(u usage) litellm.Usage {
	return litellm.Usage{
		InputTokens:      u.InputTokens + u.CacheReadInputTokens + u.CacheWriteInputTokens,
		OutputTokens:     u.OutputTokens,
		CacheReadTokens:  u.CacheReadInputTokens,
		CacheWriteTokens: u.CacheWriteInputTokens,
	}
}
