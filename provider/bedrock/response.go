package bedrock

import (
	"github.com/voocel/litellm"
	"github.com/voocel/litellm/internal/wire"
)

func convertResponse(resp *response, model string) *litellm.Response {
	out := &litellm.Response{
		Model:           model,
		Provider:        "bedrock",
		FinishReason:    wire.FinishReason(resp.StopReason),
		FinishReasonRaw: resp.StopReason,
		Usage:           convertUsage(resp.Usage),
	}
	for _, c := range resp.Output.Message.Content {
		switch {
		case c.Text != "":
			out.Blocks = append(out.Blocks, litellm.TextBlock{Text: c.Text})
		case c.ReasoningContent != nil:
			out.Blocks = append(out.Blocks, convertReasoning(c.ReasoningContent, model))
		case c.ToolUse != nil:
			out.Blocks = append(out.Blocks, litellm.ToolUseBlock{ID: c.ToolUse.ToolUseID, Name: c.ToolUse.Name, Arguments: c.ToolUse.Input})
		}
	}
	return out
}

func convertReasoning(r *reasoningContent, model string) litellm.ReasoningBlock {
	state := reasoningState{RedactedContent: r.RedactedContent}
	var text string
	if r.ReasoningText != nil {
		text, state.Signature = r.ReasoningText.Text, r.ReasoningText.Signature
	}
	return litellm.ReasoningBlock{Text: text, State: wire.NewState("bedrock", model, state)}
}

// convertUsage reports input as the total: Bedrock counts cache reads and
// writes separately from uncached input.
func convertUsage(u usage) litellm.Usage {
	input := wire.AddTokenDetails(u.InputTokens, u.CacheReadInputTokens, u.CacheWriteInputTokens)
	return litellm.Usage{
		InputTokens:      input,
		OutputTokens:     u.OutputTokens,
		TotalTokens:      wire.SumTokens(input, u.OutputTokens),
		CacheReadTokens:  u.CacheReadInputTokens,
		CacheWriteTokens: u.CacheWriteInputTokens,
	}
}
