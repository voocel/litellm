package bedrock

import (
	"encoding/json"
	"fmt"

	"github.com/voocel/litellm"
)

func convertResponse(resp *response, model string) (*litellm.Response, error) {
	if resp == nil {
		return nil, fmt.Errorf("bedrock: response cannot be nil")
	}
	out := &litellm.Response{
		Model:        model,
		Provider:     "bedrock",
		FinishReason: litellm.NormalizeFinishReason(resp.StopReason),
		Usage: litellm.Usage{
			InputTokens:      resp.Usage.InputTokens + resp.Usage.CacheReadInputTokens,
			OutputTokens:     resp.Usage.OutputTokens,
			TotalTokens:      resp.Usage.TotalTokens,
			CacheReadTokens:  resp.Usage.CacheReadInputTokens,
			CacheWriteTokens: resp.Usage.CacheWriteInputTokens,
			Provider:         "bedrock",
			Model:            model,
		},
	}
	for _, block := range resp.Output.Message.Content {
		if block.Text != "" {
			out.Blocks = append(out.Blocks, litellm.TextBlock{Text: block.Text})
		}
		if block.ToolUse != nil {
			args, err := json.Marshal(block.ToolUse.Input)
			if err != nil {
				return nil, fmt.Errorf("bedrock: marshal tool use %q arguments: %w", block.ToolUse.Name, err)
			}
			out.Blocks = append(out.Blocks, litellm.ToolUseBlock{
				ID:        block.ToolUse.ToolUseID,
				Name:      block.ToolUse.Name,
				Arguments: args,
			})
		}
	}
	return out, nil
}
