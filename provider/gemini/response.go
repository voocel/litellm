package gemini

import (
	"encoding/json"
	"fmt"
	"sync/atomic"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/internal/wire"
)

var generatedToolCallSeq atomic.Uint64

// convertResponse maps the first candidate. A blocked prompt or candidate is
// a Response with its finish reason, not an error.
func convertResponse(resp *response, model string) *litellm.Response {
	out := &litellm.Response{Provider: "gemini", Model: model}
	if resp.UsageMetadata != nil {
		out.Usage = convertUsage(resp.UsageMetadata)
	}
	if len(resp.Candidates) == 0 {
		if resp.PromptFeedback != nil && resp.PromptFeedback.BlockReason != "" {
			out.FinishReason, out.FinishReasonRaw = wire.FinishReason(resp.PromptFeedback.BlockReason), resp.PromptFeedback.BlockReason
		}
		return out
	}
	candidate := resp.Candidates[0]
	var toolCalls bool
	for _, p := range candidate.Content.Parts {
		switch b := partBlock(p).(type) {
		case litellm.TextBlock:
			if last, ok := lastBlock[litellm.TextBlock](out.Blocks); ok {
				last.Text += b.Text
				out.Blocks[len(out.Blocks)-1] = last
				continue
			}
			out.Blocks = append(out.Blocks, b)
		case litellm.ReasoningBlock:
			if last, ok := lastBlock[litellm.ReasoningBlock](out.Blocks); ok {
				last.Text += b.Text
				if b.Signature != "" {
					last.Signature = b.Signature
				}
				out.Blocks[len(out.Blocks)-1] = last
				continue
			}
			out.Blocks = append(out.Blocks, b)
		case litellm.ToolUseBlock:
			if p.FunctionCall.ID == "" {
				out.Warnings = append(out.Warnings, generatedIDWarning(b))
			}
			out.Blocks = append(out.Blocks, b)
			toolCalls = true
		}
	}
	out.FinishReason, out.FinishReasonRaw = finishReason(candidate.FinishReason, toolCalls), candidate.FinishReason
	return out
}

// partBlock maps one part. Adjacent text or thought parts form one block, so
// callers merge them; a function call without an id gets a generated one.
func partBlock(p part) litellm.Block {
	switch {
	case p.FunctionCall != nil:
		id := p.FunctionCall.ID
		if id == "" {
			id = fmt.Sprintf("call_%d", generatedToolCallSeq.Add(1))
		}
		args := p.FunctionCall.Args
		if len(args) == 0 {
			args = json.RawMessage("{}") // args is optional on the wire
		}
		return litellm.ToolUseBlock{ID: id, Name: p.FunctionCall.Name, Arguments: args, Signature: p.ThoughtSignature}
	case p.Thought:
		return litellm.ReasoningBlock{Text: p.Text, Signature: p.ThoughtSignature}
	case p.Text != "":
		return litellm.TextBlock{Text: p.Text}
	}
	return nil
}

func lastBlock[T litellm.Block](blocks []litellm.Block) (T, bool) {
	var zero T
	if len(blocks) == 0 {
		return zero, false
	}
	last, ok := blocks[len(blocks)-1].(T)
	return last, ok
}

func generatedIDWarning(tool litellm.ToolUseBlock) litellm.Warning {
	return litellm.Warning{Code: "gemini.tool_call_id_generated", Provider: "gemini", Message: fmt.Sprintf("function call %q had no id; generated %q", tool.Name, tool.ID)}
}

// finishReason reports tool calls: Gemini ends function-call turns with STOP.
func finishReason(raw string, toolCalls bool) litellm.FinishReason {
	finish := wire.FinishReason(raw)
	if finish == litellm.FinishReasonStop && toolCalls {
		return litellm.FinishReasonToolCall
	}
	return finish
}

func convertUsage(u *usageMetadata) litellm.Usage {
	return litellm.Usage{
		InputTokens:     u.PromptTokenCount,
		OutputTokens:    wire.AddTokenDetails(u.CandidatesTokenCount, u.ThoughtsTokenCount),
		ReasoningTokens: u.ThoughtsTokenCount,
		TotalTokens:     u.TotalTokenCount,
		CacheReadTokens: u.CachedContentTokenCount,
	}
}
