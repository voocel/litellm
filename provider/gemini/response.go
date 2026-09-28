package gemini

import (
	"crypto/rand"
	"encoding/json"
	"fmt"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/internal/wire"
)

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
		switch b := partBlock(p, model).(type) {
		case litellm.TextBlock:
			if last, ok := lastBlock[litellm.TextBlock](out.Blocks); ok && last.State == nil && b.State == nil {
				last.Text += b.Text
				out.Blocks[len(out.Blocks)-1] = last
				continue
			}
			out.Blocks = append(out.Blocks, b)
		case litellm.ReasoningBlock:
			if last, ok := lastBlock[litellm.ReasoningBlock](out.Blocks); ok && last.State == nil && b.State == nil {
				last.Text += b.Text
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
	if candidate.FinishMessage != "" {
		out.Warnings = append(out.Warnings, finishMessageWarning(candidate.FinishMessage))
	}
	return out
}

// partBlock maps one part for the requested model; its thought signature
// becomes the block's State. Callers merge adjacent text or thought parts into
// one block only when neither carries a signature: signed parts must retain
// their original boundaries when replayed.
// A function call without an id gets a generated one.
func partBlock(p part, model string) litellm.Block {
	var text string
	if p.Text != nil {
		text = *p.Text
	}
	switch {
	case p.FunctionCall != nil:
		id := p.FunctionCall.ID
		if id == "" {
			id = "call_" + rand.Text() // unique across processes, as persisted history needs
		}
		args := p.FunctionCall.Args
		if len(args) == 0 {
			args = json.RawMessage("{}") // args is optional on the wire
		}
		return litellm.ToolUseBlock{ID: id, Name: p.FunctionCall.Name, Arguments: args, State: signed(model, p.ThoughtSignature)}
	case p.Thought:
		return litellm.ReasoningBlock{Text: text, State: signed(model, p.ThoughtSignature)}
	case text != "" || p.ThoughtSignature != "":
		// Streams may send a text run's signature in an empty text part.
		return litellm.TextBlock{Text: text, State: signed(model, p.ThoughtSignature)}
	}
	return nil
}

// signatureState is the ProviderState of a part Gemini signed.
type signatureState struct {
	ThoughtSignature string `json:"thoughtSignature"`
}

// signed returns the State for a thought signature, nil when there is none.
func signed(model, signature string) *litellm.ProviderState {
	if signature == "" {
		return nil
	}
	return wire.NewState("gemini", model, signatureState{signature})
}

// signature returns the thought signature of a block Gemini produced.
func signature(state *litellm.ProviderState) string {
	s, _ := wire.ReadState[signatureState](state, "gemini")
	return s.ThoughtSignature
}

func lastBlock[T litellm.Block](blocks []litellm.Block) (T, bool) {
	var zero T
	if len(blocks) == 0 {
		return zero, false
	}
	last, ok := blocks[len(blocks)-1].(T)
	return last, ok
}

// finishMessageWarning keeps the vendor's explanation of the finish reason,
// such as the rejected call of MALFORMED_FUNCTION_CALL.
func finishMessageWarning(message string) litellm.Warning {
	return litellm.Warning{Code: "gemini.finish_message", Provider: "gemini", Message: message}
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

// convertUsage reads omitted counts as zero. Empty metadata reports nothing.
func convertUsage(u *usageMetadata) litellm.Usage {
	if *u == (usageMetadata{}) {
		return litellm.Usage{}
	}
	return litellm.Usage{
		InputTokens:     new(u.PromptTokenCount),
		OutputTokens:    new(u.CandidatesTokenCount + u.ThoughtsTokenCount),
		ReasoningTokens: new(u.ThoughtsTokenCount),
		TotalTokens:     new(u.TotalTokenCount),
		CacheReadTokens: new(u.CachedContentTokenCount),
	}
}
