package gemini

import (
	"crypto/rand"
	"fmt"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/internal/wire"
)

// convertResponse maps the first candidate. A blocked prompt or candidate is
// a Response with its finish reason, not an error.
func convertResponse(resp *response, provider, model string) *litellm.Response {
	out := &litellm.Response{Provider: provider, Model: model}
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
	for _, p := range candidate.Content.Parts {
		switch b := partBlock(p, provider, model).(type) {
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
				out.Warnings = append(out.Warnings, generatedIDWarning(provider, b))
			}
			out.Blocks = append(out.Blocks, b)
		}
	}
	out.FinishReason, out.FinishReasonRaw = wire.FinishReason(candidate.FinishReason), candidate.FinishReason
	if candidate.FinishMessage != "" {
		out.Warnings = append(out.Warnings, finishMessageWarning(provider, candidate.FinishMessage))
	}
	return out
}

// partBlock maps one part for the requested model; its thought signature
// becomes the block's State, of provider. Callers merge adjacent text or thought parts into
// one block only when neither carries a signature: signed parts must retain
// their original boundaries when replayed.
// A function call without an id gets a generated one.
func partBlock(p part, provider, model string) litellm.Block {
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
		args := string(p.FunctionCall.Args)
		if args == "" {
			args = "{}" // args is optional on the wire
		}
		return litellm.ToolUseBlock{ID: id, Name: p.FunctionCall.Name, Arguments: args, State: signed(provider, model, p.ThoughtSignature)}
	case p.Thought:
		return litellm.ReasoningBlock{Text: text, State: signed(provider, model, p.ThoughtSignature)}
	case text != "" || p.ThoughtSignature != "":
		// Streams may send a text run's signature in an empty text part.
		return litellm.TextBlock{Text: text, State: signed(provider, model, p.ThoughtSignature)}
	}
	return nil
}

// signatureState is the ProviderState of a part Gemini signed.
type signatureState struct {
	ThoughtSignature string `json:"thoughtSignature"`
}

// signed returns the State for a thought signature, nil when there is none.
func signed(provider, model, signature string) *litellm.ProviderState {
	if signature == "" {
		return nil
	}
	return wire.NewState(provider, model, signatureState{signature})
}

// signature returns the thought signature of a block provider produced.
func signature(state *litellm.ProviderState, provider string) string {
	s, _ := wire.ReadState[signatureState](state, provider)
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
func finishMessageWarning(provider, message string) litellm.Warning {
	return litellm.Warning{Code: "gemini.finish_message", Provider: provider, Message: message}
}

func generatedIDWarning(provider string, tool litellm.ToolUseBlock) litellm.Warning {
	return litellm.Warning{Code: "gemini.tool_call_id_generated", Provider: provider, Message: fmt.Sprintf("function call %q had no id; generated %q", tool.Name, tool.ID)}
}

// convertUsage reads omitted counts as zero.
func convertUsage(u *usageMetadata) litellm.Usage {
	return litellm.Usage{
		InputTokens:     u.PromptTokenCount,
		OutputTokens:    u.CandidatesTokenCount + u.ThoughtsTokenCount,
		ReasoningTokens: u.ThoughtsTokenCount,
		CacheReadTokens: u.CachedContentTokenCount,
	}
}
