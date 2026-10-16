package gemini

import (
	"crypto/rand"
	"encoding/base64"
	"fmt"
	"strings"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/internal/wire"
)

// convertResponse maps the first candidate. A blocked prompt or candidate is
// a Response with its finish reason, not an error.
func convertResponse(resp *response, provider, model string) (*litellm.Response, error) {
	out := &litellm.Response{Provider: provider, Model: model}
	if resp.UsageMetadata != nil {
		out.Usage = convertUsage(resp.UsageMetadata)
	}
	if len(resp.Candidates) == 0 {
		if resp.PromptFeedback != nil && resp.PromptFeedback.BlockReason != "" {
			out.FinishReason, out.FinishReasonRaw = wire.FinishReason(resp.PromptFeedback.BlockReason), resp.PromptFeedback.BlockReason
		}
		return out, nil
	}
	candidate := resp.Candidates[0]
	for _, p := range candidate.Content.Parts {
		block, err := partBlock(p, provider, model)
		if err != nil {
			return nil, err
		}
		switch b := block.(type) {
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
		case litellm.ImageBlock:
			out.Blocks = append(out.Blocks, b)
		case nil:
			if w, ok := unsupportedData(provider, p); ok {
				out.Warnings = append(out.Warnings, w)
			}
		}
	}
	out.FinishReason, out.FinishReasonRaw = wire.FinishReason(candidate.FinishReason), candidate.FinishReason
	if w, ok := finishMessage(provider, candidate); ok {
		out.Warnings = append(out.Warnings, w)
	}
	return out, nil
}

// partBlock maps one part for the requested model; its thought signature
// becomes the block's State, of provider. A function call without an id gets
// a generated one. A part with no block, such as a thought image, is nil.
// Callers merge adjacent text or thought parts into one block only when
// neither carries a signature: signed parts must retain their original
// boundaries when replayed.
func partBlock(p part, provider, model string) (litellm.Block, error) {
	var text string
	if p.Text != nil {
		text = *p.Text
	}
	switch {
	case p.InlineData != nil:
		// Thought images are unsigned drafts of the final image; inline data
		// other than an image has no block.
		if p.Thought || !strings.HasPrefix(p.InlineData.MimeType, "image/") {
			return nil, nil
		}
		data, err := base64.StdEncoding.DecodeString(p.InlineData.Data)
		if err != nil {
			return nil, fmt.Errorf("decode inline image: %w", err)
		}
		return litellm.ImageBlock{Data: data, MIME: p.InlineData.MimeType, State: signed(provider, model, p.ThoughtSignature)}, nil
	case p.FunctionCall != nil:
		id := p.FunctionCall.ID
		if id == "" {
			id = "call_" + rand.Text() // unique across processes, as persisted history needs
		}
		return litellm.ToolUseBlock{ID: id, Name: p.FunctionCall.Name, Arguments: string(p.FunctionCall.Args), State: signed(provider, model, p.ThoughtSignature)}, nil
	case text == "" && p.ThoughtSignature == "":
		// An empty unsigned thought or text part carries nothing.
		return nil, nil
	case p.Thought:
		return litellm.ReasoningBlock{Text: text, State: signed(provider, model, p.ThoughtSignature)}, nil
	default:
		// Streams may send a text run's signature in an empty text part.
		return litellm.TextBlock{Text: text, State: signed(provider, model, p.ThoughtSignature)}, nil
	}
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

// finishMessage keeps the vendor's explanation of a finish other than STOP,
// such as the rejected call of MALFORMED_FUNCTION_CALL. A STOP needs none,
// whatever its message, such as "Model generated function call(s)."
func finishMessage(provider string, c candidate) (litellm.Warning, bool) {
	if c.FinishMessage == "" || c.FinishReason == "STOP" {
		return litellm.Warning{}, false
	}
	return litellm.Warning{Code: "gemini.finish_message", Provider: provider, Message: c.FinishMessage}, true
}

// unsupportedData reports the inline data of a part that has no block, such
// as a speech model's audio. A thought image, a draft of the final one, is
// dropped without a warning.
func unsupportedData(provider string, p part) (litellm.Warning, bool) {
	if p.InlineData == nil || p.Thought {
		return litellm.Warning{}, false
	}
	return litellm.Warning{Code: "gemini.unsupported_part", Provider: provider, Message: fmt.Sprintf("dropped inline data of type %q, which litellm does not model", p.InlineData.MimeType)}, true
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
