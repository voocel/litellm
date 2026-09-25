package openai

import (
	"encoding/json"
	"strings"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/internal/wire"
	"github.com/voocel/litellm/provider/internal/openaicompat"
)

type responsesResponse struct {
	Model             string                `json:"model"`
	Status            string                `json:"status"`
	Output            []responsesOutputItem `json:"output"`
	Usage             responsesUsage        `json:"usage"`
	IncompleteDetails *struct {
		Reason string `json:"reason"`
	} `json:"incomplete_details"`
	Error *struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	} `json:"error"`
}

type responsesOutputItem struct {
	ID        string                 `json:"id"`
	Type      string                 `json:"type"`
	CallID    string                 `json:"call_id"`
	Name      string                 `json:"name"`
	Arguments string                 `json:"arguments"`
	Content   []responsesContentPart `json:"content"`
	Summary   []struct {
		Text string `json:"text"`
	} `json:"summary"`
	// Raw is the item as received, replayed for reasoning items.
	Raw json.RawMessage `json:"-"`
}

func (i *responsesOutputItem) UnmarshalJSON(data []byte) error {
	type plain responsesOutputItem
	if err := json.Unmarshal(data, (*plain)(i)); err != nil {
		return err
	}
	i.Raw = append(json.RawMessage(nil), data...)
	return nil
}

type responsesContentPart struct {
	Type        string            `json:"type"`
	Text        string            `json:"text"`
	Refusal     string            `json:"refusal"`
	Annotations []json.RawMessage `json:"annotations"`
	Logprobs    json.RawMessage   `json:"logprobs"`
}

type responsesUsage struct {
	InputTokens        *int `json:"input_tokens"`
	OutputTokens       *int `json:"output_tokens"`
	TotalTokens        *int `json:"total_tokens"`
	InputTokensDetails *struct {
		CachedTokens     *int `json:"cached_tokens"`
		CacheWriteTokens *int `json:"cache_write_tokens"`
	} `json:"input_tokens_details"`
	OutputTokensDetails *struct {
		ReasoningTokens *int `json:"reasoning_tokens"`
	} `json:"output_tokens_details"`
}

// convertResponsesResponse maps output items in order. Item types without a
// Block equivalent, such as hosted tool calls, remain available in Raw.
func convertResponsesResponse(resp *responsesResponse, model string) *litellm.Response {
	out := &litellm.Response{Provider: "openai", Model: model, Usage: convertResponsesUsage(resp.Usage)}
	if resp.Model != "" {
		out.Model = resp.Model
	}
	var toolCalls, refused bool
	for _, item := range resp.Output {
		switch item.Type {
		case "message":
			for _, part := range item.Content {
				if block, refusal, ok := contentPartBlock(part); ok {
					out.Blocks = append(out.Blocks, block)
					refused = refused || refusal
				}
			}
		case "function_call":
			out.Blocks = append(out.Blocks, litellm.ToolUseBlock{ID: item.CallID, Name: item.Name, Arguments: json.RawMessage(item.Arguments)})
			toolCalls = true
		case "reasoning":
			out.Blocks = append(out.Blocks, reasoningBlock(item))
		}
	}
	reason := ""
	if resp.IncompleteDetails != nil {
		reason = resp.IncompleteDetails.Reason
	}
	out.FinishReason, out.FinishReasonRaw = finish(resp.Status, reason, toolCalls, refused)
	return out
}

// reasoningBlock uses the summary, or the raw reasoning text some models
// return instead. Extra keeps the item for replay.
func reasoningBlock(item responsesOutputItem) litellm.ReasoningBlock {
	var summaries, texts []string
	for _, summary := range item.Summary {
		summaries = append(summaries, summary.Text)
	}
	for _, part := range item.Content {
		if part.Type == "reasoning_text" {
			texts = append(texts, part.Text)
		}
	}
	if len(summaries) == 0 && len(texts) > 0 {
		return litellm.ReasoningBlock{Text: strings.Join(texts, ""), Extra: item.Raw}
	}
	return litellm.ReasoningBlock{Text: strings.Join(summaries, "\n"), Summary: true, Extra: item.Raw}
}

func contentPartBlock(part responsesContentPart) (litellm.Block, bool, bool) {
	switch part.Type {
	case "output_text":
		return litellm.TextBlock{Text: part.Text, Annotations: openaicompat.Annotations(part.Annotations), Logprobs: part.Logprobs}, false, true
	case "refusal":
		return litellm.TextBlock{Text: part.Refusal}, true, true
	}
	return nil, false, false
}

// finish derives the finish reason shared by responses and streams.
func finish(status, incompleteReason string, toolCalls, refused bool) (litellm.FinishReason, string) {
	raw := status
	if incompleteReason != "" {
		raw = incompleteReason
	}
	switch {
	case refused:
		return litellm.FinishReasonSafety, raw
	case status == "completed" && toolCalls:
		return litellm.FinishReasonToolCall, raw
	}
	return wire.FinishReason(raw), raw
}

func convertResponsesUsage(u responsesUsage) litellm.Usage {
	out := litellm.Usage{InputTokens: u.InputTokens, OutputTokens: u.OutputTokens, TotalTokens: u.TotalTokens}
	if u.InputTokensDetails != nil {
		out.CacheReadTokens = u.InputTokensDetails.CachedTokens
		out.CacheWriteTokens = u.InputTokensDetails.CacheWriteTokens
	}
	if u.OutputTokensDetails != nil {
		out.ReasoningTokens = u.OutputTokensDetails.ReasoningTokens
	}
	return out
}
