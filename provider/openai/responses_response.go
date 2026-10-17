package openai

import (
	"cmp"
	"encoding/json"
	"strings"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/internal/wire"
	"github.com/voocel/litellm/provider/internal/openaicompat"
)

type responsesResponse struct {
	Model             string                `json:"model"`
	Status            string                `json:"status"`
	Store             *bool                 `json:"store"`
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
	Status    string                 `json:"status"`
	Phase     string                 `json:"phase"`
	CallID    string                 `json:"call_id"`
	Name      string                 `json:"name"`
	Arguments string                 `json:"arguments"`
	Content   []responsesContentPart `json:"content"`
	Summary   []struct {
		Text string `json:"text"`
	} `json:"summary"`
	// Result is the image of an image generation call, in its OutputFormat;
	// Error explains a failed call.
	Result       []byte `json:"result"`
	OutputFormat string `json:"output_format"`
	Error        *struct {
		Message string `json:"message"`
	} `json:"error"`
	// Raw is the item as received, kept for reasoning items, whose State it
	// is.
	Raw json.RawMessage `json:"-"`
}

// itemState is the State of text, function calls and generated images: the
// id of their output item, which replay pairs with the reasoning before it or,
// for an image, sends alone, and a message's phase.
type itemState struct {
	ID    string `json:"id,omitempty"`
	Phase string `json:"phase,omitempty"`
}

// state returns the State of a block from an output item with the given id
// and phase, nil when there are neither.
func (s itemState) state(provider, model string) *litellm.ProviderState {
	if s == (itemState{}) {
		return nil
	}
	return wire.NewState(provider, model, s)
}

// stored reports whether the API keeps the response, as it does unless it
// says otherwise: store defaults to true.
func (r *responsesResponse) stored() bool {
	return r.Store == nil || *r.Store
}

func (i *responsesOutputItem) UnmarshalJSON(data []byte) error {
	type plain responsesOutputItem
	if err := json.Unmarshal(data, (*plain)(i)); err != nil {
		return err
	}
	if i.Type == "reasoning" {
		i.Raw = append(json.RawMessage(nil), data...)
	}
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
	InputTokens        int `json:"input_tokens"`
	OutputTokens       int `json:"output_tokens"`
	InputTokensDetails struct {
		CachedTokens     int `json:"cached_tokens"`
		CacheWriteTokens int `json:"cache_write_tokens"`
	} `json:"input_tokens_details"`
	OutputTokensDetails struct {
		ReasoningTokens int `json:"reasoning_tokens"`
	} `json:"output_tokens_details"`
}

// convertResponsesResponse maps output items in order. Item types without a
// Block equivalent, such as hosted tool calls, remain available in Raw.
func convertResponsesResponse(resp *responsesResponse, provider, model string) *litellm.Response {
	out := &litellm.Response{Provider: provider, Model: model, Usage: convertResponsesUsage(resp.Usage)}
	if resp.Model != "" {
		out.Model = resp.Model
	}
	var refused bool
	for _, item := range resp.Output {
		switch item.Type {
		case "message":
			for _, part := range item.Content {
				if block, refusal, ok := contentPartBlock(part); ok {
					block.State = itemState{item.ID, item.Phase}.state(provider, model)
					out.Blocks = append(out.Blocks, block)
					refused = refused || refusal
				}
			}
		case "function_call":
			out.Blocks = append(out.Blocks, litellm.ToolUseBlock{
				ID: item.CallID, Name: item.Name, Arguments: item.Arguments,
				State: itemState{ID: item.ID}.state(provider, model),
			})
		case "reasoning":
			out.Blocks = append(out.Blocks, reasoningBlock(item, provider, model))
		case "image_generation_call":
			if image, ok := generatedImage(item, provider, model, resp.stored()); ok {
				out.Blocks = append(out.Blocks, image)
			} else if item.Status == "failed" {
				out.Warnings = append(out.Warnings, imageFailure(provider, item))
			}
		}
	}
	reason := ""
	if resp.IncompleteDetails != nil {
		reason = resp.IncompleteDetails.Reason
	}
	out.FinishReason, out.FinishReasonRaw = finish(resp.Status, reason, refused)
	return out
}

// reasoningBlock uses the summary, or the raw reasoning text some models
// return instead. The item is its State.
func reasoningBlock(item responsesOutputItem, provider, model string) litellm.ReasoningBlock {
	var summaries, texts []string
	for _, summary := range item.Summary {
		summaries = append(summaries, summary.Text)
	}
	for _, part := range item.Content {
		if part.Type == "reasoning_text" {
			texts = append(texts, part.Text)
		}
	}
	state := wire.NewState(provider, model, item.Raw)
	if len(summaries) == 0 && len(texts) > 0 {
		return litellm.ReasoningBlock{Text: strings.Join(texts, ""), State: state}
	}
	return litellm.ReasoningBlock{Text: strings.Join(summaries, "\n"), Summary: true, State: state}
}

// generatedImage maps the image of an image generation call; a call without
// one, such as a failed call, has none. The call's id is its State only when
// the API stored the response, where replay finds the image by that id.
func generatedImage(item responsesOutputItem, provider, model string, stored bool) (litellm.ImageBlock, bool) {
	if len(item.Result) == 0 {
		return litellm.ImageBlock{}, false
	}
	image := litellm.ImageBlock{Data: item.Result, MIME: "image/" + cmp.Or(item.OutputFormat, "png")}
	if stored {
		image.State = itemState{ID: item.ID}.state(provider, model)
	}
	return image, true
}

// imageFailure reports an image generation call that failed, such as one
// moderation blocked.
func imageFailure(provider string, item responsesOutputItem) litellm.Warning {
	message := "image generation failed"
	if item.Error != nil {
		message += ": " + item.Error.Message
	}
	return litellm.Warning{Code: "openai.image_generation_failed", Provider: provider, Message: message}
}

func contentPartBlock(part responsesContentPart) (litellm.TextBlock, bool, bool) {
	switch part.Type {
	case "output_text":
		return litellm.TextBlock{Text: part.Text, Annotations: openaicompat.Annotations(part.Annotations), Logprobs: part.Logprobs}, false, true
	case "refusal":
		return litellm.TextBlock{Text: part.Refusal}, true, true
	}
	return litellm.TextBlock{}, false, false
}

// finish derives the finish reason shared by responses and streams.
func finish(status, incompleteReason string, refused bool) (litellm.FinishReason, string) {
	raw := cmp.Or(incompleteReason, status)
	if refused {
		return litellm.FinishReasonSafety, raw
	}
	return wire.FinishReason(raw), raw
}

func convertResponsesUsage(u responsesUsage) litellm.Usage {
	return litellm.Usage{
		InputTokens:      u.InputTokens,
		OutputTokens:     u.OutputTokens,
		ReasoningTokens:  u.OutputTokensDetails.ReasoningTokens,
		CacheReadTokens:  u.InputTokensDetails.CachedTokens,
		CacheWriteTokens: u.InputTokensDetails.CacheWriteTokens,
	}
}
