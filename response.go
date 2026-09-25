package litellm

import "encoding/json"

// Response is a complete model reply. Providers set Raw to the vendor body of
// a non-streaming reply; the Client keeps it only with WithCaptureRawResponse.
type Response struct {
	Blocks []Block
	Usage  Usage

	Model    string
	Provider string

	FinishReason    FinishReason
	FinishReasonRaw string
	Warnings        []Warning
	Raw             json.RawMessage
}

// Text concatenates the text blocks.
func (r *Response) Text() string {
	if r == nil {
		return ""
	}
	var out string
	for _, block := range r.Blocks {
		if text, ok := block.(TextBlock); ok {
			out += text.Text
		}
	}
	return out
}

// ToolCalls returns the tool use blocks in order.
func (r *Response) ToolCalls() []ToolUseBlock {
	if r == nil {
		return nil
	}
	var calls []ToolUseBlock
	for _, block := range r.Blocks {
		if call, ok := block.(ToolUseBlock); ok {
			calls = append(calls, call)
		}
	}
	return calls
}

// Reasoning concatenates the reasoning block texts.
func (r *Response) Reasoning() string {
	if r == nil {
		return ""
	}
	var out string
	for _, block := range r.Blocks {
		if reasoning, ok := block.(ReasoningBlock); ok {
			out += reasoning.Text
		}
	}
	return out
}

// Usage contains reported token counts. A nil count is unknown; a pointer to
// zero is a known zero. InputTokens includes cache reads and writes;
// OutputTokens includes reasoning. Detail counts are subsets, not additions.
type Usage struct {
	InputTokens      *int
	OutputTokens     *int
	TotalTokens      *int
	ReasoningTokens  *int
	CacheReadTokens  *int
	CacheWriteTokens *int
}

// Input returns InputTokens and whether it is known.
func (u Usage) Input() (int, bool) { return tokenCount(u.InputTokens) }

// Output returns OutputTokens and whether it is known.
func (u Usage) Output() (int, bool) { return tokenCount(u.OutputTokens) }

// Total returns TotalTokens and whether it is known.
func (u Usage) Total() (int, bool) { return tokenCount(u.TotalTokens) }

// Reasoning returns ReasoningTokens and whether it is known.
func (u Usage) Reasoning() (int, bool) { return tokenCount(u.ReasoningTokens) }

// CacheRead returns CacheReadTokens and whether it is known.
func (u Usage) CacheRead() (int, bool) { return tokenCount(u.CacheReadTokens) }

// CacheWrite returns CacheWriteTokens and whether it is known.
func (u Usage) CacheWrite() (int, bool) { return tokenCount(u.CacheWriteTokens) }

func tokenCount(count *int) (int, bool) {
	if count == nil {
		return 0, false
	}
	return *count, true
}

// HasTokens reports whether any token count is known, including a known zero.
func (u Usage) HasTokens() bool {
	return u.InputTokens != nil || u.OutputTokens != nil || u.TotalTokens != nil ||
		u.ReasoningTokens != nil || u.CacheReadTokens != nil || u.CacheWriteTokens != nil
}

// Clone returns an independent copy of the reported counts.
func (u Usage) Clone() Usage {
	u.InputTokens = clonePtr(u.InputTokens)
	u.OutputTokens = clonePtr(u.OutputTokens)
	u.TotalTokens = clonePtr(u.TotalTokens)
	u.ReasoningTokens = clonePtr(u.ReasoningTokens)
	u.CacheReadTokens = clonePtr(u.CacheReadTokens)
	u.CacheWriteTokens = clonePtr(u.CacheWriteTokens)
	return u
}

// FinishReason is a normalized stop reason; FinishReasonRaw keeps the vendor
// value. Empty means the vendor reported none.
type FinishReason string

const (
	FinishReasonStop     FinishReason = "stop"
	FinishReasonLength   FinishReason = "length"
	FinishReasonToolCall FinishReason = "tool_calls"
	FinishReasonError    FinishReason = "error"
	// FinishReasonSafety also covers explicit refusals; the refusal text, if
	// any, is an ordinary TextBlock.
	FinishReasonSafety FinishReason = "safety"
	// FinishReasonOther is a reported reason without a normalized equivalent.
	FinishReasonOther FinishReason = "other"
)

// Warning reports a non-fatal issue, such as dropped content or a generated ID.
// Code has the form "<source>.<snake_case>", such as "litellm.tool_arguments_invalid".
type Warning struct {
	Code     string
	Provider string
	Message  string
}
