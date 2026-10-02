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

func cloneResponse(resp *Response) *Response {
	if resp == nil {
		return nil
	}
	out := *resp
	out.Blocks = cloneBlocks(resp.Blocks)
	out.Warnings = append([]Warning(nil), resp.Warnings...)
	out.Raw = cloneBytes(resp.Raw)
	return &out
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

// Usage holds the token counts a response reported; a count the vendor did
// not report is zero. InputTokens includes cache reads and writes;
// OutputTokens includes reasoning.
type Usage struct {
	InputTokens      int `json:"input_tokens,omitempty"`
	OutputTokens     int `json:"output_tokens,omitempty"`
	ReasoningTokens  int `json:"reasoning_tokens,omitempty"`
	CacheReadTokens  int `json:"cache_read_tokens,omitempty"`
	CacheWriteTokens int `json:"cache_write_tokens,omitempty"`
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
	Code     string `json:"code"`
	Provider string `json:"provider,omitempty"`
	Message  string `json:"message"`
}
