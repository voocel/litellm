package litellm

import (
	"fmt"
	"sort"
	"strings"
)

// ---------------------------------------------------------------------------
// ToolCallAccumulator — reusable tool call delta reconstruction
// ---------------------------------------------------------------------------

// ToolCallAccumulator reconstructs complete ToolCall objects from streaming deltas.
// Safe for single-goroutine use only.
type ToolCallAccumulator struct {
	order []string
	byKey map[string]*ToolCall
}

// NewToolCallAccumulator creates an empty accumulator.
func NewToolCallAccumulator() *ToolCallAccumulator {
	return &ToolCallAccumulator{
		byKey: make(map[string]*ToolCall),
	}
}

// Apply processes a single ToolCallDelta, creating or updating the corresponding ToolCall.
func (a *ToolCallAccumulator) Apply(delta *ToolCallDelta) {
	if delta == nil {
		return
	}

	// Always key by index so that start events (which carry ID/name)
	// and subsequent delta events (which carry only arguments) merge
	// into the same ToolCall entry.
	key := fmt.Sprintf("index:%d", delta.Index)

	tc := a.byKey[key]
	if tc == nil {
		tc = &ToolCall{
			ID:   delta.ID,
			Type: delta.Type,
			Function: FunctionCall{
				Name: delta.FunctionName,
			},
		}
		if tc.Type == "" {
			tc.Type = "function"
		}
		a.byKey[key] = tc
		a.order = append(a.order, key)
	}

	if delta.ID != "" {
		tc.ID = delta.ID
	}
	if delta.FunctionName != "" {
		tc.Function.Name = delta.FunctionName
	}
	if delta.ArgumentsDelta != "" {
		tc.Function.Arguments += delta.ArgumentsDelta
	}
}

// Build returns the completed ToolCall list in first-seen order.
func (a *ToolCallAccumulator) Build() []ToolCall {
	if len(a.order) == 0 {
		return nil
	}
	result := make([]ToolCall, 0, len(a.order))
	for _, key := range a.order {
		if tc := a.byKey[key]; tc != nil {
			result = append(result, *tc)
		}
	}
	return result
}

// Started reports whether a delta with the given index has been received.
func (a *ToolCallAccumulator) Started(index int) bool {
	_, ok := a.byKey[fmt.Sprintf("index:%d", index)]
	return ok
}

// ---------------------------------------------------------------------------
// StreamCallbacks & CollectStream
// ---------------------------------------------------------------------------

// StreamCallbacks provides optional per-chunk handlers during stream collection.
type StreamCallbacks struct {
	OnChunk     func(*StreamChunk)
	OnContent   func(string)
	OnReasoning func(*ReasoningChunk)
	OnToolCall  func(*ToolCallDelta)
}

// CollectStream consumes a StreamReader and returns a unified Response.
// Callers are responsible for closing the stream.
func CollectStream(stream StreamReader) (*Response, error) {
	return CollectStreamWithHandler(stream, nil)
}

// CollectStreamWithCallbacks consumes a StreamReader, calls callbacks for each chunk, and returns a unified Response.
// Callers are responsible for closing the stream.
func CollectStreamWithCallbacks(stream StreamReader, callbacks StreamCallbacks) (*Response, error) {
	return CollectStreamWithHandler(stream, func(chunk *StreamChunk) {
		if callbacks.OnChunk != nil {
			callbacks.OnChunk(chunk)
		}
		if callbacks.OnContent != nil && chunk.Type == ChunkTypeContent && chunk.Content != "" {
			callbacks.OnContent(chunk.Content)
		}
		if callbacks.OnReasoning != nil && chunk.Reasoning != nil {
			callbacks.OnReasoning(chunk.Reasoning)
		}
		if callbacks.OnToolCall != nil && chunk.ToolCallDelta != nil {
			callbacks.OnToolCall(chunk.ToolCallDelta)
		}
	})
}

// CollectStreamWithHandler consumes a StreamReader, calls onChunk for each chunk, and returns a unified Response.
// Callers are responsible for closing the stream.
func CollectStreamWithHandler(stream StreamReader, onChunk func(*StreamChunk)) (*Response, error) {
	if stream == nil {
		return nil, fmt.Errorf("stream cannot be nil")
	}

	var (
		contentBuilder       strings.Builder
		refusalBuilder       strings.Builder
		contentByOutputIndex = map[int]*strings.Builder{}
		refusalByOutputIndex = map[int]*strings.Builder{}
		reasoningSummary     strings.Builder
		reasoningContent     strings.Builder
		toolAcc              = NewToolCallAccumulator()
		resp                 Response
	)

	for {
		chunk, err := stream.Next()
		if err != nil {
			return nil, err
		}
		if chunk == nil {
			continue
		}

		if onChunk != nil {
			onChunk(chunk)
		}

		if resp.Provider == "" && chunk.Provider != "" {
			resp.Provider = chunk.Provider
		}
		if chunk.Model != "" {
			resp.Model = chunk.Model
		}
		if resp.FinishReason == "" && chunk.FinishReason != "" {
			resp.FinishReason = chunk.FinishReason
		}

		if chunk.Content != "" {
			switch chunk.Type {
			case ChunkTypeContent:
				if chunk.OutputIndex != nil {
					builder := contentByOutputIndex[*chunk.OutputIndex]
					if builder == nil {
						builder = &strings.Builder{}
						contentByOutputIndex[*chunk.OutputIndex] = builder
					}
					builder.WriteString(chunk.Content)
				} else {
					contentBuilder.WriteString(chunk.Content)
				}
			case "refusal":
				if chunk.OutputIndex != nil {
					builder := refusalByOutputIndex[*chunk.OutputIndex]
					if builder == nil {
						builder = &strings.Builder{}
						refusalByOutputIndex[*chunk.OutputIndex] = builder
					}
					builder.WriteString(chunk.Content)
				} else {
					refusalBuilder.WriteString(chunk.Content)
				}
			}
		}

		if chunk.Reasoning != nil {
			if chunk.Reasoning.Summary != "" {
				if reasoningSummary.Len() > 0 {
					reasoningSummary.WriteString("\n")
				}
				reasoningSummary.WriteString(chunk.Reasoning.Summary)
			}
			if chunk.Reasoning.Content != "" {
				if reasoningContent.Len() > 0 {
					reasoningContent.WriteString("\n")
				}
				reasoningContent.WriteString(chunk.Reasoning.Content)
			}
		}

		if chunk.ToolCallDelta != nil {
			toolAcc.Apply(chunk.ToolCallDelta)
		}

		if chunk.Usage != nil {
			resp.Usage = *chunk.Usage
		}

		if chunk.Done {
			break
		}
	}

	if len(contentByOutputIndex) > 0 || len(refusalByOutputIndex) > 0 {
		indices := make([]int, 0, len(contentByOutputIndex)+len(refusalByOutputIndex))
		for index := range contentByOutputIndex {
			indices = append(indices, index)
		}
		for index := range refusalByOutputIndex {
			indices = append(indices, index)
		}
		sort.Ints(indices)
		var merged strings.Builder
		seen := map[int]bool{}
		for _, index := range indices {
			if seen[index] {
				continue
			}
			seen[index] = true
			if builder := contentByOutputIndex[index]; builder != nil && builder.Len() > 0 {
				merged.WriteString(builder.String())
				continue
			}
			if builder := refusalByOutputIndex[index]; builder != nil {
				merged.WriteString(builder.String())
			}
		}
		merged.WriteString(contentBuilder.String())
		resp.Content = merged.String()
	} else {
		resp.Content = contentBuilder.String()
	}
	if resp.Content == "" && refusalBuilder.Len() > 0 {
		resp.Content = refusalBuilder.String()
	}

	if reasoningSummary.Len() > 0 || reasoningContent.Len() > 0 {
		resp.Reasoning = &ReasoningData{
			Summary: reasoningSummary.String(),
			Content: reasoningContent.String(),
		}
	}

	resp.ToolCalls = toolAcc.Build()

	return &resp, nil
}
