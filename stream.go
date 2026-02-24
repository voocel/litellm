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

// Get returns the accumulated ToolCall for the given index, or nil if not found.
func (a *ToolCallAccumulator) Get(index int) *ToolCall {
	return a.byKey[fmt.Sprintf("index:%d", index)]
}

// PartialArguments returns a best-effort parse of the accumulated (possibly incomplete)
// function arguments for the tool call at the given index.
// Useful for streaming UIs that want to display arguments as they arrive.
// Returns nil if the index has no accumulated data.
func (a *ToolCallAccumulator) PartialArguments(index int) any {
	tc := a.Get(index)
	if tc == nil || tc.Function.Arguments == "" {
		return nil
	}
	return ParsePartialJSON(tc.Function.Arguments)
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

	// Lifecycle callbacks — bracket start/end of each content block.
	// Transitions are detected automatically from chunk types.
	OnContentStart   func()
	OnContentEnd     func(content string)     // content = full accumulated block
	OnReasoningStart func()
	OnReasoningEnd   func(content string)     // content = full accumulated reasoning
	OnToolCallStart  func(delta *ToolCallDelta) // carries ID and FunctionName
	OnToolCallEnd    func(call ToolCall)        // carries complete ToolCall
}

// CollectStream consumes a StreamReader and returns a unified Response.
// Callers are responsible for closing the stream.
func CollectStream(stream StreamReader) (*Response, error) {
	return CollectStreamWithHandler(stream, nil)
}

// CollectStreamWithCallbacks consumes a StreamReader, calls callbacks for each chunk, and returns a unified Response.
// Callers are responsible for closing the stream.
func CollectStreamWithCallbacks(stream StreamReader, callbacks StreamCallbacks) (*Response, error) {
	lc := newLifecycleTracker(&callbacks)
	return CollectStreamWithHandler(stream, func(chunk *StreamChunk) {
		lc.process(chunk)

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

// ---------------------------------------------------------------------------
// lifecycleTracker — emits start/end callbacks for content block transitions
// ---------------------------------------------------------------------------

type lifecycleTracker struct {
	cb          *StreamCallbacks
	active      string // "", "content", "reasoning"
	contentAcc  strings.Builder
	reasonAcc   strings.Builder
	toolStarted map[int]bool
	toolAcc     *ToolCallAccumulator
	closed      bool
}

func newLifecycleTracker(cb *StreamCallbacks) *lifecycleTracker {
	return &lifecycleTracker{
		cb:          cb,
		toolStarted: make(map[int]bool),
		toolAcc:     NewToolCallAccumulator(),
	}
}

func (t *lifecycleTracker) hasCallbacks() bool {
	cb := t.cb
	return cb.OnContentStart != nil || cb.OnContentEnd != nil ||
		cb.OnReasoningStart != nil || cb.OnReasoningEnd != nil ||
		cb.OnToolCallStart != nil || cb.OnToolCallEnd != nil
}

func (t *lifecycleTracker) process(chunk *StreamChunk) {
	if !t.hasCallbacks() {
		return
	}

	switch chunk.Type {
	case ChunkTypeContent:
		if t.active != "content" {
			t.closeActive()
			t.active = "content"
			if t.cb.OnContentStart != nil {
				t.cb.OnContentStart()
			}
		}
		t.contentAcc.WriteString(chunk.Content)

	case ChunkTypeReasoning:
		if t.active != "reasoning" {
			t.closeActive()
			t.active = "reasoning"
			if t.cb.OnReasoningStart != nil {
				t.cb.OnReasoningStart()
			}
		}
		if chunk.Reasoning != nil {
			t.reasonAcc.WriteString(chunk.Reasoning.Content)
		}

	case ChunkTypeToolCallDelta, ChunkTypeToolCallStart:
		if chunk.ToolCallDelta != nil {
			idx := chunk.ToolCallDelta.Index
			t.toolAcc.Apply(chunk.ToolCallDelta)
			if !t.toolStarted[idx] {
				if len(t.toolStarted) == 0 {
					t.closeActive()
				}
				t.toolStarted[idx] = true
				if t.cb.OnToolCallStart != nil {
					t.cb.OnToolCallStart(chunk.ToolCallDelta)
				}
			}
		}

	case ChunkTypeToolCallEnd, "tool_call_done":
		if chunk.ToolCallDelta != nil {
			idx := chunk.ToolCallDelta.Index
			t.toolAcc.Apply(chunk.ToolCallDelta)
			if t.cb.OnToolCallEnd != nil {
				if tc := t.toolAcc.Get(idx); tc != nil {
					t.cb.OnToolCallEnd(*tc)
				}
			}
			delete(t.toolStarted, idx)
		}

	case "reasoning_done":
		// OpenAI Responses API signals reasoning complete explicitly.
		if t.active == "reasoning" {
			t.closeActive()
		}
	}

	if chunk.Done || chunk.FinishReason != "" {
		t.finish()
	}
}

func (t *lifecycleTracker) closeActive() {
	switch t.active {
	case "content":
		if t.cb.OnContentEnd != nil {
			t.cb.OnContentEnd(t.contentAcc.String())
		}
		t.contentAcc.Reset()
	case "reasoning":
		if t.cb.OnReasoningEnd != nil {
			t.cb.OnReasoningEnd(t.reasonAcc.String())
		}
		t.reasonAcc.Reset()
	}
	t.active = ""
}

func (t *lifecycleTracker) finish() {
	if t.closed {
		return
	}
	t.closed = true
	t.closeActive()
	if t.cb.OnToolCallEnd != nil {
		for idx := range t.toolStarted {
			if tc := t.toolAcc.Get(idx); tc != nil {
				t.cb.OnToolCallEnd(*tc)
			}
		}
	}
}
