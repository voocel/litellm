package litellm

import (
	"context"
	"fmt"
	"sort"
	"strings"

	"github.com/voocel/litellm/providers"
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
	OnReasoning func(content string)
	OnToolCall  func(*ToolCallDelta)

	// Lifecycle callbacks — bracket start/end of each content block.
	// Transitions are detected automatically from chunk types.
	OnContentStart   func()
	OnContentEnd     func(content string) // content = full accumulated block
	OnReasoningStart func()
	OnReasoningEnd   func(content string)       // content = full accumulated reasoning
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
	dispatcher := newStreamCallbackDispatcher(callbacks)
	return CollectStreamWithHandler(stream, func(chunk *StreamChunk) {
		dispatcher.process(chunk)
	})
}

// CollectStreamWithHandler consumes a StreamReader, calls onChunk for each chunk, and returns a unified Response.
// Callers are responsible for closing the stream.
func CollectStreamWithHandler(stream StreamReader, onChunk func(*StreamChunk)) (*Response, error) {
	if stream == nil {
		return nil, fmt.Errorf("stream cannot be nil")
	}

	collector := newStreamCollector()

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
		collector.applyChunk(chunk)

		if chunk.Done {
			break
		}
	}
	return collector.buildResponse()
}

type streamCollector struct {
	contentBuilder       strings.Builder
	refusalBuilder       strings.Builder
	contentByOutputIndex map[int]*strings.Builder
	refusalByOutputIndex map[int]*strings.Builder
	reasoningContent     strings.Builder
	toolAcc              *ToolCallAccumulator
	resp                 Response
}

func newStreamCollector() *streamCollector {
	return &streamCollector{
		contentByOutputIndex: make(map[int]*strings.Builder),
		refusalByOutputIndex: make(map[int]*strings.Builder),
		toolAcc:              NewToolCallAccumulator(),
	}
}

func (c *streamCollector) applyChunk(chunk *StreamChunk) {
	if chunk == nil {
		return
	}

	if c.resp.Provider == "" && chunk.Provider != "" {
		c.resp.Provider = chunk.Provider
	}
	if chunk.Model != "" {
		c.resp.Model = chunk.Model
	}
	if c.resp.FinishReason == "" && chunk.FinishReason != "" {
		c.resp.FinishReason = chunk.FinishReason
	}

	c.collectContent(chunk)
	c.collectReasoning(chunk)
	c.collectToolCall(chunk)

	if chunk.Usage != nil {
		c.resp.Usage = *chunk.Usage
	}
}

func (c *streamCollector) collectContent(chunk *StreamChunk) {
	if chunk.Content == "" {
		return
	}

	switch chunk.Type {
	case ChunkTypeContent:
		c.outputBuilder(chunk.OutputIndex, c.contentByOutputIndex, &c.contentBuilder).WriteString(chunk.Content)
	case "refusal":
		c.outputBuilder(chunk.OutputIndex, c.refusalByOutputIndex, &c.refusalBuilder).WriteString(chunk.Content)
	}
}

func (c *streamCollector) collectReasoning(chunk *StreamChunk) {
	if chunk.ReasoningContent == "" {
		return
	}
	if c.reasoningContent.Len() > 0 {
		c.reasoningContent.WriteString("\n")
	}
	c.reasoningContent.WriteString(chunk.ReasoningContent)
}

func (c *streamCollector) collectToolCall(chunk *StreamChunk) {
	if chunk.ToolCallDelta != nil {
		c.toolAcc.Apply(chunk.ToolCallDelta)
	}
}

func (c *streamCollector) outputBuilder(index *int, byIndex map[int]*strings.Builder, fallback *strings.Builder) *strings.Builder {
	if index == nil {
		return fallback
	}

	builder := byIndex[*index]
	if builder == nil {
		builder = &strings.Builder{}
		byIndex[*index] = builder
	}
	return builder
}

func (c *streamCollector) buildResponse() (*Response, error) {
	c.resp.Content = c.mergedContent()
	if c.reasoningContent.Len() > 0 {
		c.resp.ReasoningContent = c.reasoningContent.String()
	}

	c.resp.ToolCalls = c.toolAcc.Build()
	if err := validateToolCalls(c.resp.Provider, c.resp.ToolCalls); err != nil {
		return nil, err
	}

	if c.resp.Content == "" && c.resp.ReasoningContent == "" && len(c.resp.ToolCalls) == 0 && c.resp.FinishReason == "" {
		return nil, providers.NewNetworkError(c.resp.Provider,
			"stream completed but produced no output (0 content, 0 reasoning, 0 tool calls, no finish reason)", nil)
	}

	return &c.resp, nil
}

func (c *streamCollector) mergedContent() string {
	if len(c.contentByOutputIndex) == 0 && len(c.refusalByOutputIndex) == 0 {
		if c.contentBuilder.Len() == 0 && c.refusalBuilder.Len() > 0 {
			return c.refusalBuilder.String()
		}
		return c.contentBuilder.String()
	}

	indices := make([]int, 0, len(c.contentByOutputIndex)+len(c.refusalByOutputIndex))
	for index := range c.contentByOutputIndex {
		indices = append(indices, index)
	}
	for index := range c.refusalByOutputIndex {
		indices = append(indices, index)
	}
	sort.Ints(indices)

	var merged strings.Builder
	seen := make(map[int]bool, len(indices))
	for _, index := range indices {
		if seen[index] {
			continue
		}
		seen[index] = true
		if builder := c.contentByOutputIndex[index]; builder != nil && builder.Len() > 0 {
			merged.WriteString(builder.String())
			continue
		}
		if builder := c.refusalByOutputIndex[index]; builder != nil {
			merged.WriteString(builder.String())
		}
	}

	merged.WriteString(c.contentBuilder.String())
	if merged.Len() == 0 && c.refusalBuilder.Len() > 0 {
		return c.refusalBuilder.String()
	}
	return merged.String()
}

type streamCallbackDispatcher struct {
	callbacks StreamCallbacks
	lifecycle *lifecycleTracker
}

func newStreamCallbackDispatcher(callbacks StreamCallbacks) *streamCallbackDispatcher {
	return &streamCallbackDispatcher{
		callbacks: callbacks,
		lifecycle: newLifecycleTracker(&callbacks),
	}
}

func (d *streamCallbackDispatcher) process(chunk *StreamChunk) {
	d.lifecycle.process(chunk)

	if d.callbacks.OnChunk != nil {
		d.callbacks.OnChunk(chunk)
	}
	if d.callbacks.OnContent != nil && chunk.Type == ChunkTypeContent && chunk.Content != "" {
		d.callbacks.OnContent(chunk.Content)
	}
	if d.callbacks.OnReasoning != nil && chunk.ReasoningContent != "" {
		d.callbacks.OnReasoning(chunk.ReasoningContent)
	}
	if d.callbacks.OnToolCall != nil && chunk.ToolCallDelta != nil {
		d.callbacks.OnToolCall(chunk.ToolCallDelta)
	}
}

type lifecycleTracker struct {
	cb          *StreamCallbacks
	active      string
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
		t.reasonAcc.WriteString(chunk.ReasoningContent)

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

type hookedStreamReader struct {
	ctx    context.Context
	meta   CallMeta
	hooks  []Hook
	stream StreamReader
}

func newHookedStreamReader(ctx context.Context, meta CallMeta, hooks []Hook, stream StreamReader) StreamReader {
	if stream == nil || len(hooks) == 0 {
		return stream
	}
	return &hookedStreamReader{
		ctx:    ctx,
		meta:   meta,
		hooks:  hooks,
		stream: stream,
	}
}

func (r *hookedStreamReader) Next() (*StreamChunk, error) {
	chunk, err := r.stream.Next()
	if err != nil {
		return nil, err
	}
	if chunk != nil {
		for _, h := range r.hooks {
			h.OnStreamChunk(r.ctx, r.meta, chunk)
		}
	}
	return chunk, nil
}

func (r *hookedStreamReader) Close() error {
	return r.stream.Close()
}
