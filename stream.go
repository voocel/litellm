package litellm

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
)

type Event interface {
	isEvent()
}

// ContentStart opens an explicitly addressed text or reasoning block. Block is
// its initial value, including any content already supplied by the provider.
// At least one coordinate is required. Tool calls use ToolUseStart/Done.
type ContentStart struct {
	Block        Block
	OutputIndex  *int
	ContentIndex *int
}

// ContentEnd closes a block opened by ContentStart. When Block is non-nil it is
// the final snapshot. Its text must match accumulated text; its metadata
// replaces prior metadata. It is not a delta or text correction. Later deltas
// for this block are invalid.
type ContentEnd struct {
	Block        Block
	OutputIndex  *int
	ContentIndex *int
}

type ContentDelta struct {
	Text string
	// Coordinates identify a block within this stream. Providers without block
	// coordinates may omit them only when content arrives in contiguous runs.
	OutputIndex  *int
	ContentIndex *int
}

type RefusalDelta struct {
	Text         string
	OutputIndex  *int
	ContentIndex *int
}

type ReasoningDelta struct {
	Text         string
	Summary      bool
	Signature    string
	Redacted     []byte
	Extra        json.RawMessage
	ExtraFull    bool
	OutputIndex  *int
	ContentIndex *int
}

type ToolUseStart struct {
	ID          string
	Name        string
	Index       *int
	OutputIndex *int
	ItemID      string
	Signature   string
}

type ToolUseDelta struct {
	ID             string
	Index          *int
	OutputIndex    *int
	ItemID         string
	ArgumentsDelta []byte
	Signature      string
}

type ToolUseDone struct {
	ID          string
	Index       *int
	OutputIndex *int
	ItemID      string
}

type UsageEvent struct {
	Usage Usage
}

type WarningEvent struct {
	Warning Warning
}

type DoneEvent struct {
	FinishReason    FinishReason
	FinishReasonRaw string
	Provider        string
	Model           string
}

type ProviderEvent struct {
	Name string
	Raw  json.RawMessage
}

func (ContentDelta) isEvent()   {}
func (ContentStart) isEvent()   {}
func (ContentEnd) isEvent()     {}
func (RefusalDelta) isEvent()   {}
func (ReasoningDelta) isEvent() {}
func (ToolUseStart) isEvent()   {}
func (ToolUseDelta) isEvent()   {}
func (ToolUseDone) isEvent()    {}
func (UsageEvent) isEvent()     {}
func (WarningEvent) isEvent()   {}
func (DoneEvent) isEvent()      {}
func (ProviderEvent) isEvent()  {}

func cloneEvent(event Event) Event {
	switch e := event.(type) {
	case ContentStart:
		e.Block = cloneBlock(e.Block)
		e.OutputIndex = cloneIntPtr(e.OutputIndex)
		e.ContentIndex = cloneIntPtr(e.ContentIndex)
		return e
	case ContentEnd:
		e.Block = cloneBlock(e.Block)
		e.OutputIndex = cloneIntPtr(e.OutputIndex)
		e.ContentIndex = cloneIntPtr(e.ContentIndex)
		return e
	case ContentDelta:
		e.OutputIndex = cloneIntPtr(e.OutputIndex)
		e.ContentIndex = cloneIntPtr(e.ContentIndex)
		return e
	case RefusalDelta:
		e.OutputIndex = cloneIntPtr(e.OutputIndex)
		e.ContentIndex = cloneIntPtr(e.ContentIndex)
		return e
	case ReasoningDelta:
		e.OutputIndex = cloneIntPtr(e.OutputIndex)
		e.ContentIndex = cloneIntPtr(e.ContentIndex)
		e.Redacted = cloneBytes(e.Redacted)
		e.Extra = cloneBytes(e.Extra)
		return e
	case ToolUseStart:
		e.Index = cloneIntPtr(e.Index)
		e.OutputIndex = cloneIntPtr(e.OutputIndex)
		return e
	case ToolUseDelta:
		e.Index = cloneIntPtr(e.Index)
		e.OutputIndex = cloneIntPtr(e.OutputIndex)
		e.ArgumentsDelta = cloneBytes(e.ArgumentsDelta)
		return e
	case ToolUseDone:
		e.Index = cloneIntPtr(e.Index)
		e.OutputIndex = cloneIntPtr(e.OutputIndex)
		return e
	case UsageEvent:
		return e
	case WarningEvent:
		return e
	case DoneEvent:
		return e
	case ProviderEvent:
		e.Raw = cloneBytes(e.Raw)
		return e
	default:
		return event
	}
}

// Stream is consumed by one goroutine. A successful stream emits exactly one
// DoneEvent. Failures are returned by Next, never encoded as events. Callers
// must Close the stream, including after completion or failure.
type Stream interface {
	Next() (Event, error)
	Close() error
}

type validatedStream struct {
	provider string
	inner    Stream
	state    *EventCollector
	done     bool
}

func newValidatedStream(provider, model string, stream Stream) Stream {
	if stream == nil {
		return nil
	}
	state := NewEventCollector()
	state.provider, state.model = provider, model
	return &validatedStream{provider: provider, inner: stream, state: state}
}

func (s *validatedStream) Next() (Event, error) {
	if s.done {
		return nil, io.EOF
	}
	event, err := s.inner.Next()
	if err != nil {
		s.done = true
		if errors.Is(err, io.EOF) {
			err = io.ErrUnexpectedEOF
		}
		return nil, WrapError(err, s.provider)
	}
	if event == nil {
		s.done = true
		return nil, NewProviderError(s.provider, ErrorTypeInternal, "stream returned nil event without error")
	}
	done, err := s.state.Apply(event)
	if err == nil && done {
		err = validateResponse(&Response{Blocks: s.state.blocks, Provider: s.state.provider, Model: s.state.model}, s.provider, s.state.model)
	}
	if err != nil {
		s.done = true
		return nil, WrapError(err, s.provider)
	}
	s.done = done
	return event, nil
}

// Client streams share their validated accumulator with Handle, avoiding a
// second full copy of streamed content. External streams are collected locally.
func (s *validatedStream) eventCollector() *EventCollector { return s.state }

func streamCollector(stream Stream) *EventCollector {
	if source, ok := stream.(interface{ eventCollector() *EventCollector }); ok {
		return source.eventCollector()
	}
	return nil
}

func (s *validatedStream) Close() error {
	s.done = true
	err := s.inner.Close()
	if err != nil {
		return WrapError(err, s.provider)
	}
	return nil
}

type warningPrefixStream struct {
	warnings []Warning
	index    int
	inner    Stream
}

func prependWarningEvents(stream Stream, warnings []Warning) Stream {
	if len(warnings) == 0 || stream == nil {
		return stream
	}
	copied := append([]Warning(nil), warnings...)
	return &warningPrefixStream{warnings: copied, inner: stream}
}

func (s *warningPrefixStream) Next() (Event, error) {
	if s.index < len(s.warnings) {
		warning := s.warnings[s.index]
		s.index++
		return WarningEvent{Warning: warning}, nil
	}
	return s.inner.Next()
}

func (s *warningPrefixStream) Close() error {
	return s.inner.Close()
}

// Collect consumes the stream and returns the aggregated Response. On failure it
// returns the partial response together with the error. EOF before DoneEvent is
// io.ErrUnexpectedEOF; partial responses must not be treated as completed output.
func Collect(stream Stream) (*Response, error) {
	return Handle(stream, nil)
}

// Handle consumes the stream, invoking fn for each event as it arrives, and
// returns the aggregated Response. It is the real-time counterpart to Collect;
// a nil fn behaves exactly like Collect. Events are aggregated before fn runs.
// On any failure Handle returns the partial response and the error. The caller
// still owns Close.
func Handle(stream Stream, fn func(Event) error) (*Response, error) {
	if stream == nil {
		return nil, fmt.Errorf("stream cannot be nil")
	}
	collector := streamCollector(stream)
	validated := collector != nil
	if !validated {
		collector = NewEventCollector()
	}
	for {
		event, err := stream.Next()
		if err != nil {
			if errors.Is(err, io.EOF) {
				return collector.Response(), fmt.Errorf("stream ended before Done event: %w", io.ErrUnexpectedEOF)
			}
			return collector.Response(), err
		}
		if event == nil {
			return collector.Response(), fmt.Errorf("stream returned nil event without error")
		}
		_, done := event.(DoneEvent)
		if !validated {
			done, err = collector.Apply(event)
			if err != nil {
				return collector.Response(), err
			}
		}
		if fn != nil {
			if err := fn(event); err != nil {
				return collector.Response(), err
			}
		}
		if done {
			resp := collector.Response()
			if !validated {
				if err := validateResponse(resp, resp.Provider, resp.Model); err != nil {
					return resp, err
				}
			}
			return resp, nil
		}
	}
}

// HandleText consumes the stream, invoking fn for each text content delta, and
// returns the aggregated Response. It is the simplest path for streaming answer
// text; reasoning and tool events are still aggregated into the Response but are
// not passed to fn.
func HandleText(stream Stream, fn func(string) error) (*Response, error) {
	return HandleWith(stream, StreamHandler{Content: fn})
}

// StreamHandler routes streamed deltas to per-category callbacks. Unset
// callbacks are skipped; every event is still aggregated into the returned
// Response. For full event fidelity (tool-call streaming, provider events), use
// Handle or the raw Stream.
type StreamHandler struct {
	Content   func(string) error
	Reasoning func(string) error
}

// HandleWith consumes the stream, dispatching content and reasoning deltas to
// the handler's callbacks, and returns the aggregated Response.
func HandleWith(stream Stream, handler StreamHandler) (*Response, error) {
	return Handle(stream, func(event Event) error {
		switch e := event.(type) {
		case ContentStart:
			switch block := e.Block.(type) {
			case TextBlock:
				if handler.Content != nil && block.Text != "" {
					return handler.Content(block.Text)
				}
			case ReasoningBlock:
				if handler.Reasoning != nil && block.Text != "" {
					return handler.Reasoning(block.Text)
				}
			}
		case ContentDelta:
			if handler.Content != nil && e.Text != "" {
				return handler.Content(e.Text)
			}
		case ReasoningDelta:
			if handler.Reasoning != nil && e.Text != "" {
				return handler.Reasoning(e.Text)
			}
		}
		return nil
	})
}

// EventCollector incrementally aggregates stream events into a Response.
// Create one with NewEventCollector, call Apply for each event in order, then
// read Response after Apply reports completion.
type EventCollector struct {
	blocks []Block
	// Each block has its own builder so interleaved deltas retain identity.
	textBuilders    map[int]*strings.Builder
	contentIndexes  map[contentAddress]int
	contentStates   map[contentAddress]contentState
	anonymousIndex  int
	anonymousOutput int
	done            bool
	toolIndexes     map[string]int
	usage           Usage
	finish          FinishReason
	finishRaw       string
	refusal         strings.Builder
	provider        string
	model           string
	warnings        []Warning
	tools           *ToolUseAccumulator
}

// NewEventCollector returns an initialized stream event collector.
func NewEventCollector() *EventCollector {
	return &EventCollector{
		textBuilders:   make(map[int]*strings.Builder),
		contentIndexes: make(map[contentAddress]int),
		contentStates:  make(map[contentAddress]contentState),
		anonymousIndex: -1,
		toolIndexes:    make(map[string]int),
		tools:          NewToolUseAccumulator(),
	}
}

func (c *EventCollector) Apply(event Event) (bool, error) {
	if c.done {
		return false, fmt.Errorf("stream event received after Done event")
	}
	switch e := event.(type) {
	case ContentStart:
		return false, c.startContent(e)
	case ContentEnd:
		return false, c.endContent(e)
	case ContentDelta:
		if err := c.checkContentDelta(e.OutputIndex, e.ContentIndex, "text", false); err != nil {
			return false, err
		}
		c.appendContent(e.Text, e.OutputIndex, e.ContentIndex)
	case RefusalDelta:
		if err := c.checkContentDelta(e.OutputIndex, e.ContentIndex, "text", false); err != nil {
			return false, err
		}
		c.refusal.WriteString(e.Text)
		c.appendContent(e.Text, e.OutputIndex, e.ContentIndex)
	case ReasoningDelta:
		if err := c.checkContentDelta(e.OutputIndex, e.ContentIndex, "reasoning", e.Summary); err != nil {
			return false, err
		}
		c.appendReasoning(e)
	case ToolUseStart:
		key, tool, err := c.tools.Start(e)
		if err != nil {
			return false, err
		}
		c.appendTool(key, tool)
	case ToolUseDelta:
		key, tool, err := c.tools.Delta(e)
		if err != nil {
			return false, err
		}
		c.appendTool(key, tool)
	case ToolUseDone:
		key, tool, err := c.tools.Done(e)
		if err != nil {
			return false, err
		}
		if warning := normalizeInvalidToolArguments(tool); warning != nil {
			c.appendWarning(*warning)
		}
		c.appendTool(key, tool)
	case UsageEvent:
		c.usage = e.Usage
		if e.Usage.Provider != "" {
			c.provider = e.Usage.Provider
		}
		if e.Usage.Model != "" {
			c.model = e.Usage.Model
		}
	case WarningEvent:
		c.appendWarning(e.Warning)
	case ProviderEvent:
		// Provider-native events are observable by stream consumers. The core
		// collector ignores them unless they are promoted to typed events.
	case DoneEvent:
		for _, state := range c.contentStates {
			if !state.closed {
				return false, fmt.Errorf("stream completed with an unclosed content block")
			}
		}
		c.done = true
		c.finish = e.FinishReason
		c.finishRaw = e.FinishReasonRaw
		if c.refusal.Len() > 0 {
			c.finish = FinishReasonSafety
		}
		if e.Provider != "" {
			c.provider = e.Provider
		}
		if e.Model != "" {
			c.model = e.Model
		}
		c.normalizeToolArguments()
		return true, nil
	default:
		return false, fmt.Errorf("unknown stream event %T", event)
	}
	return false, nil
}

func (c *EventCollector) appendWarning(w Warning) {
	if w.Provider == "" {
		w.Provider = c.provider
	}
	c.warnings = append(c.warnings, w)
}

func (c *EventCollector) normalizeToolArguments() {
	for i, block := range c.blocks {
		tool, ok := block.(ToolUseBlock)
		if !ok {
			continue
		}
		if warning := normalizeInvalidToolArguments(&tool); warning != nil {
			c.blocks[i] = tool
			c.appendWarning(*warning)
		}
	}
}

func normalizeInvalidToolArguments(tool *ToolUseBlock) *Warning {
	if tool == nil || len(tool.Arguments) == 0 || json.Valid(tool.Arguments) {
		return nil
	}
	var probe any
	parseErr := json.Unmarshal(tool.Arguments, &probe)
	return &Warning{
		Code:     "stream.tool_arguments_invalid",
		Message:  fmt.Sprintf("tool use %q returned malformed JSON arguments: %v", tool.ID, parseErr),
		Provider: "",
		// Keep raw arguments out of Warning to avoid leaking large or sensitive
		// payloads through observability hooks. Consumers that need raw deltas
		// can observe ToolUseDelta events directly.
	}
}

func (c *EventCollector) Response() *Response {
	resp := &Response{
		Blocks:          c.cloneBlocks(),
		Usage:           c.usage,
		Model:           c.model,
		Provider:        c.provider,
		FinishReason:    c.finish,
		FinishReasonRaw: c.finishRaw,
		Refusal:         c.refusal.String(),
		Warnings:        append([]Warning(nil), c.warnings...),
	}
	resp.Usage.StampModel(resp.Provider, resp.Model)
	return resp
}

// A content coordinate identifies a block; an output coordinate alone only
// identifies its channel. Unaddressed channels are collected in contiguous runs.
func (c *EventCollector) contentIndex(kind string, output, content *int, summary bool) int {
	address := addressOf(output, content)
	_, explicit := c.contentStates[address]
	indexed := content != nil || explicit
	if indexed {
		c.anonymousIndex = -1
		if index, ok := c.contentIndexes[address]; ok {
			return index
		}
	} else if c.anonymousIndex >= 0 && c.anonymousIndex == len(c.blocks)-1 && c.anonymousOutput == address.output {
		switch block := c.blocks[c.anonymousIndex].(type) {
		case TextBlock:
			if kind == "text" {
				return c.anonymousIndex
			}
		case ReasoningBlock:
			if kind == "reasoning" && len(block.Redacted) == 0 && block.Summary == summary {
				return c.anonymousIndex
			}
		}
	}
	index := len(c.blocks)
	if kind == "text" {
		c.blocks = append(c.blocks, TextBlock{})
	} else {
		c.blocks = append(c.blocks, ReasoningBlock{Summary: summary})
	}
	if indexed {
		c.contentIndexes[address] = index
		c.anonymousIndex = -1
	} else {
		c.anonymousIndex = index
		c.anonymousOutput = address.output
	}
	return index
}

func (c *EventCollector) appendText(index int, text string) {
	if text == "" {
		return
	}
	builder := c.textBuilders[index]
	if builder == nil {
		builder = &strings.Builder{}
		switch block := c.blocks[index].(type) {
		case TextBlock:
			builder.WriteString(block.Text)
		case ReasoningBlock:
			builder.WriteString(block.Text)
		}
		c.textBuilders[index] = builder
	}
	builder.WriteString(text)
}

func (c *EventCollector) appendContent(text string, output, content *int) {
	if text == "" {
		return
	}
	c.appendText(c.contentIndex("text", output, content, false), text)
}

func (c *EventCollector) appendReasoning(delta ReasoningDelta) {
	if delta.Text == "" && delta.Signature == "" && len(delta.Redacted) == 0 && len(delta.Extra) == 0 {
		return
	}
	index := c.contentIndex("reasoning", delta.OutputIndex, delta.ContentIndex, delta.Summary)
	block := c.blocks[index].(ReasoningBlock)
	c.appendText(index, delta.Text)
	if delta.Signature != "" {
		block.Signature = delta.Signature
	}
	block.Redacted = append(block.Redacted, delta.Redacted...)
	block.Extra = mergeReasoningExtra(block.Extra, delta)
	c.blocks[index] = block
}

func mergeReasoningExtra(current json.RawMessage, delta ReasoningDelta) json.RawMessage {
	if len(delta.Extra) == 0 {
		return current
	}
	if delta.ExtraFull {
		return cloneBytes(delta.Extra)
	}
	if len(current) == 0 {
		return cloneBytes(delta.Extra)
	}
	var currentItems, deltaItems []json.RawMessage
	if json.Unmarshal(current, &currentItems) == nil && json.Unmarshal(delta.Extra, &deltaItems) == nil {
		merged := make([]json.RawMessage, 0, len(currentItems)+len(deltaItems))
		merged = append(merged, currentItems...)
		merged = append(merged, deltaItems...)
		if data, err := json.Marshal(merged); err == nil {
			return data
		}
	}
	return cloneBytes(delta.Extra)
}

func (c *EventCollector) appendTool(key string, tool *ToolUseBlock) {
	if key == "" || tool == nil {
		return
	}
	if index, ok := c.toolIndexes[key]; ok {
		c.blocks[index] = cloneToolUseBlock(*tool)
		return
	}
	c.toolIndexes[key] = len(c.blocks)
	c.blocks = append(c.blocks, cloneToolUseBlock(*tool))
}

func (c *EventCollector) cloneBlocks() []Block {
	out := cloneBlocks(c.blocks)
	for index, builder := range c.textBuilders {
		switch block := out[index].(type) {
		case TextBlock:
			block.Text = builder.String()
			out[index] = block
		case ReasoningBlock:
			block.Text = builder.String()
			out[index] = block
		}
	}
	return out
}

func cloneToolUseBlock(block ToolUseBlock) ToolUseBlock {
	block.Arguments = json.RawMessage(cloneBytes(block.Arguments))
	block.Extra = json.RawMessage(cloneBytes(block.Extra))
	return block
}

type ToolUseAccumulator struct {
	order   []string
	byKey   map[string]*ToolUseBlock
	aliases map[string]string
}

func NewToolUseAccumulator() *ToolUseAccumulator {
	return &ToolUseAccumulator{
		byKey:   make(map[string]*ToolUseBlock),
		aliases: make(map[string]string),
	}
}

func (a *ToolUseAccumulator) Start(start ToolUseStart) (string, *ToolUseBlock, error) {
	key, tool, err := a.ensureFor(start.ID, start.Index, start.OutputIndex, start.ItemID)
	if err != nil {
		return "", nil, fmt.Errorf("tool use start: %w", err)
	}
	if start.ID != "" {
		tool.ID = start.ID
	}
	if start.Name != "" {
		tool.Name = start.Name
	}
	if start.Signature != "" {
		tool.Signature = start.Signature
	}
	return key, tool, nil
}

func (a *ToolUseAccumulator) Delta(delta ToolUseDelta) (string, *ToolUseBlock, error) {
	key, tool, err := a.ensureFor(delta.ID, delta.Index, delta.OutputIndex, delta.ItemID)
	if err != nil {
		return "", nil, fmt.Errorf("tool use delta: %w", err)
	}
	if delta.ID != "" {
		tool.ID = delta.ID
	}
	if delta.Signature != "" {
		tool.Signature = delta.Signature
	}
	if len(delta.ArgumentsDelta) > 0 {
		tool.Arguments = append(tool.Arguments, delta.ArgumentsDelta...)
	}
	return key, tool, nil
}

func (a *ToolUseAccumulator) Done(done ToolUseDone) (string, *ToolUseBlock, error) {
	key, tool, err := a.findFor(done.ID, done.Index, done.OutputIndex, done.ItemID)
	if err != nil {
		return "", nil, fmt.Errorf("tool use done: %w", err)
	}
	if done.ID != "" {
		tool.ID = done.ID
	}
	// A tool call with no streamed argument deltas (an argument-less call)
	// normalizes to an empty JSON object, keeping the block valid JSON for
	// response validation and replay rather than dangling as empty bytes.
	if len(tool.Arguments) == 0 {
		tool.Arguments = json.RawMessage("{}")
	}
	return key, tool, nil
}

func (a *ToolUseAccumulator) ensureFor(id string, index, outputIndex *int, itemID string) (string, *ToolUseBlock, error) {
	keys := toolUseKeys(id, index, outputIndex, itemID)
	if len(keys) == 0 {
		return "", nil, fmt.Errorf("tool use missing id and index")
	}
	primary := keys[0]
	if key, tool := a.lookup(keys); tool != nil {
		a.aliasAll(key, keys)
		return key, tool, nil
	}
	tool := &ToolUseBlock{}
	a.byKey[primary] = tool
	a.order = append(a.order, primary)
	a.aliasAll(primary, keys)
	return primary, tool, nil
}

func (a *ToolUseAccumulator) findFor(id string, index, outputIndex *int, itemID string) (string, *ToolUseBlock, error) {
	keys := toolUseKeys(id, index, outputIndex, itemID)
	if len(keys) == 0 {
		return "", nil, fmt.Errorf("tool use missing id and index")
	}
	if key, tool := a.lookup(keys); tool != nil {
		a.aliasAll(key, keys)
		return key, tool, nil
	}
	return "", nil, fmt.Errorf("tool use done references unknown tool use")
}

func (a *ToolUseAccumulator) lookup(keys []string) (string, *ToolUseBlock) {
	for _, key := range keys {
		if canonical := a.aliases[key]; canonical != "" {
			if tool := a.byKey[canonical]; tool != nil {
				return canonical, tool
			}
		}
		if tool := a.byKey[key]; tool != nil {
			return key, tool
		}
	}
	return "", nil
}

func (a *ToolUseAccumulator) aliasAll(canonical string, keys []string) {
	for _, key := range keys {
		a.aliases[key] = canonical
	}
}

func toolUseKeys(id string, index, outputIndex *int, itemID string) []string {
	keys := make([]string, 0, 4)
	if id != "" {
		keys = append(keys, "id:"+id)
	}
	if itemID != "" {
		keys = append(keys, "item:"+itemID)
	}
	if index != nil && outputIndex != nil {
		keys = append(keys, fmt.Sprintf("output:%d/index:%d", *outputIndex, *index))
		return keys
	}
	if index != nil {
		keys = append(keys, fmt.Sprintf("index:%d", *index))
	}
	if outputIndex != nil {
		keys = append(keys, fmt.Sprintf("output:%d", *outputIndex))
	}
	return keys
}
