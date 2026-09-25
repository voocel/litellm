package litellm

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
)

// Event is a stream event. See the types below for the block lifecycle.
type Event interface {
	isEvent()
}

// Content events address a block by Index, its position in Response.Blocks.
// Providers number blocks densely in order of first appearance. Each block is
// opened by BlockStart, extended only by deltas of its own kind and closed by
// BlockEnd before DoneEvent. Blocks may interleave.

// BlockStart opens a TextBlock, ReasoningBlock or ToolUseBlock. Block carries
// the metadata known up front, such as a tool's ID and Name, and may carry
// initial text.
type BlockStart struct {
	Index int
	Block Block
}

// TextDelta appends text to a TextBlock.
type TextDelta struct {
	Index int
	Text  string
}

// ReasoningDelta appends text to a ReasoningBlock.
type ReasoningDelta struct {
	Index int
	Text  string
}

// ToolUseDelta appends a fragment of a ToolUseBlock's JSON arguments.
type ToolUseDelta struct {
	Index     int
	Arguments string
}

// BlockEnd closes a block. Providers set Block only to deliver metadata that
// arrives late (signatures, redacted data, annotations, provider extras): its
// non-empty metadata fields replace the started ones, while its text and
// arguments are ignored. Streams read through Handle, Collect or a Client
// deliver the completed block instead.
type BlockEnd struct {
	Index int
	Block Block
}

// UsageEvent carries cumulative usage; a later event replaces an earlier one.
type UsageEvent struct {
	Usage Usage
}

// WarningEvent carries a Warning; it is also collected into Response.Warnings.
type WarningEvent struct {
	Warning Warning
}

// DoneEvent ends a successful stream.
type DoneEvent struct {
	FinishReason    FinishReason
	FinishReasonRaw string
	Provider        string
	Model           string
}

// ProviderEvent exposes a provider-native event that has no typed equivalent.
// It does not affect the aggregated Response.
type ProviderEvent struct {
	Name string
	Raw  json.RawMessage
}

func (BlockStart) isEvent()     {}
func (TextDelta) isEvent()      {}
func (ReasoningDelta) isEvent() {}
func (ToolUseDelta) isEvent()   {}
func (BlockEnd) isEvent()       {}
func (UsageEvent) isEvent()     {}
func (WarningEvent) isEvent()   {}
func (DoneEvent) isEvent()      {}
func (ProviderEvent) isEvent()  {}

func cloneEvent(event Event) Event {
	switch e := event.(type) {
	case BlockStart:
		e.Block = cloneBlock(e.Block)
		return e
	case BlockEnd:
		e.Block = cloneBlock(e.Block)
		return e
	case UsageEvent:
		e.Usage = e.Usage.Clone()
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
	state    *collector
	done     bool
}

func newValidatedStream(provider, model string, stream Stream) Stream {
	if stream == nil {
		return nil
	}
	state := newCollector()
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
		return nil, WrapError(s.provider, ErrorTypeProvider, err)
	}
	if event == nil {
		s.done = true
		return nil, NewError(s.provider, ErrorTypeInternal, "stream returned nil event without error", nil)
	}
	event, done, err := s.state.Apply(event)
	if err == nil && done {
		err = validateResponse(&Response{Blocks: s.state.blocks, Provider: s.state.provider, Model: s.state.model}, s.provider, s.state.model)
	}
	if err != nil {
		s.done = true
		return nil, WrapError(s.provider, ErrorTypeProvider, err)
	}
	s.done = done
	return event, nil
}

// Client streams share their validated accumulator with Handle, avoiding a
// second full copy of streamed content. External streams are collected locally.
func (s *validatedStream) eventCollector() *collector { return s.state }

func streamCollector(stream Stream) *collector {
	if source, ok := stream.(interface{ eventCollector() *collector }); ok {
		return source.eventCollector()
	}
	return nil
}

func (s *validatedStream) Close() error {
	s.done = true
	err := s.inner.Close()
	if err != nil {
		return WrapError(s.provider, ErrorTypeProvider, err)
	}
	return nil
}

// Collect consumes the stream and returns the aggregated Response. On failure it
// returns the partial response together with the error. EOF before DoneEvent is
// io.ErrUnexpectedEOF; partial responses must not be treated as completed output.
func Collect(stream Stream) (*Response, error) {
	return Handle(stream, nil)
}

// Handle consumes the stream, invoking fn for each event as it arrives, and
// returns the aggregated Response. Events are aggregated before fn runs, so a
// BlockEnd passed to fn carries the completed block. A nil fn behaves exactly
// like Collect. On any failure Handle returns the partial response and the
// error. The caller still owns Close.
func Handle(stream Stream, fn func(Event) error) (*Response, error) {
	if stream == nil {
		return nil, fmt.Errorf("stream cannot be nil")
	}
	state := streamCollector(stream)
	validated := state != nil
	if !validated {
		state = newCollector()
	}
	for {
		event, err := stream.Next()
		if err != nil {
			if errors.Is(err, io.EOF) {
				// A Client stream may have delivered Done through Next already.
				if validated && state.done {
					return state.Response(), nil
				}
				return state.Response(), fmt.Errorf("stream ended before Done event: %w", io.ErrUnexpectedEOF)
			}
			return state.Response(), err
		}
		if event == nil {
			return state.Response(), fmt.Errorf("stream returned nil event without error")
		}
		_, done := event.(DoneEvent)
		if !validated {
			event, done, err = state.Apply(event)
			if err != nil {
				return state.Response(), err
			}
		}
		if fn != nil {
			if err := fn(event); err != nil {
				return state.Response(), err
			}
		}
		if done {
			resp := state.Response()
			if !validated {
				if err := validateResponse(resp, resp.Provider, resp.Model); err != nil {
					return resp, err
				}
			}
			return resp, nil
		}
	}
}

// collector aggregates stream events into a Response and enforces the block
// lifecycle. Streamed text and arguments grow in builders until the block ends.
type collector struct {
	blocks    []Block
	closed    []bool
	builders  map[int]*strings.Builder
	done      bool
	usage     Usage
	finish    FinishReason
	finishRaw string
	provider  string
	model     string
	warnings  []Warning
}

func newCollector() *collector {
	return &collector{builders: make(map[int]*strings.Builder)}
}

// Apply records event and returns the event to deliver: a BlockEnd is replaced
// by the completed block, other events are passed through unchanged.
func (c *collector) Apply(event Event) (Event, bool, error) {
	if c.done {
		return nil, false, fmt.Errorf("stream event received after Done event")
	}
	switch e := event.(type) {
	case BlockStart:
		if e.Index != len(c.blocks) {
			return nil, false, fmt.Errorf("block %d started out of order; next block is %d", e.Index, len(c.blocks))
		}
		if blockKind(e.Block) == "" {
			return nil, false, fmt.Errorf("block %d: stream does not support %T", e.Index, e.Block)
		}
		c.blocks = append(c.blocks, cloneBlock(e.Block))
		c.closed = append(c.closed, false)
	case TextDelta:
		if err := c.grow(e.Index, "text", e.Text); err != nil {
			return nil, false, err
		}
	case ReasoningDelta:
		if err := c.grow(e.Index, "reasoning", e.Text); err != nil {
			return nil, false, err
		}
	case ToolUseDelta:
		if err := c.grow(e.Index, "tool_use", e.Arguments); err != nil {
			return nil, false, err
		}
	case BlockEnd:
		block, err := c.end(e)
		if err != nil {
			return nil, false, err
		}
		return BlockEnd{Index: e.Index, Block: block}, false, nil
	case UsageEvent:
		c.usage = e.Usage.Clone()
	case WarningEvent:
		c.warnings = append(c.warnings, e.Warning)
	case ProviderEvent:
	case DoneEvent:
		for i, closed := range c.closed {
			if !closed {
				return nil, false, fmt.Errorf("stream completed with block %d still open", i)
			}
		}
		c.done = true
		c.finish = e.FinishReason
		c.finishRaw = e.FinishReasonRaw
		if e.Provider != "" {
			c.provider = e.Provider
		}
		if e.Model != "" {
			c.model = e.Model
		}
		c.warnings = append(c.warnings, malformedToolArgumentWarnings(c.blocks, c.provider)...)
		return event, true, nil
	default:
		return nil, false, fmt.Errorf("unknown stream event %T", event)
	}
	return event, false, nil
}

func (c *collector) open(index int) error {
	if index < 0 || index >= len(c.blocks) {
		return fmt.Errorf("block %d was not started", index)
	}
	if c.closed[index] {
		return fmt.Errorf("block %d already ended", index)
	}
	return nil
}

func (c *collector) grow(index int, kind, text string) error {
	if err := c.open(index); err != nil {
		return err
	}
	if got := blockKind(c.blocks[index]); got != kind {
		return fmt.Errorf("block %d is %s, not %s", index, got, kind)
	}
	if text == "" {
		return nil
	}
	builder := c.builders[index]
	if builder == nil {
		builder = &strings.Builder{}
		builder.WriteString(blockContent(c.blocks[index]))
		c.builders[index] = builder
	}
	builder.WriteString(text)
	return nil
}

func (c *collector) end(e BlockEnd) (Block, error) {
	if err := c.open(e.Index); err != nil {
		return nil, err
	}
	block := c.flushed(e.Index)
	if e.Block != nil {
		merged, err := mergeBlockMetadata(block, cloneBlock(e.Block))
		if err != nil {
			return nil, fmt.Errorf("block %d: %w", e.Index, err)
		}
		block = merged
	}
	// An argument-less call streams no deltas; keep its arguments valid JSON.
	if tool, ok := block.(ToolUseBlock); ok && len(tool.Arguments) == 0 {
		tool.Arguments = json.RawMessage("{}")
		block = tool
	}
	c.blocks[e.Index] = block
	delete(c.builders, e.Index)
	c.closed[e.Index] = true
	return cloneBlock(block), nil
}

// flushed returns the block at index with its streamed content applied.
func (c *collector) flushed(index int) Block {
	block := c.blocks[index]
	builder := c.builders[index]
	if builder == nil {
		return block
	}
	switch b := block.(type) {
	case TextBlock:
		b.Text = builder.String()
		return b
	case ReasoningBlock:
		b.Text = builder.String()
		return b
	case ToolUseBlock:
		b.Arguments = json.RawMessage(builder.String())
		return b
	}
	return block
}

func (c *collector) Response() *Response {
	blocks := make([]Block, len(c.blocks))
	for i := range c.blocks {
		blocks[i] = cloneBlock(c.flushed(i))
	}
	if len(blocks) == 0 {
		blocks = nil
	}
	// The provider of an external stream is known only once Done arrives.
	warnings := append([]Warning(nil), c.warnings...)
	for i := range warnings {
		if warnings[i].Provider == "" {
			warnings[i].Provider = c.provider
		}
	}
	return &Response{
		Blocks:          blocks,
		Usage:           c.usage.Clone(),
		Model:           c.model,
		Provider:        c.provider,
		FinishReason:    c.finish,
		FinishReasonRaw: c.finishRaw,
		Warnings:        warnings,
	}
}

func blockKind(block Block) string {
	switch block.(type) {
	case TextBlock:
		return "text"
	case ReasoningBlock:
		return "reasoning"
	case ToolUseBlock:
		return "tool_use"
	default:
		return ""
	}
}

func blockContent(block Block) string {
	switch b := block.(type) {
	case TextBlock:
		return b.Text
	case ReasoningBlock:
		return b.Text
	case ToolUseBlock:
		return string(b.Arguments)
	default:
		return ""
	}
}

// mergeBlockMetadata applies the non-empty metadata of a BlockEnd snapshot.
func mergeBlockMetadata(block, final Block) (Block, error) {
	switch b := block.(type) {
	case TextBlock:
		f, ok := final.(TextBlock)
		if !ok {
			break
		}
		if f.Annotations != nil {
			b.Annotations = f.Annotations
		}
		if f.Logprobs != nil {
			b.Logprobs = f.Logprobs
		}
		if f.Signature != "" {
			b.Signature = f.Signature
		}
		return b, nil
	case ReasoningBlock:
		f, ok := final.(ReasoningBlock)
		if !ok {
			break
		}
		if f.Signature != "" {
			b.Signature = f.Signature
		}
		if f.Redacted != nil {
			b.Redacted = f.Redacted
		}
		if f.Extra != nil {
			b.Extra = f.Extra
		}
		return b, nil
	case ToolUseBlock:
		f, ok := final.(ToolUseBlock)
		if !ok {
			break
		}
		if f.ID != "" {
			b.ID = f.ID
		}
		if f.Name != "" {
			b.Name = f.Name
		}
		if f.Signature != "" {
			b.Signature = f.Signature
		}
		return b, nil
	}
	return nil, fmt.Errorf("end snapshot %T does not match %T", final, block)
}
