package openaicompat

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/internal/wire"
)

// Chat Completions has no block index: a choice carries at most one reasoning
// run and one text run, and tool calls are keyed by their own index.
type blockKey struct {
	kind int
	call int
}

const (
	reasoningKind = iota
	textKind
	toolKind
)

type toolState struct {
	id, name string
}

type stream struct {
	resp      *http.Response
	sse       *wire.SSEReader
	spec      Spec
	pending   []litellm.Event
	done      bool
	model     string
	finish    litellm.FinishReason
	finishRaw string
	refused   bool
	blocks    wire.BlockTracker[blockKey]
	tools     map[int]*toolState
	// Reasoning extras accumulate here and are delivered when the block ends.
	extra         []json.RawMessage
	lastText      string
	lastReasoning string
}

func newStream(resp *http.Response, req *litellm.Request, spec Spec) *stream {
	return &stream{
		resp:  resp,
		sse:   wire.NewSSEReader(resp.Body, spec.Name),
		spec:  spec,
		model: req.Model,
		tools: make(map[int]*toolState),
	}
}

func (s *stream) Next() (event litellm.Event, err error) {
	defer func() {
		if err != nil {
			s.done = true
		}
	}()
	for len(s.pending) == 0 {
		if s.done {
			return nil, io.EOF
		}
		frame, err := s.sse.Next()
		if errors.Is(err, io.EOF) {
			return nil, litellm.NewError(s.spec.Name, litellm.ErrorTypeProvider, "stream ended before [DONE]", nil)
		}
		if err != nil {
			return nil, err
		}
		if frame.Data == "[DONE]" {
			s.pending = s.closeAll(s.pending)
			finish := s.finish
			if s.refused {
				finish = litellm.FinishReasonSafety
			}
			s.pending = append(s.pending, litellm.DoneEvent{FinishReason: finish, FinishReasonRaw: s.finishRaw, Provider: s.spec.Name, Model: s.model})
			s.done = true
			break
		}
		var chunk streamChunk
		if err := json.Unmarshal([]byte(frame.Data), &chunk); err != nil {
			return nil, litellm.NewError(s.spec.Name, litellm.ErrorTypeProvider, "parse stream chunk", err)
		}
		if err := wire.ChunkError(s.spec.Name, chunk.Error); err != nil {
			return nil, err
		}
		if s.pending, err = s.events(s.pending, chunk); err != nil {
			return nil, err
		}
	}
	event = s.pending[0]
	s.pending = s.pending[1:]
	return event, nil
}

func (s *stream) Close() error {
	s.done = true
	s.pending = nil
	return s.resp.Body.Close()
}

func (s *stream) events(events []litellm.Event, chunk streamChunk) ([]litellm.Event, error) {
	if chunk.Model != "" {
		s.model = chunk.Model
	}
	if chunk.Usage != nil {
		events = append(events, litellm.UsageEvent{Usage: convertUsage(*chunk.Usage)})
	}
	for _, choice := range chunk.Choices {
		var err error
		if events, err = s.reasoning(events, choice.Delta.Fields); err != nil {
			return nil, err
		}
		text, err := s.increment(choice.Delta.Content, &s.lastText)
		if err != nil {
			return nil, err
		}
		if text != "" {
			events = s.text(events, text)
		}
		if choice.Delta.Refusal != "" {
			events = s.text(events, choice.Delta.Refusal)
			s.refused = true
		}
		for i, call := range choice.Delta.ToolCalls {
			events = s.tool(events, i, call)
		}
		if choice.FinishReason != "" {
			s.finish, s.finishRaw = wire.FinishReason(choice.FinishReason), choice.FinishReason
			events = s.closeAll(events)
		}
	}
	return events, nil
}

func (s *stream) text(events []litellm.Event, text string) []litellm.Event {
	events, index := s.blocks.Open(events, blockKey{kind: textKind}, litellm.TextBlock{})
	return append(events, litellm.TextDelta{Index: index, Text: text})
}

func (s *stream) reasoning(events []litellm.Event, fields map[string]json.RawMessage) ([]litellm.Event, error) {
	var text string
	var extra json.RawMessage
	for _, field := range s.spec.ReasoningFields {
		raw := fields[field]
		if field == "reasoning_details" && len(raw) > 0 && string(raw) != "null" {
			extra = raw
		}
		if text == "" {
			text = reasoningText(raw)
		}
	}
	text, err := s.increment(text, &s.lastReasoning)
	if err != nil {
		return nil, err
	}
	if text == "" && extra == nil {
		return events, nil
	}
	events, index := s.blocks.Open(events, blockKey{kind: reasoningKind}, litellm.ReasoningBlock{})
	if extra != nil {
		if s.spec.CumulativeStream {
			s.extra = []json.RawMessage{extra}
		} else {
			s.extra = append(s.extra, extra)
		}
	}
	if text != "" {
		events = append(events, litellm.ReasoningDelta{Index: index, Text: text})
	}
	return events, nil
}

// increment turns a cumulative snapshot into the newly added text.
func (s *stream) increment(text string, last *string) (string, error) {
	if !s.spec.CumulativeStream || text == "" {
		return text, nil
	}
	next, ok := strings.CutPrefix(text, *last)
	if !ok {
		return "", litellm.NewError(s.spec.Name, litellm.ErrorTypeProvider, "cumulative stream changed unexpectedly", nil)
	}
	*last = text
	return next, nil
}

func (s *stream) tool(events []litellm.Event, position int, call toolCallDelta) []litellm.Event {
	index := position
	if call.Index != nil {
		index = *call.Index
	}
	key := blockKey{kind: toolKind, call: index}
	state := s.tools[index]
	if state == nil {
		if call.ID == "" && call.Function.Name == "" && call.Function.Arguments == "" {
			return events
		}
		state = &toolState{id: call.ID, name: call.Function.Name}
		s.tools[index] = state
		events, _ = s.blocks.Open(events, key, litellm.ToolUseBlock{ID: call.ID, Name: call.Function.Name})
	}
	// Some gateways send the id or name after the opening chunk.
	if state.id == "" {
		state.id = call.ID
	}
	if state.name == "" {
		state.name = call.Function.Name
	}
	if call.Function.Arguments != "" {
		if blockIndex, ok := s.blocks.Index(key); ok {
			events = append(events, litellm.ToolUseDelta{Index: blockIndex, Arguments: call.Function.Arguments})
		}
	}
	return events
}

// closeAll ends every open block in index order, delivering each tool's final
// id and name and the joined reasoning extras.
func (s *stream) closeAll(events []litellm.Event) []litellm.Event {
	events = s.blocks.CloseAll(events, func(key blockKey) litellm.Block {
		switch key.kind {
		case toolKind:
			state := s.tools[key.call]
			return litellm.ToolUseBlock{ID: state.id, Name: state.name}
		case reasoningKind:
			if len(s.extra) > 0 {
				return litellm.ReasoningBlock{Extra: joinExtra(s.extra)}
			}
		}
		return nil
	})
	clear(s.tools)
	s.extra = nil
	return events
}

// joinExtra concatenates reasoning_details arrays streamed across chunks.
func joinExtra(parts []json.RawMessage) json.RawMessage {
	if len(parts) == 1 {
		return parts[0]
	}
	var items []json.RawMessage
	for _, part := range parts {
		var chunk []json.RawMessage
		if json.Unmarshal(part, &chunk) != nil {
			return parts[len(parts)-1]
		}
		items = append(items, chunk...)
	}
	data, _ := json.Marshal(items)
	return data
}
