package openaicompat

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"net/http"

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
	requested string // the requested model, for ProviderState
	model     string
	finish    litellm.FinishReason
	finishRaw string
	refused   bool
	blocks    wire.BlockTracker[blockKey]
	tools     map[int]*toolState
	// details accumulates reasoning_details, delivered when the block ends.
	details     []map[string]json.RawMessage
	logprobs    *chatLogprobs
	annotations []json.RawMessage
}

func newStream(resp *http.Response, req *litellm.Request, spec Spec) *stream {
	return &stream{
		resp:      resp,
		sse:       wire.NewSSEReader(resp.Body, spec.Name),
		spec:      spec,
		requested: req.Model,
		model:     req.Model,
		tools:     make(map[int]*toolState),
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
			// Some vendors close the stream after the finish chunk without
			// [DONE]; only an EOF before any finish reason is a truncation.
			if s.finishRaw == "" {
				return nil, litellm.NewError(s.spec.Name, litellm.ErrorTypeProvider, "stream ended before a finish reason", nil)
			}
			s.end()
			break
		}
		if err != nil {
			return nil, err
		}
		if frame.Data == "[DONE]" {
			s.end()
			break
		}
		var chunk streamChunk
		if err := json.Unmarshal([]byte(frame.Data), &chunk); err != nil {
			return nil, litellm.NewError(s.spec.Name, litellm.ErrorTypeProvider, "parse stream chunk", err)
		}
		if err := wire.ErrorField(s.spec.Name, chunk.Error); err != nil {
			return nil, err
		}
		s.pending = s.events(s.pending, chunk)
	}
	event = s.pending[0]
	s.pending = s.pending[1:]
	return event, nil
}

// end queues the remaining block ends and the DoneEvent.
func (s *stream) end() {
	s.pending = s.closeAll(s.pending)
	finish := s.finish
	if s.refused {
		finish = litellm.FinishReasonSafety
	}
	s.pending = append(s.pending, litellm.DoneEvent{FinishReason: finish, FinishReasonRaw: s.finishRaw, Provider: s.spec.Name, Model: s.model})
	s.done = true
}

func (s *stream) Close() error {
	s.done = true
	s.pending = nil
	return s.resp.Body.Close()
}

func (s *stream) events(events []litellm.Event, chunk streamChunk) []litellm.Event {
	if chunk.Model != "" {
		s.model = chunk.Model
	}
	if chunk.Usage != nil {
		events = append(events, litellm.UsageEvent{Usage: s.spec.usage(*chunk.Usage)})
	}
	for _, choice := range chunk.Choices {
		events = s.reasoning(events, choice.Delta.Fields)
		if choice.Delta.Content != "" {
			events = s.text(events, choice.Delta.Content)
		}
		if choice.Delta.Refusal != "" {
			events = s.text(events, choice.Delta.Refusal)
			s.refused = true
		}
		if choice.Logprobs != nil {
			if s.logprobs == nil {
				s.logprobs = choice.Logprobs
			} else {
				s.logprobs.Content = append(s.logprobs.Content, choice.Logprobs.Content...)
				s.logprobs.Refusal = append(s.logprobs.Refusal, choice.Logprobs.Refusal...)
			}
		}
		s.annotations = append(s.annotations, choice.Delta.Annotations...)
		for i, call := range choice.Delta.ToolCalls {
			events = s.tool(events, i, call)
		}
		if choice.FinishReason != "" {
			s.finish, s.finishRaw = wire.FinishReason(choice.FinishReason), choice.FinishReason
			events = s.closeAll(events)
		}
	}
	return events
}

func (s *stream) text(events []litellm.Event, text string) []litellm.Event {
	events, index := s.blocks.Open(events, blockKey{kind: textKind}, litellm.TextBlock{})
	return append(events, litellm.TextDelta{Index: index, Text: text})
}

func (s *stream) reasoning(events []litellm.Event, fields map[string]json.RawMessage) []litellm.Event {
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
	if text == "" && extra == nil {
		return events
	}
	events, index := s.blocks.Open(events, blockKey{kind: reasoningKind}, litellm.ReasoningBlock{})
	if extra != nil {
		s.details = addDetails(s.details, extra)
	}
	if text != "" {
		events = append(events, litellm.ReasoningDelta{Index: index, Text: text})
	}
	return events
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
// id and name and the merged reasoning_details.
func (s *stream) closeAll(events []litellm.Event) []litellm.Event {
	events = s.blocks.CloseAll(events, func(key blockKey) litellm.Block {
		switch key.kind {
		case textKind:
			if s.logprobs != nil || len(s.annotations) > 0 {
				block := litellm.TextBlock{Annotations: Annotations(s.annotations)}
				if s.logprobs != nil {
					block.Logprobs, _ = json.Marshal(s.logprobs) // tokens are valid JSON from decoded chunks
				}
				return block
			}
		case toolKind:
			state := s.tools[key.call]
			return litellm.ToolUseBlock{ID: state.id, Name: state.name}
		case reasoningKind:
			if len(s.details) > 0 {
				details, _ := json.Marshal(s.details)
				return litellm.ReasoningBlock{State: wire.NewState(s.spec.Name, s.requested, json.RawMessage(details))}
			}
		}
		return nil
	})
	clear(s.tools)
	s.details = nil
	s.logprobs = nil
	s.annotations = nil
	return events
}

// addDetails folds one chunk's reasoning_details into entries. Vendors split
// an entry across adjacent fragments that repeat its type and index: text and
// summary arrive in pieces, fields such as the signature once. Encrypted
// fragments are always whole. The merged entries match the non-streaming
// response and are what replay expects.
func addDetails(entries []map[string]json.RawMessage, chunk json.RawMessage) []map[string]json.RawMessage {
	var fragments []map[string]json.RawMessage
	if json.Unmarshal(chunk, &fragments) != nil {
		return entries
	}
	for _, fragment := range fragments {
		n := len(entries)
		if n == 0 || !continues(entries[n-1], fragment) {
			entries = append(entries, fragment)
			continue
		}
		for key, value := range fragment {
			if key == "text" || key == "summary" {
				value = appendJSONString(entries[n-1][key], value)
			} else if entries[n-1][key] != nil {
				continue
			}
			entries[n-1][key] = value
		}
	}
	return entries
}

func continues(entry, fragment map[string]json.RawMessage) bool {
	return string(fragment["type"]) != `"reasoning.encrypted"` &&
		bytes.Equal(entry["type"], fragment["type"]) && bytes.Equal(entry["index"], fragment["index"])
}

// appendJSONString concatenates two JSON strings; a non-string b replaces a.
func appendJSONString(a, b json.RawMessage) json.RawMessage {
	var left, right string
	if json.Unmarshal(b, &right) != nil {
		return b
	}
	_ = json.Unmarshal(a, &left)
	out, _ := json.Marshal(left + right)
	return out
}
