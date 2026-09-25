package bedrock

import (
	"bufio"
	"encoding/json"
	"errors"
	"io"
	"net/http"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/internal/wire"
)

// streamEvent holds the fields of the Converse stream events used here; the
// event type travels in the :event-type header.
type streamEvent struct {
	ContentBlockIndex int `json:"contentBlockIndex"`
	Start             struct {
		ToolUse *struct {
			ToolUseID string `json:"toolUseId"`
			Name      string `json:"name"`
		} `json:"toolUse"`
	} `json:"start"`
	Delta struct {
		Text             *string `json:"text"`
		ReasoningContent *struct {
			Text            string `json:"text"`
			Signature       string `json:"signature"`
			RedactedContent []byte `json:"redactedContent"`
		} `json:"reasoningContent"`
		ToolUse *struct {
			Input string `json:"input"`
		} `json:"toolUse"`
	} `json:"delta"`
	StopReason string `json:"stopReason"`
	Usage      *usage `json:"usage"`
}

type stream struct {
	reader    *bufio.Reader
	response  *http.Response
	model     string
	pending   []litellm.Event
	done      bool
	finish    litellm.FinishReason
	finishRaw string
	blocks    wire.BlockTracker[int] // native index to litellm index
	reasoning map[int]reasoningState
}

func newStream(resp *http.Response, model string) *stream {
	return &stream{reader: bufio.NewReader(resp.Body), response: resp, model: model, reasoning: make(map[int]reasoningState)}
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
		msg, err := readEventStreamMessage(s.reader)
		switch {
		case errors.Is(err, io.EOF):
			return nil, litellm.NewError("bedrock", litellm.ErrorTypeProvider, "stream ended before metadata", nil)
		case errors.Is(err, errInvalidFrame):
			return nil, litellm.NewError("bedrock", litellm.ErrorTypeProvider, "read stream", err)
		case err != nil:
			return nil, litellm.NewNetworkError("bedrock", "read stream", err)
		}
		switch msg.headers[":message-type"] {
		case "exception":
			return nil, streamException(msg.headers[":exception-type"], msg.payload)
		case "error":
			return nil, wire.StreamError("bedrock", msg.headers[":error-code"], "stream error: "+msg.headers[":error-message"])
		}
		name := msg.headers[":event-type"]
		var e streamEvent
		if err := json.Unmarshal(msg.payload, &e); err != nil {
			return nil, litellm.NewError("bedrock", litellm.ErrorTypeProvider, "parse "+name, err)
		}
		s.pending = s.events(s.pending, name, e, msg.payload)
	}
	event = s.pending[0]
	s.pending = s.pending[1:]
	return event, nil
}

func (s *stream) Close() error {
	s.done = true
	s.pending = nil
	return s.response.Body.Close()
}

func (s *stream) events(events []litellm.Event, name string, e streamEvent, raw []byte) []litellm.Event {
	index := e.ContentBlockIndex
	switch name {
	case "contentBlockStart":
		if tool := e.Start.ToolUse; tool != nil {
			events, _ = s.blocks.Open(events, index, litellm.ToolUseBlock{ID: tool.ToolUseID, Name: tool.Name})
			return events
		}
	case "contentBlockDelta":
		switch d := e.Delta; {
		case d.Text != nil:
			events, i := s.blocks.Open(events, index, litellm.TextBlock{})
			return append(events, litellm.TextDelta{Index: i, Text: *d.Text})
		case d.ReasoningContent != nil:
			events, i := s.blocks.Open(events, index, litellm.ReasoningBlock{})
			r := s.reasoning[index]
			r.Signature += d.ReasoningContent.Signature
			r.RedactedContent = append(r.RedactedContent, d.ReasoningContent.RedactedContent...)
			s.reasoning[index] = r
			if d.ReasoningContent.Text != "" {
				events = append(events, litellm.ReasoningDelta{Index: i, Text: d.ReasoningContent.Text})
			}
			return events
		case d.ToolUse != nil:
			if i, ok := s.blocks.Index(index); ok {
				return append(events, litellm.ToolUseDelta{Index: i, Arguments: d.ToolUse.Input})
			}
		}
	case "contentBlockStop":
		return s.blocks.Close(events, index, s.final(index))
	case "messageStop":
		s.finish, s.finishRaw = wire.FinishReason(e.StopReason), e.StopReason
		return events
	case "metadata":
		if e.Usage != nil {
			events = append(events, litellm.UsageEvent{Usage: convertUsage(*e.Usage)})
		}
		events = s.blocks.CloseAll(events, s.final)
		s.done = true
		return append(events, litellm.DoneEvent{FinishReason: s.finish, FinishReasonRaw: s.finishRaw, Provider: "bedrock", Model: s.model})
	}
	return append(events, litellm.ProviderEvent{Name: "bedrock." + name, Raw: json.RawMessage(raw)})
}

// final returns the state of a reasoning block, whose signature or redacted
// content arrives as deltas.
func (s *stream) final(index int) litellm.Block {
	r, ok := s.reasoning[index]
	if !ok {
		return nil
	}
	delete(s.reasoning, index)
	return litellm.ReasoningBlock{State: wire.NewState("bedrock", s.model, r)}
}

// streamException maps a modeled exception frame, named by :exception-type
// (for example throttlingException) with a {"message": ...} payload.
func streamException(name string, raw []byte) error {
	var payload struct {
		Message string `json:"message"`
	}
	_ = json.Unmarshal(raw, &payload)
	message := payload.Message
	if message == "" {
		message = name
	}
	return wire.StreamError("bedrock", name, "stream error: "+message)
}
