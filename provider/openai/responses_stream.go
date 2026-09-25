package openai

import (
	"cmp"
	"encoding/json"
	"errors"
	"io"
	"net/http"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/internal/wire"
)

// responsesEvent holds the fields of every Responses stream event used here.
type responsesEvent struct {
	Type         string                `json:"type"`
	Sequence     int                   `json:"sequence_number"`
	OutputIndex  int                   `json:"output_index"`
	ContentIndex int                   `json:"content_index"`
	SummaryIndex int                   `json:"summary_index"`
	Delta        string                `json:"delta"`
	Arguments    string                `json:"arguments"`
	Item         *responsesOutputItem  `json:"item"`
	Part         *responsesContentPart `json:"part"`
	Response     *responsesResponse    `json:"response"`
	// The error event uses a flat shape or a nested error object.
	Code    string `json:"code"`
	Message string `json:"message"`
	Error   *struct {
		Type    string `json:"type"`
		Code    string `json:"code"`
		Message string `json:"message"`
	} `json:"error"`
}

// A block is an output item, or a content part of a message item.
type partKey struct{ output, content int }

func itemKey(output int) partKey { return partKey{output, -1} }

type responsesStream struct {
	resp      *http.Response
	sse       *wire.SSEReader
	pending   []litellm.Event
	done      bool
	requested string // the requested model, for ProviderState
	model     string
	blocks    wire.BlockTracker[partKey]
	messages  map[int]itemState // message items by output index
	tools     map[int]litellm.ToolUseBlock
	streamed  map[int]bool // function calls whose arguments arrived as deltas
	summaries map[int]int  // last summary index per reasoning item
	toolCalls bool
	refused   bool
	sequence  int
}

func newResponsesStream(resp *http.Response, model string) *responsesStream {
	return &responsesStream{
		resp:      resp,
		sse:       wire.NewSSEReader(resp.Body, "openai"),
		requested: model,
		model:     model,
		messages:  make(map[int]itemState),
		tools:     make(map[int]litellm.ToolUseBlock),
		streamed:  make(map[int]bool),
		summaries: make(map[int]int),
	}
}

func (s *responsesStream) Next() (event litellm.Event, err error) {
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
			return nil, litellm.NewError("openai", litellm.ErrorTypeProvider, "stream ended before response.completed", nil)
		}
		if err != nil {
			return nil, err
		}
		var e responsesEvent
		if err := json.Unmarshal([]byte(frame.Data), &e); err != nil {
			return nil, litellm.NewError("openai", litellm.ErrorTypeProvider, "parse stream event", err)
		}
		e.Type = cmp.Or(frame.Name, e.Type)
		// Resumed streams may repeat events; sequence numbers identify them.
		if e.Sequence != 0 {
			if e.Sequence <= s.sequence {
				continue
			}
			s.sequence = e.Sequence
		}
		if s.pending, err = s.events(s.pending, e, json.RawMessage(frame.Data)); err != nil {
			return nil, err
		}
	}
	event = s.pending[0]
	s.pending = s.pending[1:]
	return event, nil
}

func (s *responsesStream) Close() error {
	s.done = true
	s.pending = nil
	return s.resp.Body.Close()
}

func (s *responsesStream) events(events []litellm.Event, e responsesEvent, raw json.RawMessage) ([]litellm.Event, error) {
	switch e.Type {
	case "response.output_item.added":
		if e.Item != nil {
			switch e.Item.Type {
			case "message":
				s.messages[e.OutputIndex] = itemState{e.Item.ID, e.Item.Phase}
				return events, nil
			case "function_call":
				tool := litellm.ToolUseBlock{ID: e.Item.CallID, Name: e.Item.Name, State: itemState{ID: e.Item.ID}.state(s.requested)}
				s.tools[e.OutputIndex] = tool
				s.toolCalls = true
				events, _ = s.blocks.Open(events, itemKey(e.OutputIndex), tool)
				return events, nil
			}
		}
	case "response.output_item.done":
		if e.Item != nil {
			switch e.Item.Type {
			case "function_call":
				final := litellm.ToolUseBlock{ID: e.Item.CallID, Name: e.Item.Name, State: itemState{ID: e.Item.ID}.state(s.requested)}
				return s.blocks.Close(events, itemKey(e.OutputIndex), final), nil
			case "reasoning":
				block := reasoningBlock(*e.Item, s.requested)
				events, _ = s.blocks.Open(events, itemKey(e.OutputIndex), litellm.ReasoningBlock{Summary: block.Summary})
				return s.blocks.Close(events, itemKey(e.OutputIndex), litellm.ReasoningBlock{State: block.State}), nil
			}
		}
	case "response.content_part.added":
		if e.Part != nil && (e.Part.Type == "output_text" || e.Part.Type == "refusal") {
			events, _ = s.blocks.Open(events, partKey{e.OutputIndex, e.ContentIndex}, s.text(e.OutputIndex))
			return events, nil
		}
	case "response.content_part.done":
		if e.Part != nil && (e.Part.Type == "output_text" || e.Part.Type == "refusal") {
			final, _, _ := contentPartBlock(*e.Part)
			return s.blocks.Close(events, partKey{e.OutputIndex, e.ContentIndex}, final), nil
		}
	case "response.output_text.delta", "response.refusal.delta":
		s.refused = s.refused || e.Type == "response.refusal.delta"
		events, index := s.blocks.Open(events, partKey{e.OutputIndex, e.ContentIndex}, s.text(e.OutputIndex))
		return append(events, litellm.TextDelta{Index: index, Text: e.Delta}), nil
	case "response.reasoning_summary_text.delta":
		events, index := s.blocks.Open(events, itemKey(e.OutputIndex), litellm.ReasoningBlock{Summary: true})
		// Summary parts are joined by newlines, as in a complete response.
		if last, ok := s.summaries[e.OutputIndex]; ok && last != e.SummaryIndex {
			e.Delta = "\n" + e.Delta
		}
		s.summaries[e.OutputIndex] = e.SummaryIndex
		return append(events, litellm.ReasoningDelta{Index: index, Text: e.Delta}), nil
	case "response.reasoning_text.delta":
		events, index := s.blocks.Open(events, itemKey(e.OutputIndex), litellm.ReasoningBlock{})
		return append(events, litellm.ReasoningDelta{Index: index, Text: e.Delta}), nil
	case "response.function_call_arguments.delta":
		s.streamed[e.OutputIndex] = true
		return s.toolDelta(events, e.OutputIndex, e.Delta), nil
	case "response.function_call_arguments.done":
		if !s.streamed[e.OutputIndex] {
			return s.toolDelta(events, e.OutputIndex, e.Arguments), nil
		}
		return events, nil
	case "response.completed", "response.incomplete":
		var status, reason string
		var usage litellm.Usage
		if r := e.Response; r != nil {
			s.model = cmp.Or(r.Model, s.model)
			status, usage = r.Status, convertResponsesUsage(r.Usage)
			if r.IncompleteDetails != nil {
				reason = r.IncompleteDetails.Reason
			}
		}
		status = cmp.Or(status, e.Type[len("response."):])
		reasonCode, reasonRaw := finish(status, reason, s.toolCalls, s.refused)
		events = append(events, litellm.UsageEvent{Usage: usage})
		events = s.blocks.CloseAll(events, nil)
		s.done = true
		return append(events, litellm.DoneEvent{FinishReason: reasonCode, FinishReasonRaw: reasonRaw, Provider: "openai", Model: s.model}), nil
	case "response.failed":
		var code, message string
		if e.Response != nil && e.Response.Error != nil {
			code, message = e.Response.Error.Code, e.Response.Error.Message
		}
		return nil, wire.StreamError("openai", code, "response failed: "+message)
	case "error":
		code, message := e.Code, e.Message
		if e.Error != nil {
			code, message = cmp.Or(e.Error.Code, e.Error.Type, code), cmp.Or(e.Error.Message, message)
		}
		return nil, wire.StreamError("openai", code, "stream error: "+message)
	case "":
		return nil, litellm.NewError("openai", litellm.ErrorTypeProvider, "stream event missing type", nil)
	}
	return append(events, litellm.ProviderEvent{Name: e.Type, Raw: raw}), nil
}

// text opens a content part of the message item at output.
func (s *responsesStream) text(output int) litellm.TextBlock {
	return litellm.TextBlock{State: s.messages[output].state(s.requested)}
}

func (s *responsesStream) toolDelta(events []litellm.Event, output int, arguments string) []litellm.Event {
	if arguments == "" {
		return events
	}
	events, index := s.blocks.Open(events, itemKey(output), s.tools[output])
	return append(events, litellm.ToolUseDelta{Index: index, Arguments: arguments})
}
