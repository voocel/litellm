package openai

import (
	"cmp"
	"encoding/json"
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
	*wire.Stream
	sse       *wire.SSEReader
	name      string // the provider's
	requested string // the requested model, for ProviderState
	model     string
	blocks    wire.BlockTracker[partKey]
	messages  map[int]itemState             // message items by output index
	parts     map[partKey]litellm.TextBlock // final content metadata, awaiting the message's phase
	tools     map[int]litellm.ToolUseBlock
	streamed  map[int]bool // function calls whose arguments arrived as deltas
	summaries map[int]int  // last summary index per reasoning item
	refused   bool
	sequence  int
}

func newResponsesStream(resp *http.Response, name, model string) *responsesStream {
	s := &responsesStream{
		sse:       wire.NewSSEReader(resp.Body, name),
		name:      name,
		requested: model,
		model:     model,
		messages:  make(map[int]itemState),
		parts:     make(map[partKey]litellm.TextBlock),
		tools:     make(map[int]litellm.ToolUseBlock),
		streamed:  make(map[int]bool),
		summaries: make(map[int]int),
	}
	s.Stream = wire.NewStream(resp.Body, s.read)
	return s
}

func (s *responsesStream) read(events []litellm.Event) ([]litellm.Event, error) {
	frame, err := s.sse.Next()
	if err != nil {
		return nil, err
	}
	var e responsesEvent
	if err := json.Unmarshal([]byte(frame.Data), &e); err != nil {
		return nil, litellm.NewError(s.name, litellm.ErrorTypeProvider, "parse stream event", err)
	}
	e.Type = cmp.Or(frame.Name, e.Type)
	// Resumed streams may repeat events; sequence numbers identify them.
	if e.Sequence != 0 {
		if e.Sequence <= s.sequence {
			return events, nil
		}
		s.sequence = e.Sequence
	}
	return s.events(events, e, json.RawMessage(frame.Data))
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
				tool := litellm.ToolUseBlock{ID: e.Item.CallID, Name: e.Item.Name, State: itemState{ID: e.Item.ID}.state(s.name, s.requested)}
				s.tools[e.OutputIndex] = tool
				events, _ = s.blocks.Open(events, itemKey(e.OutputIndex), tool)
				return events, nil
			}
		}
	case "response.output_item.done":
		if e.Item != nil {
			switch e.Item.Type {
			case "message":
				previous := s.messages[e.OutputIndex]
				s.messages[e.OutputIndex] = itemState{cmp.Or(e.Item.ID, previous.ID), cmp.Or(e.Item.Phase, previous.Phase)}
				for i, part := range e.Item.Content {
					if final, refused, ok := contentPartBlock(part); ok {
						key := partKey{e.OutputIndex, i}
						final.Text = "" // content was already delivered by deltas
						s.parts[key] = final
						s.refused = s.refused || refused
						events = s.blocks.Close(events, key, s.finalText(key))
						delete(s.parts, key)
					}
				}
				delete(s.messages, e.OutputIndex)
				return events, nil
			case "function_call":
				final := litellm.ToolUseBlock{ID: e.Item.CallID, Name: e.Item.Name, State: itemState{ID: e.Item.ID}.state(s.name, s.requested)}
				return s.blocks.Close(events, itemKey(e.OutputIndex), final), nil
			case "reasoning":
				block := reasoningBlock(*e.Item, s.name, s.requested)
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
			// The message's phase can arrive after its content. Keep the block
			// open until output_item.done so BlockEnd carries all replay state.
			final, refused, _ := contentPartBlock(*e.Part)
			final.Text = ""
			s.parts[partKey{e.OutputIndex, e.ContentIndex}] = final
			s.refused = s.refused || refused
			return events, nil
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
		reasonCode, reasonRaw := finish(status, reason, s.refused)
		events = append(events, litellm.UsageEvent{Usage: usage})
		events = s.blocks.CloseAll(events, func(key partKey) litellm.Block {
			if key.content >= 0 {
				return s.finalText(key)
			}
			return nil
		})
		return append(events, litellm.DoneEvent{FinishReason: reasonCode, FinishReasonRaw: reasonRaw, Provider: s.name, Model: s.model}), nil
	case "response.failed":
		var code, message string
		if e.Response != nil && e.Response.Error != nil {
			code, message = e.Response.Error.Code, e.Response.Error.Message
		}
		return nil, wire.StreamError(s.name, code, "response failed: "+message)
	case "error":
		code, message := e.Code, e.Message
		if e.Error != nil {
			code, message = cmp.Or(e.Error.Code, e.Error.Type, code), cmp.Or(e.Error.Message, message)
		}
		return nil, wire.StreamError(s.name, code, "stream error: "+message)
	case "":
		return nil, litellm.NewError(s.name, litellm.ErrorTypeProvider, "stream event missing type", nil)
	}
	return append(events, litellm.ProviderEvent{Name: e.Type, Raw: raw}), nil
}

// text opens a content part of the message item at output.
func (s *responsesStream) text(output int) litellm.TextBlock {
	return litellm.TextBlock{State: s.messages[output].state(s.name, s.requested)}
}

func (s *responsesStream) finalText(key partKey) litellm.TextBlock {
	block := s.parts[key]
	block.State = s.messages[key.output].state(s.name, s.requested)
	return block
}

func (s *responsesStream) toolDelta(events []litellm.Event, output int, arguments string) []litellm.Event {
	if arguments == "" {
		return events
	}
	events, index := s.blocks.Open(events, itemKey(output), s.tools[output])
	return append(events, litellm.ToolUseDelta{Index: index, Arguments: arguments})
}
