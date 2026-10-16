package gemini

import (
	"encoding/json"
	"net/http"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/internal/wire"
)

type stream struct {
	*wire.Stream
	sse       *wire.SSEReader
	name      string // the provider's
	model     string
	blocks    wire.BlockTracker[int] // key: run number
	run       int
	open      string // kind of the open text or thought run
	signature string // signature of the open run, delivered when it ends
}

func newStream(resp *http.Response, name, model string) *stream {
	reader := wire.NewSSEReader(resp.Body, name)
	reader.AcceptBare = true
	s := &stream{sse: reader, name: name, model: model}
	s.Stream = wire.NewStream(resp.Body, s.read)
	return s
}

func (s *stream) read(events []litellm.Event) ([]litellm.Event, error) {
	frame, err := s.sse.Next()
	if err != nil {
		return nil, err
	}
	var chunk response
	if err := json.Unmarshal([]byte(frame.Data), &chunk); err != nil {
		return nil, litellm.NewError(s.name, litellm.ErrorTypeProvider, "parse stream chunk", err)
	}
	if err := wire.ErrorField(s.name, chunk.Error); err != nil {
		return nil, err
	}
	events, err = s.events(events, chunk)
	if err != nil {
		return nil, litellm.NewError(s.name, litellm.ErrorTypeProvider, "parse stream chunk", err)
	}
	return events, nil
}

func (s *stream) events(events []litellm.Event, chunk response) ([]litellm.Event, error) {
	if chunk.UsageMetadata != nil {
		events = append(events, litellm.UsageEvent{Usage: convertUsage(chunk.UsageMetadata)})
	}
	if len(chunk.Candidates) == 0 {
		if chunk.PromptFeedback != nil && chunk.PromptFeedback.BlockReason != "" {
			return s.finish(events, chunk.PromptFeedback.BlockReason, wire.FinishReason(chunk.PromptFeedback.BlockReason)), nil
		}
		return events, nil
	}
	candidate := chunk.Candidates[0]
	for _, p := range candidate.Content.Parts {
		block, err := partBlock(p, s.name, s.model)
		if err != nil {
			return nil, err
		}
		switch b := block.(type) {
		case litellm.TextBlock:
			events = s.extendRun(events, "text", b.Text, p.ThoughtSignature)
		case litellm.ReasoningBlock:
			events = s.extendRun(events, "reasoning", b.Text, p.ThoughtSignature)
		case litellm.ToolUseBlock:
			events = s.endRun(events)
			if p.FunctionCall.ID == "" {
				events = append(events, litellm.WarningEvent{Warning: generatedIDWarning(s.name, b)})
			}
			s.run++
			var index int
			events, index = s.blocks.Open(events, s.run, litellm.ToolUseBlock{ID: b.ID, Name: b.Name, State: b.State})
			events = append(events, litellm.ToolUseDelta{Index: index, Arguments: b.Arguments})
			events = s.blocks.Close(events, s.run, nil)
		case litellm.ImageBlock:
			// An image part is taken to arrive whole, its start carrying all
			// of it: the API reference does not say whether a stream splits one.
			events = s.endRun(events)
			s.run++
			events, _ = s.blocks.Open(events, s.run, b)
			events = s.blocks.Close(events, s.run, nil)
		case nil:
			if w, ok := unsupportedData(s.name, p); ok {
				events = append(events, litellm.WarningEvent{Warning: w})
			}
		}
	}
	if w, ok := finishMessage(s.name, candidate); ok {
		events = append(events, litellm.WarningEvent{Warning: w})
	}
	if candidate.FinishReason != "" {
		return s.finish(events, candidate.FinishReason, wire.FinishReason(candidate.FinishReason)), nil
	}
	return events, nil
}

// extendRun joins unsigned deltas, preserving every signed part's boundaries,
// including signature-only parts with empty text.
func (s *stream) extendRun(events []litellm.Event, kind, text, signature string) []litellm.Event {
	if s.open != kind || s.signature != "" || signature != "" {
		events = s.endRun(events)
		s.run++
		s.open = kind
		var block litellm.Block = litellm.TextBlock{}
		if kind == "reasoning" {
			block = litellm.ReasoningBlock{}
		}
		events, _ = s.blocks.Open(events, s.run, block)
	}
	if text != "" {
		index, _ := s.blocks.Index(s.run)
		if kind == "reasoning" {
			events = append(events, litellm.ReasoningDelta{Index: index, Text: text})
		} else {
			events = append(events, litellm.TextDelta{Index: index, Text: text})
		}
	}
	if signature != "" {
		s.signature = signature
	}
	return events
}

// endRun closes the open run, delivering its signature if any.
func (s *stream) endRun(events []litellm.Event) []litellm.Event {
	if s.open == "" {
		return events
	}
	var final litellm.Block
	if state := signed(s.name, s.model, s.signature); state != nil {
		final = litellm.TextBlock{State: state}
		if s.open == "reasoning" {
			final = litellm.ReasoningBlock{State: state}
		}
	}
	s.open, s.signature = "", ""
	return s.blocks.Close(events, s.run, final)
}

func (s *stream) finish(events []litellm.Event, raw string, reason litellm.FinishReason) []litellm.Event {
	events = s.endRun(events)
	events = s.blocks.CloseAll(events, nil)
	return append(events, litellm.DoneEvent{FinishReason: reason, FinishReasonRaw: raw, Provider: s.name, Model: s.model})
}
