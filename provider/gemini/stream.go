package gemini

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/internal/wire"
)

type stream struct {
	resp      *http.Response
	sse       *wire.SSEReader
	pending   []litellm.Event
	done      bool
	model     string
	blocks    wire.BlockTracker[int] // key: run number
	run       int
	open      string // kind of the open text or thought run
	signature string // signature of the open run, delivered when it ends
	toolCalls bool
}

func newStream(resp *http.Response, model string) *stream {
	reader := wire.NewSSEReader(resp.Body, "gemini")
	reader.AcceptBare = true
	return &stream{resp: resp, sse: reader, model: model}
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
			return nil, litellm.NewError("gemini", litellm.ErrorTypeProvider, "stream ended before finishReason", nil)
		}
		if err != nil {
			return nil, err
		}
		var chunk response
		if err := json.Unmarshal([]byte(frame.Data), &chunk); err != nil {
			return nil, litellm.NewError("gemini", litellm.ErrorTypeProvider, "parse stream chunk", err)
		}
		if err := wire.ErrorField("gemini", chunk.Error); err != nil {
			return nil, err
		}
		s.pending = s.events(s.pending, chunk)
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

func (s *stream) events(events []litellm.Event, chunk response) []litellm.Event {
	if chunk.UsageMetadata != nil {
		events = append(events, litellm.UsageEvent{Usage: convertUsage(chunk.UsageMetadata)})
	}
	if len(chunk.Candidates) == 0 {
		if chunk.PromptFeedback != nil && chunk.PromptFeedback.BlockReason != "" {
			return s.finish(events, chunk.PromptFeedback.BlockReason, wire.FinishReason(chunk.PromptFeedback.BlockReason))
		}
		return events
	}
	candidate := chunk.Candidates[0]
	for _, p := range candidate.Content.Parts {
		switch b := partBlock(p).(type) {
		case litellm.TextBlock:
			events = s.extendRun(events, "text", b.Text, b.Signature)
		case litellm.ReasoningBlock:
			events = s.extendRun(events, "reasoning", b.Text, b.Signature)
		case litellm.ToolUseBlock:
			events = s.endRun(events)
			if p.FunctionCall.ID == "" {
				events = append(events, litellm.WarningEvent{Warning: generatedIDWarning(b)})
			}
			s.run++
			var index int
			events, index = s.blocks.Open(events, s.run, litellm.ToolUseBlock{ID: b.ID, Name: b.Name, Signature: b.Signature})
			events = append(events, litellm.ToolUseDelta{Index: index, Arguments: string(b.Arguments)})
			events = s.blocks.Close(events, s.run, nil)
			s.toolCalls = true
		}
	}
	if candidate.FinishMessage != "" {
		events = append(events, litellm.WarningEvent{Warning: finishMessageWarning(candidate.FinishMessage)})
	}
	if candidate.FinishReason != "" {
		return s.finish(events, candidate.FinishReason, finishReason(candidate.FinishReason, s.toolCalls))
	}
	return events
}

// extendRun adds a text or thought part to the open run of its kind. A new
// run starts on a change of kind, or when both carry a signature: signatures
// cannot be merged.
func (s *stream) extendRun(events []litellm.Event, kind, text, signature string) []litellm.Event {
	if s.open != kind || (signature != "" && s.signature != "") {
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
	if s.signature != "" {
		final = litellm.TextBlock{Signature: s.signature}
		if s.open == "reasoning" {
			final = litellm.ReasoningBlock{Signature: s.signature}
		}
	}
	s.open, s.signature = "", ""
	return s.blocks.Close(events, s.run, final)
}

func (s *stream) finish(events []litellm.Event, raw string, reason litellm.FinishReason) []litellm.Event {
	events = s.endRun(events)
	events = s.blocks.CloseAll(events, nil)
	s.done = true
	return append(events, litellm.DoneEvent{FinishReason: reason, FinishReasonRaw: raw, Provider: "gemini", Model: s.model})
}
