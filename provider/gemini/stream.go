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
	signature string // latest thought signature of the open run
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
		if err := wire.ChunkError("gemini", chunk.Error); err != nil {
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
			events = s.continueRun(events, "text", litellm.TextBlock{})
			index, _ := s.blocks.Index(s.run)
			events = append(events, litellm.TextDelta{Index: index, Text: b.Text})
		case litellm.ReasoningBlock:
			events = s.continueRun(events, "reasoning", litellm.ReasoningBlock{})
			if b.Signature != "" {
				s.signature = b.Signature
			}
			if b.Text != "" {
				index, _ := s.blocks.Index(s.run)
				events = append(events, litellm.ReasoningDelta{Index: index, Text: b.Text})
			}
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
	if candidate.FinishReason != "" {
		return s.finish(events, candidate.FinishReason, finishReason(candidate.FinishReason, s.toolCalls))
	}
	return events
}

// continueRun keeps the open run when it has the same kind, else starts one.
func (s *stream) continueRun(events []litellm.Event, kind string, block litellm.Block) []litellm.Event {
	if s.open == kind {
		return events
	}
	events = s.endRun(events)
	s.run++
	s.open = kind
	events, _ = s.blocks.Open(events, s.run, block)
	return events
}

func (s *stream) endRun(events []litellm.Event) []litellm.Event {
	if s.open == "" {
		return events
	}
	var final litellm.Block
	if s.signature != "" {
		final = litellm.ReasoningBlock{Signature: s.signature}
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
