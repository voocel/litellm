package anthropic

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/internal/wire"
)

type streamEvent struct {
	Type         string   `json:"type"`
	Index        int      `json:"index"`
	ContentBlock *content `json:"content_block"`
	Delta        *struct {
		Type        string          `json:"type"`
		Text        string          `json:"text"`
		Thinking    string          `json:"thinking"`
		Signature   string          `json:"signature"`
		PartialJSON string          `json:"partial_json"`
		StopReason  string          `json:"stop_reason"`
		Citation    json.RawMessage `json:"citation"`
	} `json:"delta"`
	Usage   *usage `json:"usage"`
	Message *struct {
		Model string `json:"model"`
		Usage *usage `json:"usage"`
	} `json:"message"`
	Error *struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
}

type stream struct {
	resp       *http.Response
	sse        *wire.SSEReader
	name       string // the provider's
	pending    []litellm.Event
	done       bool
	requested  string // the requested model, for ProviderState
	model      string
	usage      usage
	finish     litellm.FinishReason
	finishRaw  string
	blocks     wire.BlockTracker[int] // native index to litellm index
	signatures map[int]string
	citations  map[int][]json.RawMessage
}

func newStream(resp *http.Response, name, model string) *stream {
	return &stream{resp: resp, sse: wire.NewSSEReader(resp.Body, name), name: name, requested: model, model: model, signatures: make(map[int]string), citations: make(map[int][]json.RawMessage)}
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
			// Before message_stop: the Client reports the truncation as a
			// retryable network error.
			return nil, io.EOF
		}
		if err != nil {
			return nil, err
		}
		var e streamEvent
		if err := json.Unmarshal([]byte(frame.Data), &e); err != nil {
			return nil, litellm.NewError(s.name, litellm.ErrorTypeProvider, "parse stream event", err)
		}
		if s.pending, err = s.events(s.pending, e, json.RawMessage(frame.Data)); err != nil {
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

func (s *stream) events(events []litellm.Event, e streamEvent, raw json.RawMessage) ([]litellm.Event, error) {
	switch e.Type {
	case "message_start":
		if e.Message != nil {
			if e.Message.Model != "" {
				s.model = e.Message.Model
			}
			if e.Message.Usage != nil {
				return append(events, s.mergeUsage(e.Message.Usage)), nil
			}
		}
	case "message_delta":
		if e.Delta != nil && e.Delta.StopReason != "" {
			s.finish, s.finishRaw = wire.FinishReason(e.Delta.StopReason), e.Delta.StopReason
		}
		if e.Usage != nil {
			return append(events, s.mergeUsage(e.Usage)), nil
		}
	case "message_stop":
		s.done = true
		events = s.blocks.CloseAll(events, s.final)
		return append(events, litellm.DoneEvent{FinishReason: s.finish, FinishReasonRaw: s.finishRaw, Provider: s.name, Model: s.model}), nil
	case "content_block_start":
		if e.ContentBlock == nil {
			return events, nil
		}
		block, ok := convertContent(*e.ContentBlock, s.name, s.requested)
		if !ok {
			events = append(events, litellm.WarningEvent{Warning: unsupportedBlock(s.name, e.ContentBlock.Type)})
			break
		}
		// Tool input normally arrives through input_json_delta after an empty
		// object; a non-empty initial input is streamed as the first delta.
		tool, isTool := block.(litellm.ToolUseBlock)
		if isTool {
			block = litellm.ToolUseBlock{ID: tool.ID, Name: tool.Name}
		}
		events, index := s.blocks.Open(events, e.Index, block)
		if isTool && tool.Arguments != "" && tool.Arguments != "{}" {
			events = append(events, litellm.ToolUseDelta{Index: index, Arguments: tool.Arguments})
		}
		return events, nil
	case "content_block_delta":
		index, ok := s.blocks.Index(e.Index)
		if !ok || e.Delta == nil {
			break
		}
		switch e.Delta.Type {
		case "text_delta":
			return append(events, litellm.TextDelta{Index: index, Text: e.Delta.Text}), nil
		case "thinking_delta":
			return append(events, litellm.ReasoningDelta{Index: index, Text: e.Delta.Thinking}), nil
		case "signature_delta":
			s.signatures[e.Index] = e.Delta.Signature
			return events, nil
		case "citations_delta":
			s.citations[e.Index] = append(s.citations[e.Index], e.Delta.Citation)
			return events, nil
		case "input_json_delta":
			return append(events, litellm.ToolUseDelta{Index: index, Arguments: e.Delta.PartialJSON}), nil
		}
	case "content_block_stop":
		if _, ok := s.blocks.Index(e.Index); ok {
			return s.blocks.Close(events, e.Index, s.final(e.Index)), nil
		}
	case "ping":
		return events, nil
	case "error":
		if e.Error == nil {
			return nil, litellm.NewError(s.name, litellm.ErrorTypeProvider, "unknown stream error", nil)
		}
		return nil, wire.StreamError(s.name, e.Error.Type, "stream error: "+e.Error.Message)
	}
	return append(events, litellm.ProviderEvent{Name: e.Type, Raw: raw}), nil
}

// final returns the late metadata of the block at a native index: signatures
// and citations arrive as deltas.
func (s *stream) final(index int) litellm.Block {
	if signature := s.signatures[index]; signature != "" {
		return litellm.ReasoningBlock{State: reasoningState(s.name, s.requested, "thinking", signature, "")}
	}
	if citations := s.citations[index]; len(citations) > 0 {
		return litellm.TextBlock{Annotations: annotations(citations)}
	}
	return nil
}

// mergeUsage folds a cumulative usage snapshot; an explicit zero overwrites.
func (s *stream) mergeUsage(u *usage) litellm.Event {
	if u.InputTokens != nil {
		s.usage.InputTokens = u.InputTokens
	}
	if u.OutputTokens != nil {
		s.usage.OutputTokens = u.OutputTokens
	}
	if u.CacheReadInputTokens != nil {
		s.usage.CacheReadInputTokens = u.CacheReadInputTokens
	}
	if u.CacheCreationInputTokens != nil {
		s.usage.CacheCreationInputTokens = u.CacheCreationInputTokens
	}
	return litellm.UsageEvent{Usage: convertUsage(s.usage)}
}
