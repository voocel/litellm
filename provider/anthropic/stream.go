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
	pending    []litellm.Event
	done       bool
	model      string
	usage      usage
	finish     litellm.FinishReason
	finishRaw  string
	blocks     wire.BlockTracker[int] // native index to litellm index
	signatures map[int]string
	citations  map[int][]json.RawMessage
}

func newStream(resp *http.Response, model string) *stream {
	return &stream{resp: resp, sse: wire.NewSSEReader(resp.Body, "anthropic"), model: model, signatures: make(map[int]string), citations: make(map[int][]json.RawMessage)}
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
			return nil, litellm.NewError("anthropic", litellm.ErrorTypeProvider, "stream ended before message_stop", nil)
		}
		if err != nil {
			return nil, err
		}
		var e streamEvent
		if err := json.Unmarshal([]byte(frame.Data), &e); err != nil {
			return nil, litellm.NewError("anthropic", litellm.ErrorTypeProvider, "parse stream event", err)
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
		events = s.blocks.CloseAll(events, nil)
		return append(events, litellm.DoneEvent{FinishReason: s.finish, FinishReasonRaw: s.finishRaw, Provider: "anthropic", Model: s.model}), nil
	case "content_block_start":
		if e.ContentBlock == nil {
			return events, nil
		}
		block, ok := convertContent(*e.ContentBlock)
		if !ok {
			events = append(events, litellm.WarningEvent{Warning: unsupportedBlock(e.ContentBlock.Type)})
			break
		}
		// Tool input normally arrives through input_json_delta after an empty
		// object; a non-empty initial input is streamed as the first delta.
		tool, isTool := block.(litellm.ToolUseBlock)
		if isTool {
			block = litellm.ToolUseBlock{ID: tool.ID, Name: tool.Name}
		}
		events, index := s.blocks.Open(events, e.Index, block)
		if isTool && len(tool.Arguments) > 0 && string(tool.Arguments) != "{}" {
			events = append(events, litellm.ToolUseDelta{Index: index, Arguments: string(tool.Arguments)})
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
			// Signatures and citations arrive as deltas and are delivered here.
			var final litellm.Block
			if signature := s.signatures[e.Index]; signature != "" {
				final = litellm.ReasoningBlock{Signature: signature}
			}
			if citations := s.citations[e.Index]; len(citations) > 0 {
				final = litellm.TextBlock{Annotations: annotations(citations)}
			}
			return s.blocks.Close(events, e.Index, final), nil
		}
	case "ping":
		return events, nil
	case "error":
		if e.Error == nil {
			return nil, litellm.NewError("anthropic", litellm.ErrorTypeProvider, "unknown stream error", nil)
		}
		return nil, wire.StreamError("anthropic", e.Error.Type, "stream error: "+e.Error.Message)
	}
	return append(events, litellm.ProviderEvent{Name: e.Type, Raw: raw}), nil
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
