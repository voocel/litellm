package openai

import (
	"bufio"
	"encoding/json"
	"io"
	"net/http"
	"strings"

	"github.com/voocel/litellm"
)

type stream struct {
	resp             *http.Response
	scanner          *bufio.Scanner
	includeReasoning bool
	pending          []litellm.Event
	done             bool
	model            string
	// tools holds the calls by number; callAt numbers the call at each
	// vendor index.
	tools  map[int]*streamCall
	callAt map[int]int
	finish litellm.FinishReason
}

// streamCall is one streamed call. OpenAI sends its id and name once, on the
// opening chunk; some gateways repeat them on every chunk or send the name
// later. The call starts once its name is known; arguments that come before
// wait in args.
type streamCall struct {
	id, name string
	started  bool
	args     string
}

func newStream(resp *http.Response, req *litellm.Request) *stream {
	scanner := bufio.NewScanner(resp.Body)
	buf := make([]byte, 0, 64*1024)
	scanner.Buffer(buf, 1024*1024)
	return &stream{
		resp:             resp,
		scanner:          scanner,
		includeReasoning: thinkingEnabled(req),
		model:            req.Model,
		tools:            make(map[int]*streamCall),
		callAt:           make(map[int]int),
	}
}

func (s *stream) Next() (litellm.Event, error) {
	if len(s.pending) > 0 {
		event := s.pending[0]
		s.pending = s.pending[1:]
		return event, nil
	}
	if s.done {
		return nil, io.EOF
	}
	for s.scanner.Scan() {
		line := s.scanner.Text()
		if line == "" || line[0] == ':' {
			continue
		}
		data, ok := strings.CutPrefix(line, "data: ")
		if !ok {
			if trimmed, found := strings.CutPrefix(line, "data:"); found {
				data = strings.TrimSpace(trimmed)
				ok = true
			}
		}
		if !ok {
			continue
		}
		if data == "[DONE]" {
			s.done = true
			// A call whose name never came starts nameless, to fail downstream
			// rather than vanish.
			events := s.startUnnamed()
			events = append(events, litellm.DoneEvent{FinishReason: s.finish, Provider: "openai", Model: s.model})
			s.pending = append(s.pending, events[1:]...)
			return events[0], nil
		}
		var chunk streamChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			return nil, litellm.NewProviderErrorWithCause("openai", litellm.ErrorTypeProvider, "openai: parse stream chunk", err)
		}
		if chunk.Model != "" {
			s.model = chunk.Model
		}
		events := s.events(chunk)
		if len(events) == 0 {
			continue
		}
		s.pending = append(s.pending, events[1:]...)
		return events[0], nil
	}
	if err := s.scanner.Err(); err != nil {
		return nil, litellm.NewNetworkError("openai", "stream read error", err)
	}
	s.done = true
	return nil, litellm.NewProviderError("openai", litellm.ErrorTypeProvider, "openai: stream ended before [DONE]")
}

func (s *stream) Close() error {
	return s.resp.Body.Close()
}

func (s *stream) events(chunk streamChunk) []litellm.Event {
	events := make([]litellm.Event, 0, 4)
	if chunk.Usage != nil {
		events = append(events, litellm.UsageEvent{Usage: convertUsage(chunk.Usage, s.model)})
	}
	for _, choice := range chunk.Choices {
		if choice.Delta.Content != "" {
			events = append(events, litellm.ContentDelta{
				Text:        choice.Delta.Content,
				OutputIndex: litellm.IntPtr(choice.Index),
			})
		}
		if choice.Delta.Refusal != "" {
			events = append(events, litellm.RefusalDelta{
				Text:        choice.Delta.Refusal,
				OutputIndex: litellm.IntPtr(choice.Index),
			})
		}
		if s.includeReasoning {
			if text, summary := extractDeltaReasoning(choice.Delta); text != "" {
				events = append(events, litellm.ReasoningDelta{
					Text:    text,
					Summary: summary,
					Index:   litellm.IntPtr(choice.Index),
				})
			}
		}
		for _, call := range choice.Delta.ToolCalls {
			events = s.tool(events, call)
		}
		if choice.FinishReason != "" {
			s.finish = litellm.NormalizeFinishReason(choice.FinishReason)
		}
	}
	return events
}

func (s *stream) tool(events []litellm.Event, call toolCallDelta) []litellm.Event {
	var name, args string
	if call.Function != nil {
		name, args = call.Function.Name, call.Function.Arguments
	}
	n, ok := s.callAt[call.Index]
	// Another id at the same index starts another call: Ollama sends each
	// parallel call whole, all at index 0.
	if ok && call.ID != "" && s.tools[n].id != "" && call.ID != s.tools[n].id {
		ok = false
	}
	if !ok {
		if call.ID == "" && name == "" && args == "" {
			return events
		}
		n = len(s.tools)
		s.callAt[call.Index] = n
		s.tools[n] = &streamCall{}
	}
	tc := s.tools[n]
	if tc.id == "" {
		tc.id = call.ID
	}
	if tc.name == "" {
		tc.name = name
	}
	if !tc.started {
		tc.args += args
		if tc.name == "" {
			return events
		}
		return s.start(events, n, tc)
	}
	if args != "" {
		events = append(events, litellm.ToolUseDelta{ID: tc.id, Index: litellm.IntPtr(n), ArgumentsDelta: []byte(args)})
	}
	return events
}

func (s *stream) start(events []litellm.Event, n int, tc *streamCall) []litellm.Event {
	tc.started = true
	events = append(events, litellm.ToolUseStart{ID: tc.id, Name: tc.name, Index: litellm.IntPtr(n)})
	if tc.args != "" {
		events = append(events, litellm.ToolUseDelta{ID: tc.id, Index: litellm.IntPtr(n), ArgumentsDelta: []byte(tc.args)})
		tc.args = ""
	}
	return events
}

func (s *stream) startUnnamed() []litellm.Event {
	var events []litellm.Event
	for n := range len(s.tools) {
		if tc := s.tools[n]; !tc.started {
			events = s.start(events, n, tc)
		}
	}
	return events
}
