package compat

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/voocel/litellm"
)

type stream struct {
	resp          *http.Response
	scanner       *bufio.Scanner
	req           *litellm.Request
	spec          Spec
	pending       []litellm.Event
	done          bool
	model         string
	usage         litellm.Usage
	finish        litellm.FinishReason
	lastReasoning string
	toolIDs       map[int]string
}

func newStream(resp *http.Response, req *litellm.Request, spec Spec) *stream {
	scanner := bufio.NewScanner(resp.Body)
	buf := make([]byte, 0, 64*1024)
	scanner.Buffer(buf, 1024*1024)
	return &stream{
		resp:    resp,
		scanner: scanner,
		req:     req,
		spec:    spec,
		model:   req.Model,
		toolIDs: make(map[int]string),
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
		data, ok := strings.CutPrefix(line, s.spec.dataPrefix())
		if !ok {
			if trimmed, found := strings.CutPrefix(line, "data:"); found {
				data = strings.TrimSpace(trimmed)
				ok = true
			}
		}
		if !ok {
			continue
		}
		if data == s.spec.doneSentinel() {
			s.done = true
			return litellm.DoneEvent{FinishReason: s.finish, Provider: s.spec.providerName(), Model: s.model}, nil
		}
		var chunk streamChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			return nil, litellm.NewProviderErrorWithCause(s.spec.providerName(), litellm.ErrorTypeProvider, fmt.Sprintf("%s: parse stream chunk", s.spec.providerName()), err)
		}
		events, err := s.events(chunk)
		if err != nil {
			return nil, err
		}
		if len(events) == 0 {
			continue
		}
		s.pending = append(s.pending, events[1:]...)
		return events[0], nil
	}
	if err := s.scanner.Err(); err != nil {
		return nil, litellm.NewNetworkError(s.spec.providerName(), "stream read error", err)
	}
	s.done = true
	return nil, litellm.NewProviderError(s.spec.providerName(), litellm.ErrorTypeProvider, fmt.Sprintf("%s: stream ended before %s", s.spec.providerName(), s.spec.doneSentinel()))
}

func (s *stream) Close() error {
	return s.resp.Body.Close()
}

func (s *stream) events(chunk streamChunk) ([]litellm.Event, error) {
	events := make([]litellm.Event, 0, 4)
	if chunk.Model != "" {
		s.model = chunk.Model
	}
	if len(chunk.Usage) > 0 {
		var usage usage
		if err := json.Unmarshal(chunk.Usage, &usage); err != nil {
			return nil, litellm.NewProviderErrorWithCause(s.spec.providerName(), litellm.ErrorTypeProvider, fmt.Sprintf("%s: parse usage", s.spec.providerName()), err)
		}
		s.usage = convertUsage(usage, s.spec, s.spec.providerName(), s.model)
		events = append(events, litellm.UsageEvent{Usage: s.usage})
	}
	for _, choice := range chunk.Choices {
		if len(choice.Delta) > 0 {
			var delta map[string]any
			if err := json.Unmarshal(choice.Delta, &delta); err != nil {
				return nil, litellm.NewProviderErrorWithCause(s.spec.providerName(), litellm.ErrorTypeProvider, fmt.Sprintf("%s: parse delta", s.spec.providerName()), err)
			}
			if text := s.findContent(delta); text != "" {
				events = append(events, litellm.ContentDelta{Text: text, OutputIndex: litellm.IntPtr(choice.Index)})
			}
			if refusal, _ := delta["refusal"].(string); refusal != "" {
				events = append(events, litellm.RefusalDelta{Text: refusal, OutputIndex: litellm.IntPtr(choice.Index)})
			}
			if s.reasoningAllowed() {
				if reasoning := findReasoning(delta, s.reasoningFields()); reasoning != "" {
					if s.spec.Stream.ReasoningCumulative {
						next, err := s.reasoningDelta(reasoning)
						if err != nil {
							return nil, err
						}
						reasoning = next
					}
					if reasoning != "" {
						events = append(events, litellm.ReasoningDelta{Text: reasoning, Index: litellm.IntPtr(choice.Index)})
					}
				}
			}
			if rawCalls, ok := delta["tool_calls"].([]any); ok {
				for _, raw := range rawCalls {
					toolEvents, err := s.toolEvents(raw, choice.Index)
					if err != nil {
						return nil, err
					}
					events = append(events, toolEvents...)
				}
			}
		}
		if choice.FinishReason != "" {
			s.finish = litellm.NormalizeFinishReason(choice.FinishReason)
		}
	}
	return events, nil
}

func (s *stream) findContent(delta map[string]any) string {
	fields := s.spec.Stream.ContentFields
	if len(fields) == 0 {
		fields = []string{"content"}
	}
	for _, field := range fields {
		if text, _ := delta[field].(string); text != "" {
			return text
		}
	}
	return ""
}

func (s *stream) reasoningAllowed() bool {
	if s.req != nil && s.req.Thinking != nil && s.req.Thinking.Mode == litellm.ThinkingDisabled {
		return false
	}
	cond := s.spec.Stream.ReasoningCondition
	if cond == "" || cond == "always" {
		return true
	}
	if after, ok := strings.CutPrefix(cond, "model_contains:"); ok {
		return strings.Contains(strings.ToLower(s.model), strings.ToLower(after))
	}
	return true
}

func (s *stream) reasoningFields() []string {
	if len(s.spec.Stream.ReasoningFields) > 0 {
		return s.spec.Stream.ReasoningFields
	}
	if len(s.spec.Response.ReasoningFields) > 0 {
		return s.spec.Response.ReasoningFields
	}
	return []string{"reasoning_summary", "reasoning_details", "reasoning_content", "reasoning", "reasoning_text"}
}

func (s *stream) reasoningDelta(current string) (string, error) {
	if s.lastReasoning == "" {
		s.lastReasoning = current
		return current, nil
	}
	if !strings.HasPrefix(current, s.lastReasoning) {
		return "", litellm.NewProviderError(s.spec.providerName(), litellm.ErrorTypeProvider, fmt.Sprintf("%s: cumulative reasoning stream changed unexpectedly", s.spec.providerName()))
	}
	next := strings.TrimPrefix(current, s.lastReasoning)
	s.lastReasoning = current
	return next, nil
}

func (s *stream) toolEvents(raw any, choiceIndex int) ([]litellm.Event, error) {
	m, ok := raw.(map[string]any)
	if !ok {
		return nil, litellm.NewProviderError(s.spec.providerName(), litellm.ErrorTypeProvider, fmt.Sprintf("%s: stream tool_call must be an object", s.spec.providerName()))
	}
	index := choiceIndex
	if v, ok := m["index"]; ok {
		number, ok := v.(float64)
		if !ok || number != float64(int(number)) {
			return nil, litellm.NewProviderError(s.spec.providerName(), litellm.ErrorTypeProvider, fmt.Sprintf("%s: stream tool_call index must be integer", s.spec.providerName()))
		}
		index = int(number)
	}
	id, err := optionalString(m, "id", s.spec.providerName(), "stream tool_call")
	if err != nil {
		return nil, err
	}
	if id != "" {
		s.toolIDs[index] = id
	} else {
		id = s.toolIDs[index]
	}
	var name, args string
	if rawFn, ok := m["function"]; ok {
		fn, ok := rawFn.(map[string]any)
		if !ok {
			return nil, litellm.NewProviderError(s.spec.providerName(), litellm.ErrorTypeProvider, fmt.Sprintf("%s: stream tool_call function must be an object", s.spec.providerName()))
		}
		name, err = optionalString(fn, "name", s.spec.providerName(), "stream tool_call function")
		if err != nil {
			return nil, err
		}
		args, err = optionalString(fn, "arguments", s.spec.providerName(), "stream tool_call function")
		if err != nil {
			return nil, err
		}
	} else if _, hasType := m["type"]; hasType {
		return nil, litellm.NewProviderError(s.spec.providerName(), litellm.ErrorTypeProvider, fmt.Sprintf("%s: stream tool_call missing function object", s.spec.providerName()))
	}
	events := make([]litellm.Event, 0, 2)
	if id != "" || name != "" {
		events = append(events, litellm.ToolUseStart{ID: id, Name: name, Index: litellm.IntPtr(index)})
	}
	if args != "" {
		events = append(events, litellm.ToolUseDelta{ID: id, Index: litellm.IntPtr(index), ArgumentsDelta: []byte(args)})
	}
	return events, nil
}

func optionalString(m map[string]any, key, provider, context string) (string, error) {
	value, ok := m[key]
	if !ok || value == nil {
		return "", nil
	}
	text, ok := value.(string)
	if !ok {
		return "", litellm.NewProviderError(provider, litellm.ErrorTypeProvider, fmt.Sprintf("%s: %s %s must be string", provider, context, key))
	}
	return text, nil
}
