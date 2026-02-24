package providers

import (
	"bufio"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
)

// ---------------------------------------------------------------------------
// compatStreamReader — unified SSE stream reader for OpenAI-compatible APIs
// ---------------------------------------------------------------------------

type compatStreamReader struct {
	resp             *http.Response
	scanner          *bufio.Scanner
	compat           *Compat
	includeReasoning bool
	done             bool
	model            string
	usage            *Usage
	pendingChunks    []*StreamChunk
}

func newCompatStreamReader(resp *http.Response, req *Request, compat *Compat) *compatStreamReader {
	scanner := bufio.NewScanner(resp.Body)
	buf := make([]byte, 0, 64*1024)
	scanner.Buffer(buf, 1024*1024)

	return &compatStreamReader{
		resp:             resp,
		scanner:          scanner,
		compat:           compat,
		includeReasoning: !isThinkingDisabled(req),
		model:            req.Model,
	}
}

func (r *compatStreamReader) Next() (*StreamChunk, error) {
	if r.done {
		return &StreamChunk{Done: true, Provider: r.compat.ProviderName, Model: r.model, Usage: r.usage}, nil
	}

	// Drain pending chunks from previous SSE event (multi-tool-call support)
	if len(r.pendingChunks) > 0 {
		chunk := r.pendingChunks[0]
		r.pendingChunks = r.pendingChunks[1:]
		return chunk, nil
	}

	prefix := r.compat.dataPrefix()

	for r.scanner.Scan() {
		line := r.scanner.Text()

		// Fast skip: empty lines and SSE comments
		if line == "" || line[0] == ':' {
			continue
		}

		data, found := strings.CutPrefix(line, prefix)
		if !found {
			// Handle "data:" without space (some providers)
			if d, ok := strings.CutPrefix(line, "data:"); ok {
				data = strings.TrimSpace(d)
				found = true
			}
		}
		if !found {
			continue
		}

		if data == "[DONE]" {
			r.done = true
			return &StreamChunk{Done: true, Provider: r.compat.ProviderName, Model: r.model, Usage: r.usage}, nil
		}

		var chunk compatStreamChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			return nil, fmt.Errorf("%s: failed to parse stream chunk: %w", r.compat.ProviderName, err)
		}

		// Update model from response
		if chunk.Model != "" {
			r.model = chunk.Model
		}

		// Handle usage (may come with empty choices or in final chunk)
		if len(chunk.Usage) > 0 {
			u := parseUsage(chunk.Usage, r.compat)
			r.usage = &u
		}

		// Handle usage-only chunk (choices empty, usage present) — store and continue
		if len(chunk.Choices) == 0 {
			continue
		}

		choice := chunk.Choices[0]
		pending := make([]*StreamChunk, 0, 4)

		// Parse delta
		if len(choice.Delta) > 0 {
			var delta map[string]any
			if err := json.Unmarshal(choice.Delta, &delta); err == nil {
				// Content
				if content, _ := delta["content"].(string); content != "" {
					pending = append(pending, &StreamChunk{
						Provider: r.compat.ProviderName,
						Model:    r.model,
						Type:     "content",
						Content:  content,
					})
				}

				// Reasoning — probe multiple field names
				if r.includeReasoning && r.compat.shouldExtractReasoning(r.model) {
					if reasoning, _ := r.compat.findReasoning(delta); reasoning != "" {
						pending = append(pending, &StreamChunk{
							Provider:  r.compat.ProviderName,
							Model:     r.model,
							Type:      "reasoning",
							Reasoning: &ReasoningChunk{Content: reasoning},
						})
					}
				}

				// Tool calls — may contain multiple deltas in one SSE event
				if rawCalls, ok := delta["tool_calls"]; ok {
					if calls, ok := rawCalls.([]any); ok {
						for _, call := range calls {
							if tc := r.parseToolCallDelta(call, choice.Index); tc != nil {
								pending = append(pending, &StreamChunk{
									Provider:      r.compat.ProviderName,
									Model:         r.model,
									Type:          "tool_call_delta",
									ToolCallDelta: tc,
								})
							}
						}
					}
				}
			}
		}

		// Finish reason
		if choice.FinishReason != "" {
			pending = append(pending, &StreamChunk{
				Provider:     r.compat.ProviderName,
				Model:        r.model,
				FinishReason: NormalizeFinishReason(choice.FinishReason),
			})
		}

		if len(pending) > 0 {
			r.pendingChunks = append(r.pendingChunks, pending[1:]...)
			return pending[0], nil
		}
	}

	if err := r.scanner.Err(); err != nil {
		return nil, fmt.Errorf("%s: stream read error: %w", r.compat.ProviderName, err)
	}

	r.done = true
	return &StreamChunk{Done: true, Provider: r.compat.ProviderName, Model: r.model, Usage: r.usage}, nil
}

func (r *compatStreamReader) Close() error {
	return r.resp.Body.Close()
}

// parseToolCallDelta extracts a ToolCallDelta from a raw tool_calls array element.
func (r *compatStreamReader) parseToolCallDelta(raw any, choiceIndex int) *ToolCallDelta {
	m, ok := raw.(map[string]any)
	if !ok {
		return nil
	}

	idx := choiceIndex
	if v, ok := m["index"].(float64); ok {
		idx = int(v)
	}

	tc := &ToolCallDelta{
		Index: idx,
		ID:    stringVal(m, "id"),
		Type:  stringVal(m, "type"),
	}

	if fn, ok := m["function"].(map[string]any); ok {
		tc.FunctionName = stringVal(fn, "name")
		tc.ArgumentsDelta = stringVal(fn, "arguments")
	}

	return tc
}

// ---------------------------------------------------------------------------
// Stream types
// ---------------------------------------------------------------------------

type compatStreamChunk struct {
	ID      string                `json:"id"`
	Model   string                `json:"model"`
	Choices []compatStreamChoice  `json:"choices"`
	Usage   json.RawMessage       `json:"usage,omitempty"`
}

type compatStreamChoice struct {
	Index        int             `json:"index"`
	Delta        json.RawMessage `json:"delta"`
	FinishReason string          `json:"finish_reason"`
}
