package anthropic

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"reflect"
	"strings"
	"testing"

	"github.com/voocel/litellm"
)

func TestStreamFixtureEvents(t *testing.T) {
	var body map[string]any
	p := newTestProvider(t, func(req *http.Request) (*http.Response, error) {
		if got := req.Header.Get("Accept"); got != "text/event-stream" {
			t.Errorf("Accept = %q", got)
		}
		if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		return fixtureResponse(t, "messages_stream.sse"), nil
	})
	stream, err := p.Stream(t.Context(), &litellm.Request{Model: "claude", MaxTokens: new(64), Messages: []litellm.Message{litellm.UserText("hi")}})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	defer stream.Close()
	if body["stream"] != true {
		t.Fatalf("stream = %v", body["stream"])
	}
	want := []litellm.Event{
		litellm.UsageEvent{Usage: litellm.Usage{InputTokens: new(7), OutputTokens: new(1), TotalTokens: new(8), CacheReadTokens: new(2)}},
		litellm.BlockStart{Index: 0, Block: litellm.ReasoningBlock{State: reasoningState("claude", "thinking", "", "")}},
		litellm.ReasoningDelta{Index: 0, Text: "think"},
		litellm.BlockEnd{Index: 0, Block: litellm.ReasoningBlock{State: reasoningState("claude", "thinking", "sig-thinking", "")}},
		litellm.BlockStart{Index: 1, Block: litellm.TextBlock{}},
		litellm.TextDelta{Index: 1, Text: "hello"},
		litellm.BlockEnd{Index: 1},
		litellm.BlockStart{Index: 2, Block: litellm.ToolUseBlock{ID: "toolu_1", Name: "lookup"}},
		litellm.ToolUseDelta{Index: 2, Arguments: `{"q":`},
		litellm.ToolUseDelta{Index: 2, Arguments: `"x"}`},
		litellm.BlockEnd{Index: 2},
		litellm.UsageEvent{Usage: litellm.Usage{InputTokens: new(7), OutputTokens: new(7), TotalTokens: new(14), CacheReadTokens: new(2)}},
		litellm.DoneEvent{FinishReason: litellm.FinishReasonToolCall, FinishReasonRaw: "tool_use", Provider: "anthropic", Model: "claude-sonnet"},
	}
	if got, err := drain(stream); err != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("events = %#v, err = %v\nwant %#v", got, err, want)
	}
}

func TestStreamEvents(t *testing.T) {
	const (
		serverStart = `{"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{}}}`
		serverDelta = `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{}"}}`
		serverStop  = `{"type":"content_block_stop","index":0}`
	)
	done := litellm.DoneEvent{Provider: "anthropic", Model: "m"}
	for _, test := range []struct {
		name  string
		lines []string
		want  []litellm.Event
	}{
		{
			name: "non-empty initial tool input becomes a delta",
			lines: []string{
				`{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"t1","name":"f","input":{"a":1}}}`,
				`{"type":"content_block_stop","index":0}`,
				`{"type":"message_stop"}`,
			},
			want: []litellm.Event{
				litellm.BlockStart{Index: 0, Block: litellm.ToolUseBlock{ID: "t1", Name: "f"}},
				litellm.ToolUseDelta{Index: 0, Arguments: `{"a":1}`},
				litellm.BlockEnd{Index: 0},
				done,
			},
		},
		{
			name: "unknown block warns and keeps indexes dense",
			lines: []string{
				serverStart, serverDelta, serverStop,
				`{"type":"content_block_start","index":1,"content_block":{"type":"redacted_thinking","data":"opaque"}}`,
				`{"type":"content_block_stop","index":1}`,
				`{"type":"message_stop"}`,
			},
			want: []litellm.Event{
				litellm.WarningEvent{Warning: unsupportedBlock("server_tool_use")},
				litellm.ProviderEvent{Name: "content_block_start", Raw: json.RawMessage(serverStart)},
				litellm.ProviderEvent{Name: "content_block_delta", Raw: json.RawMessage(serverDelta)},
				litellm.ProviderEvent{Name: "content_block_stop", Raw: json.RawMessage(serverStop)},
				litellm.BlockStart{Index: 0, Block: litellm.ReasoningBlock{State: reasoningState("m", "redacted_thinking", "", "opaque")}},
				litellm.BlockEnd{Index: 0},
				done,
			},
		},
		{
			name: "citations are delivered at block end",
			lines: []string{
				`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
				`{"type":"content_block_delta","index":0,"delta":{"type":"citations_delta","citation":` + citation + `}}`,
				`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"x"}}`,
				`{"type":"content_block_stop","index":0}`,
				`{"type":"message_stop"}`,
			},
			want: []litellm.Event{
				litellm.BlockStart{Index: 0, Block: litellm.TextBlock{}},
				litellm.TextDelta{Index: 0, Text: "x"},
				litellm.BlockEnd{Index: 0, Block: litellm.TextBlock{Annotations: []litellm.Annotation{
					{Type: "web_search_result_location", Text: "quoted", URL: "https://x.test", Extra: json.RawMessage(citation)},
				}}},
				done,
			},
		},
		{
			name: "message_stop closes open blocks",
			lines: []string{
				`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":"a"}}`,
				`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"b"}}`,
				`{"type":"message_stop"}`,
			},
			want: []litellm.Event{
				litellm.BlockStart{Index: 0, Block: litellm.TextBlock{Text: "a"}},
				litellm.TextDelta{Index: 0, Text: "b"},
				litellm.BlockEnd{Index: 0},
				done,
			},
		},
		{
			// Thinking is replayable only once signed, so a cut-off block has no
			// State; message_stop still delivers a pending signature.
			name: "thinking gets State with its signature",
			lines: []string{
				`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}`,
				`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"t"}}`,
				`{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig"}}`,
				`{"type":"message_stop"}`,
			},
			want: []litellm.Event{
				litellm.BlockStart{Index: 0, Block: litellm.ReasoningBlock{}},
				litellm.ReasoningDelta{Index: 0, Text: "t"},
				litellm.BlockEnd{Index: 0, Block: litellm.ReasoningBlock{State: reasoningState("m", "thinking", "sig", "")}},
				done,
			},
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			got, err := drain(newStream(sseResponse(test.lines...), "m"))
			if err != nil || !reflect.DeepEqual(got, test.want) {
				t.Fatalf("events = %#v, err = %v\nwant %#v", got, err, test.want)
			}
		})
	}
}

func TestStreamErrors(t *testing.T) {
	for _, test := range []struct {
		name     string
		lines    []string
		wantType litellm.ErrorType
		want     string
	}{
		{
			name:     "error event",
			lines:    []string{`{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`},
			wantType: litellm.ErrorTypeOverloaded,
			want:     "anthropic: overloaded_error: stream error: Overloaded",
		},
		{
			name:     "error event without detail",
			lines:    []string{`{"type":"error"}`},
			wantType: litellm.ErrorTypeProvider,
			want:     "anthropic: unknown stream error",
		},
		{
			name:     "EOF before message_stop",
			lines:    []string{`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":"partial"}}`},
			wantType: litellm.ErrorTypeProvider,
			want:     "anthropic: stream ended before message_stop",
		},
		{
			name:     "malformed event",
			lines:    []string{`{"type":`},
			wantType: litellm.ErrorTypeProvider,
			want:     "anthropic: parse stream event",
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			stream := newStream(sseResponse(test.lines...), "m")
			_, err := drain(stream)
			var e *litellm.Error
			if !errors.As(err, &e) || e.Type != test.wantType || err.Error() != test.want {
				t.Fatalf("err = %v, want %s %q", err, test.wantType, test.want)
			}
			if _, err := stream.Next(); err != io.EOF {
				t.Fatalf("Next after failure = %v, want io.EOF", err)
			}
		})
	}
}

// A streamed message aggregates to the same Response as its complete form.
func TestStreamMatchesCompleteResponse(t *testing.T) {
	var complete response
	if err := json.Unmarshal([]byte(`{"model":"claude","stop_reason":"tool_use",
		"usage":{"input_tokens":2,"output_tokens":3,"cache_creation_input_tokens":4},
		"content":[
			{"type":"text","text":"hello"},
			{"type":"text","text":""},
			{"type":"thinking","thinking":"think","signature":"sig"},
			{"type":"redacted_thinking","data":"opaque"},
			{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{}},
			{"type":"tool_use","id":"toolu_1","name":"lookup","input":{"q":"x"}}]}`), &complete); err != nil {
		t.Fatal(err)
	}
	want := convertResponse(&complete, "claude")
	got, err := litellm.Collect(newStream(sseResponse(
		`{"type":"message_start","message":{"model":"claude","usage":{"input_tokens":2,"cache_creation_input_tokens":4,"output_tokens":1}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":"hel"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"lo"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_stop","index":1}`,
		`{"type":"content_block_start","index":2,"content_block":{"type":"thinking","thinking":"","signature":""}}`,
		`{"type":"content_block_delta","index":2,"delta":{"type":"thinking_delta","thinking":"think"}}`,
		`{"type":"content_block_delta","index":2,"delta":{"type":"signature_delta","signature":"sig"}}`,
		`{"type":"content_block_stop","index":2}`,
		`{"type":"content_block_start","index":3,"content_block":{"type":"redacted_thinking","data":"opaque"}}`,
		`{"type":"content_block_stop","index":3}`,
		`{"type":"content_block_start","index":4,"content_block":{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{}}}`,
		`{"type":"content_block_stop","index":4}`,
		`{"type":"content_block_start","index":5,"content_block":{"type":"tool_use","id":"toolu_1","name":"lookup","input":{}}}`,
		`{"type":"content_block_delta","index":5,"delta":{"type":"input_json_delta","partial_json":"{\"q\":\"x\"}"}}`,
		`{"type":"content_block_stop","index":5}`,
		`{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":3}}`,
		`{"type":"message_stop"}`,
	), "claude"))
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("stream   %#v\ncomplete %#v", got, want)
	}
}

// Usage snapshots merge field by field; an explicit zero overwrites.
func TestStreamMergesUsage(t *testing.T) {
	s := &stream{}
	s.mergeUsage(&usage{InputTokens: new(5), OutputTokens: new(2), CacheReadInputTokens: new(3), CacheCreationInputTokens: new(4)})
	got := s.mergeUsage(&usage{OutputTokens: new(0)})
	want := litellm.UsageEvent{Usage: litellm.Usage{InputTokens: new(12), OutputTokens: new(0), TotalTokens: new(12), CacheReadTokens: new(3), CacheWriteTokens: new(4)}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("usage = %#v, want %#v", got, want)
	}
}

func drain(stream litellm.Stream) ([]litellm.Event, error) {
	var events []litellm.Event
	for {
		event, err := stream.Next()
		if err == io.EOF {
			return events, nil
		}
		if err != nil {
			return events, err
		}
		events = append(events, event)
	}
}

func sseResponse(data ...string) *http.Response {
	var b strings.Builder
	for _, line := range data {
		b.WriteString("data: " + line + "\n\n")
	}
	return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(b.String()))}
}
