package openaicompat_test

import (
	"context"
	"encoding/json"
	"net/http"
	"reflect"
	"strings"
	"testing"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/internal/testgolden"
	"github.com/voocel/litellm/provider/internal/openaicompat"
	"github.com/voocel/litellm/provider/internal/openaicompat/compattest"
)

// streamComplete is the complete response equivalent to testdata/compat/stream.sse.
const streamComplete = `{"model":"m","choices":[{"message":{"reasoning_content":"think","content":"hi","tool_calls":[
	{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"q\":\"x\"}"}}]},"finish_reason":"tool_calls"}],
	"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}`

func TestStreamFixtureEvents(t *testing.T) {
	events, err := compattest.Events(t, plain, testgolden.ReadFixtureString(t, "../../../testdata/compat/stream.sse"))
	if err != nil {
		t.Fatal(err)
	}
	want := []litellm.Event{
		litellm.BlockStart{Index: 0, Block: litellm.ReasoningBlock{}},
		litellm.ReasoningDelta{Index: 0, Text: "think"},
		litellm.BlockStart{Index: 1, Block: litellm.TextBlock{}},
		litellm.TextDelta{Index: 1, Text: "hi"},
		litellm.BlockStart{Index: 2, Block: litellm.ToolUseBlock{ID: "call_1", Name: "lookup"}},
		litellm.ToolUseDelta{Index: 2, Arguments: `{"q":`},
		litellm.ToolUseDelta{Index: 2, Arguments: `"x"}`},
		litellm.BlockEnd{Index: 0},
		litellm.BlockEnd{Index: 1},
		litellm.BlockEnd{Index: 2, Block: litellm.ToolUseBlock{ID: "call_1", Name: "lookup"}},
		litellm.UsageEvent{Usage: litellm.Usage{InputTokens: 1, OutputTokens: 2}},
		litellm.DoneEvent{FinishReason: litellm.FinishReasonToolCall, FinishReasonRaw: "tool_calls", Provider: "test", Model: "m"},
	}
	if !reflect.DeepEqual(events, want) {
		t.Fatalf("events = %#v\nwant %#v", events, want)
	}
}

func TestStreamMatchesCompleteResponse(t *testing.T) {
	got, err := compattest.Collect(t, plain, testgolden.ReadFixtureString(t, "../../../testdata/compat/stream.sse"))
	if err != nil {
		t.Fatal(err)
	}
	want, err := compattest.Chat(t, plain, streamComplete)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("stream   %#v\ncomplete %#v", got, want)
	}
}

func TestStreamEvents(t *testing.T) {
	details := compattest.Spec(openaicompat.Spec{Name: "test", ReasoningFields: []string{"reasoning_details"}})
	done := litellm.DoneEvent{Provider: "test", Model: "m"}
	tests := []struct {
		name   string
		newFn  compattest.NewFunc
		chunks []string
		want   []litellm.Event
	}{
		{
			name:  "tool index falls back to position and late id and name are kept",
			newFn: plain,
			chunks: []string{
				`{"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"{"}},{"id":"b","function":{"name":"g"}}]}}]}`,
				`{"choices":[{"delta":{"tool_calls":[{"id":"a","function":{"name":"f","arguments":"}"}}]}}]}`,
			},
			want: []litellm.Event{
				litellm.BlockStart{Index: 0, Block: litellm.ToolUseBlock{}},
				litellm.ToolUseDelta{Index: 0, Arguments: "{"},
				litellm.BlockStart{Index: 1, Block: litellm.ToolUseBlock{ID: "b", Name: "g"}},
				litellm.ToolUseDelta{Index: 0, Arguments: "}"},
				litellm.BlockEnd{Index: 0, Block: litellm.ToolUseBlock{ID: "a", Name: "f"}},
				litellm.BlockEnd{Index: 1, Block: litellm.ToolUseBlock{ID: "b", Name: "g"}},
				done,
			},
		},
		{
			name:  "refusal ends with a safety finish",
			newFn: plain,
			chunks: []string{
				`{"choices":[{"delta":{"refusal":"no"},"finish_reason":"stop"}]}`,
			},
			want: []litellm.Event{
				litellm.BlockStart{Index: 0, Block: litellm.TextBlock{}},
				litellm.TextDelta{Index: 0, Text: "no"},
				litellm.BlockEnd{Index: 0},
				litellm.DoneEvent{FinishReason: litellm.FinishReasonSafety, FinishReasonRaw: "stop", Provider: "test", Model: "m"},
			},
		},
		{
			// Adjacent fragments sharing a type and index form one entry, as in
			// the complete response; encrypted fragments are always whole.
			name:  "reasoning detail fragments merge at block end",
			newFn: details,
			chunks: []string{
				`{"choices":[{"delta":{"reasoning_details":[{"type":"reasoning.text","index":0,"text":"Let"}]}}]}`,
				`{"choices":[{"delta":{"reasoning_details":[{"type":"reasoning.text","index":0,"text":" me"}]}}]}`,
				`{"choices":[{"delta":{"reasoning_details":[{"type":"reasoning.text","index":0,"signature":"SIG"},{"type":"reasoning.encrypted","index":1,"data":"x"}]}}]}`,
				`{"choices":[{"delta":{"reasoning_details":[{"type":"reasoning.encrypted","index":1,"data":"y"}]}}]}`,
			},
			want: []litellm.Event{
				litellm.BlockStart{Index: 0, Block: litellm.ReasoningBlock{}},
				litellm.ReasoningDelta{Index: 0, Text: "Let"},
				litellm.ReasoningDelta{Index: 0, Text: " me"},
				litellm.BlockEnd{Index: 0, Block: litellm.ReasoningBlock{State: &litellm.ProviderState{Provider: "test", Model: "m", Data: json.RawMessage(`[{"index":0,"signature":"SIG","text":"Let me","type":"reasoning.text"},{"data":"x","index":1,"type":"reasoning.encrypted"},{"data":"y","index":1,"type":"reasoning.encrypted"}]`)}}},
				done,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			events, err := compattest.Events(t, tt.newFn, compattest.SSE(tt.chunks...))
			if err != nil || !reflect.DeepEqual(events, tt.want) {
				t.Fatalf("events = %#v, err = %v\nwant %#v", events, err, tt.want)
			}
		})
	}
}

func TestStreamErrors(t *testing.T) {
	tests := []struct {
		name      string
		newFn     compattest.NewFunc
		sse       string
		errorType litellm.ErrorType
		temporary bool
		want      string
	}{
		{name: "error chunk", newFn: plain, sse: compattest.SSE(`{"error":{"code":"server_error","message":"boom"}}`), errorType: litellm.ErrorTypeProvider, temporary: true, want: "test: server_error: boom"},
		{name: "error chunk with HTTP status", newFn: plain, sse: compattest.SSE(`{"error":{"code":429,"message":"slow down"}}`), errorType: litellm.ErrorTypeRateLimit, temporary: true},
		{name: "malformed chunk", newFn: plain, sse: compattest.SSE(`{`), errorType: litellm.ErrorTypeProvider},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := compattest.Events(t, tt.newFn, tt.sse)
			if litellm.ErrorTypeOf(err) != tt.errorType || litellm.IsTemporaryError(err) != tt.temporary || (tt.want != "" && err.Error() != tt.want) {
				t.Fatalf("err = %v", err)
			}
		})
	}
}

// TestStreamTruncationIsRetryable checks that a stream cut off before a
// finish reason ends in io.EOF, which the Client reports as a temporary
// network error.
func TestStreamTruncationIsRetryable(t *testing.T) {
	sse := "data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n"
	if _, err := compattest.Stream(t, plain, sse).Next(); err != nil {
		t.Fatal(err)
	}
	p := compattest.Provider(t, plain, compattest.Doer(func(*http.Request) (*http.Response, error) { return compattest.Response(sse), nil }))
	client, err := litellm.New(p)
	if err != nil {
		t.Fatal(err)
	}
	stream, err := client.Stream(context.Background(), *compattest.Request())
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	resp, err := litellm.Collect(stream)
	if litellm.ErrorTypeOf(err) != litellm.ErrorTypeNetwork || !litellm.IsTemporaryError(err) || resp.Text() != "a" {
		t.Fatalf("resp = %#v, err = %v", resp, err)
	}
}

func TestStreamEOFAfterFinishReasonCompletes(t *testing.T) {
	resp, err := compattest.Collect(t, plain, "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}\n\n")
	if err != nil || resp.Text() != "hi" || resp.FinishReason != litellm.FinishReasonStop {
		t.Fatalf("resp = %#v, err = %v", resp, err)
	}
}

func TestStreamAfterFailureReturnsEOF(t *testing.T) {
	s := compattest.Stream(t, plain, compattest.SSE(`{`))
	if _, err := s.Next(); err == nil {
		t.Fatal("malformed chunk accepted")
	}
	if _, err := s.Next(); err == nil || !strings.Contains(err.Error(), "EOF") {
		t.Fatalf("Next after failure = %v", err)
	}
}
