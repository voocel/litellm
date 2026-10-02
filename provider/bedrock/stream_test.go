package bedrock

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"reflect"
	"strings"
	"testing"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/internal/testgolden"
)

func TestStreamFixtureEvents(t *testing.T) {
	got, err := drain(newStream(fixtureResponse(t, "eventstream.bin"), "bedrock", "claude"))
	if err != nil {
		t.Fatal(err)
	}
	want := []litellm.Event{
		litellm.ProviderEvent{Name: "bedrock.messageStart", Raw: json.RawMessage(`{"p":"abcd","role":"assistant"}`)},
		litellm.BlockStart{Index: 0, Block: litellm.TextBlock{}},
		litellm.TextDelta{Index: 0, Text: "hel"},
		litellm.BlockStart{Index: 1, Block: litellm.ToolUseBlock{ID: "toolu_1", Name: "lookup"}},
		litellm.ToolUseDelta{Index: 1, Arguments: `{"q":`},
		litellm.ToolUseDelta{Index: 1, Arguments: `"x"}`},
		litellm.BlockEnd{Index: 1},
		litellm.UsageEvent{Usage: litellm.Usage{InputTokens: 10, OutputTokens: 7, CacheReadTokens: 2, CacheWriteTokens: 3}},
		litellm.BlockEnd{Index: 0},
		litellm.DoneEvent{FinishReason: litellm.FinishReasonToolCall, FinishReasonRaw: "tool_use", Provider: "bedrock", Model: "claude"},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("events = %#v\nwant %#v", got, want)
	}
}

func TestStreamMatchesCompleteResponse(t *testing.T) {
	var complete response
	if err := json.Unmarshal([]byte(`{
		"output":{"message":{"role":"assistant","content":[{"text":"hel"},{"toolUse":{"toolUseId":"toolu_1","name":"lookup","input":{"q":"x"}}}]}},
		"stopReason":"tool_use",
		"usage":{"inputTokens":5,"outputTokens":7,"totalTokens":12,"cacheReadInputTokens":2,"cacheWriteInputTokens":3}
	}`), &complete); err != nil {
		t.Fatal(err)
	}
	got, err := litellm.Collect(newStream(fixtureResponse(t, "eventstream.bin"), "bedrock", "claude"))
	if err != nil {
		t.Fatal(err)
	}
	if want := convertResponse(&complete, "bedrock", "claude"); !reflect.DeepEqual(got, want) {
		t.Fatalf("stream   %#v\ncomplete %#v", got, want)
	}
}

func TestStreamEvents(t *testing.T) {
	const (
		futureStart = `{"contentBlockIndex":0,"start":{"image":{}}}`
		futureEvent = `{"x":1}`
	)
	metadata := event("metadata", `{}`)
	done := litellm.DoneEvent{Provider: "bedrock", Model: "m"}
	for _, test := range []struct {
		name   string
		frames []eventFrame
		want   []litellm.Event
	}{
		{
			name: "reasoning signature is delivered at block end",
			frames: []eventFrame{
				event("contentBlockDelta", `{"contentBlockIndex":0,"delta":{"reasoningContent":{"text":"a"}}}`),
				event("contentBlockDelta", `{"contentBlockIndex":0,"delta":{"reasoningContent":{"text":"b"}}}`),
				event("contentBlockDelta", `{"contentBlockIndex":0,"delta":{"reasoningContent":{"signature":"sig"}}}`),
				event("contentBlockStop", `{"contentBlockIndex":0}`),
				event("messageStop", `{"stopReason":"end_turn"}`),
				metadata,
			},
			want: []litellm.Event{
				litellm.BlockStart{Index: 0, Block: litellm.ReasoningBlock{}},
				litellm.ReasoningDelta{Index: 0, Text: "a"},
				litellm.ReasoningDelta{Index: 0, Text: "b"},
				litellm.BlockEnd{Index: 0, Block: litellm.ReasoningBlock{State: testState(`{"signature":"sig"}`)}},
				litellm.DoneEvent{FinishReason: litellm.FinishReasonStop, FinishReasonRaw: "end_turn", Provider: "bedrock", Model: "m"},
			},
		},
		{
			name: "redacted reasoning is delivered at block end",
			frames: []eventFrame{
				event("contentBlockDelta", `{"contentBlockIndex":0,"delta":{"reasoningContent":{"redactedContent":"b3BhcXVl"}}}`),
				event("contentBlockStop", `{"contentBlockIndex":0}`),
				metadata,
			},
			want: []litellm.Event{
				litellm.BlockStart{Index: 0, Block: litellm.ReasoningBlock{}},
				litellm.BlockEnd{Index: 0, Block: litellm.ReasoningBlock{State: testState(`{"redactedContent":"b3BhcXVl"}`)}},
				done,
			},
		},
		{
			name: "interleaved blocks keep first-appearance indexes",
			frames: []eventFrame{
				event("contentBlockDelta", `{"contentBlockIndex":0,"delta":{"reasoningContent":{"text":"r"}}}`),
				event("contentBlockDelta", `{"contentBlockIndex":1,"delta":{"text":"t"}}`),
				event("contentBlockDelta", `{"contentBlockIndex":0,"delta":{"reasoningContent":{"text":"r2"}}}`),
				event("contentBlockStop", `{"contentBlockIndex":1}`),
				event("contentBlockStop", `{"contentBlockIndex":0}`),
				metadata,
			},
			want: []litellm.Event{
				litellm.BlockStart{Index: 0, Block: litellm.ReasoningBlock{}},
				litellm.ReasoningDelta{Index: 0, Text: "r"},
				litellm.BlockStart{Index: 1, Block: litellm.TextBlock{}},
				litellm.TextDelta{Index: 1, Text: "t"},
				litellm.ReasoningDelta{Index: 0, Text: "r2"},
				litellm.BlockEnd{Index: 1},
				litellm.BlockEnd{Index: 0, Block: litellm.ReasoningBlock{State: testState(`{}`)}},
				done,
			},
		},
		{
			name: "unmodeled events keep indexes dense",
			frames: []eventFrame{
				event("contentBlockStart", futureStart),
				event("futureEvent", futureEvent),
				event("contentBlockDelta", `{"contentBlockIndex":1,"delta":{"text":"t"}}`),
				metadata,
			},
			want: []litellm.Event{
				litellm.ProviderEvent{Name: "bedrock.contentBlockStart", Raw: json.RawMessage(futureStart)},
				litellm.ProviderEvent{Name: "bedrock.futureEvent", Raw: json.RawMessage(futureEvent)},
				litellm.BlockStart{Index: 0, Block: litellm.TextBlock{}},
				litellm.TextDelta{Index: 0, Text: "t"},
				litellm.BlockEnd{Index: 0},
				done,
			},
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			got, err := drain(newStream(&http.Response{Body: eventStream(test.frames...)}, "bedrock", "m"))
			if err != nil || !reflect.DeepEqual(got, test.want) {
				t.Fatalf("events = %#v, err = %v\nwant %#v", got, err, test.want)
			}
		})
	}
}

func TestStreamErrors(t *testing.T) {
	is := func(want litellm.ErrorType, temporary bool) func(error) bool {
		return func(err error) bool {
			return litellm.ErrorTypeOf(err) == want && litellm.IsTemporaryError(err) == temporary
		}
	}
	for _, test := range []struct {
		name string
		body io.ReadCloser
		is   func(error) bool
		msg  string
	}{
		{
			name: "throttling exception",
			body: fixtureResponse(t, "eventstream_exception.bin").Body,
			is: func(err error) bool {
				return litellm.ErrorTypeOf(err) == litellm.ErrorTypeRateLimit && litellm.IsTemporaryError(err)
			},
			msg: "bedrock: throttlingException: stream error: Too many requests, please wait before trying again.",
		},
		{name: "validation exception", body: eventStream(exception("validationException", `{"message":"bad input"}`)), is: is(litellm.ErrorTypeValidation, false), msg: "bedrock: validationException: stream error: bad input"},
		{name: "unavailable exception", body: eventStream(exception("serviceUnavailableException", `{"message":"busy"}`)), is: is(litellm.ErrorTypeOverloaded, true)},
		{name: "model exception", body: eventStream(exception("modelStreamErrorException", `{}`)), is: is(litellm.ErrorTypeProvider, true), msg: "bedrock: modelStreamErrorException: stream error: modelStreamErrorException"},
		{
			name: "error frame",
			body: eventStream(eventFrame{headers: [][2]string{{":message-type", "error"}, {":error-code", "ThrottlingException"}, {":error-message", "slow down"}}}),
			is:   is(litellm.ErrorTypeRateLimit, true),
			msg:  "bedrock: ThrottlingException: stream error: slow down",
		},
		{
			// The Client reports the truncation as a temporary network error.
			name: "EOF before metadata",
			body: eventStream(event("contentBlockDelta", `{"contentBlockIndex":0,"delta":{"text":"partial"}}`)),
			is:   func(err error) bool { return err == io.EOF },
		},
		{name: "server fault", body: eventStream(exception("internalServerException", `{"message":"boom"}`)), is: is(litellm.ErrorTypeProvider, true)},
		{name: "corrupt frame", body: io.NopCloser(strings.NewReader(strings.Repeat("\x00", 16))), is: is(litellm.ErrorTypeProvider, false)},
		{name: "malformed payload", body: eventStream(event("contentBlockDelta", `{`)), is: is(litellm.ErrorTypeProvider, false)},
	} {
		t.Run(test.name, func(t *testing.T) {
			s := newStream(&http.Response{Body: test.body}, "bedrock", "m")
			_, err := drain(s)
			if err == nil || !test.is(err) || (test.msg != "" && err.Error() != test.msg) {
				t.Fatalf("err = %v", err)
			}
			if _, err := s.Next(); !errors.Is(err, io.EOF) {
				t.Fatalf("Next after failure = %v, want io.EOF", err)
			}
		})
	}
}

// drain reads events until Done or an error.
func drain(s *stream) ([]litellm.Event, error) {
	defer s.Close()
	var events []litellm.Event
	for {
		event, err := s.Next()
		if err != nil {
			return events, err
		}
		events = append(events, event)
		if _, ok := event.(litellm.DoneEvent); ok {
			return events, nil
		}
	}
}

func fixtureResponse(t *testing.T, name string) *http.Response {
	data := testgolden.ReadFixture(t, "../../testdata/bedrock/"+name)
	return &http.Response{Body: io.NopCloser(strings.NewReader(string(data)))}
}
