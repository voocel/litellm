package litellm

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"reflect"
	"strings"
	"testing"
)

// unknownEvent is an Event no collector knows.
type unknownEvent struct{}

func (unknownEvent) isEvent() {}

func TestCollectAssemblesInterleavedBlocks(t *testing.T) {
	resp, err := Collect(&testStream{events: []Event{
		BlockStart{Index: 0, Block: TextBlock{}},
		TextDelta{Index: 0, Text: "hel"},
		BlockStart{Index: 1, Block: ReasoningBlock{}},
		BlockStart{Index: 2, Block: ToolUseBlock{ID: "call_1", Name: "lookup"}},
		ToolUseDelta{Index: 2, Arguments: `{"q":`},
		ReasoningDelta{Index: 1, Text: "think"},
		TextDelta{Index: 0, Text: "lo"},
		ToolUseDelta{Index: 2, Arguments: `"x"}`},
		BlockEnd{Index: 2},
		WarningEvent{Warning: Warning{Code: "w"}},
		BlockEnd{Index: 1, Block: ReasoningBlock{State: testState(`"sig"`)}},
		BlockEnd{Index: 0},
		DoneEvent{FinishReason: FinishReasonToolCall, FinishReasonRaw: "tool_use", Provider: "test", Model: "m"},
	}})
	if err != nil {
		t.Fatal(err)
	}
	want := &Response{
		Blocks: []Block{
			TextBlock{Text: "hello"},
			ReasoningBlock{Text: "think", State: testState(`"sig"`)},
			ToolUseBlock{ID: "call_1", Name: "lookup", Arguments: json.RawMessage(`{"q":"x"}`)},
		},
		Provider:        "test",
		Model:           "m",
		FinishReason:    FinishReasonToolCall,
		FinishReasonRaw: "tool_use",
		Warnings:        []Warning{{Code: "w", Provider: "test"}},
	}
	if !reflect.DeepEqual(resp, want) {
		t.Fatalf("response = %#v\nwant %#v", resp, want)
	}
}

func TestBlockEndMergesMetadataOnly(t *testing.T) {
	resp, err := Collect(&testStream{events: []Event{
		BlockStart{Index: 0, Block: TextBlock{Text: "he"}},
		TextDelta{Index: 0, Text: "llo"},
		BlockEnd{Index: 0, Block: TextBlock{Text: "ignored", Annotations: []Annotation{{Type: "url", URL: "u"}}, Logprobs: json.RawMessage(`[]`), State: testState(`"t"`)}},
		BlockStart{Index: 1, Block: ReasoningBlock{Summary: true, State: testState(`"early"`)}},
		ReasoningDelta{Index: 1, Text: "r"},
		BlockEnd{Index: 1, Block: ReasoningBlock{Text: "ignored", State: testState(`"late"`)}},
		BlockStart{Index: 2, Block: ToolUseBlock{Name: "lookup"}},
		ToolUseDelta{Index: 2, Arguments: `{}`},
		BlockEnd{Index: 2, Block: ToolUseBlock{ID: "call", Arguments: json.RawMessage(`"ignored"`), State: testState(`"s"`)}},
		BlockStart{Index: 3, Block: ToolUseBlock{ID: "empty", Name: "noop"}},
		BlockEnd{Index: 3},
		DoneEvent{Provider: "test", Model: "m"},
	}})
	if err != nil {
		t.Fatal(err)
	}
	want := []Block{
		TextBlock{Text: "hello", Annotations: []Annotation{{Type: "url", URL: "u"}}, Logprobs: json.RawMessage(`[]`), State: testState(`"t"`)},
		ReasoningBlock{Text: "r", Summary: true, State: testState(`"late"`)},
		ToolUseBlock{ID: "call", Name: "lookup", Arguments: json.RawMessage(`{}`), State: testState(`"s"`)},
		// An argument-less call keeps valid JSON arguments.
		ToolUseBlock{ID: "empty", Name: "noop", Arguments: json.RawMessage(`{}`)},
	}
	if !reflect.DeepEqual(resp.Blocks, want) {
		t.Fatalf("blocks = %#v\nwant %#v", resp.Blocks, want)
	}
}

func TestCollectorRejectsLifecycleViolations(t *testing.T) {
	start := BlockStart{Index: 0, Block: TextBlock{}}
	done := DoneEvent{Provider: "test", Model: "m"}
	for _, tc := range []struct {
		name   string
		events []Event
		want   string
	}{
		{"start out of order", []Event{BlockStart{Index: 1, Block: TextBlock{}}}, "started out of order"},
		{"start twice", []Event{start, start}, "started out of order"},
		{"unsupported block", []Event{BlockStart{Index: 0, Block: ImageBlock{}}}, "does not support"},
		{"delta before start", []Event{TextDelta{Index: 0, Text: "x"}}, "was not started"},
		{"wrong delta kind", []Event{start, ReasoningDelta{Index: 0, Text: "x"}}, "is text, not reasoning"},
		{"delta after end", []Event{start, BlockEnd{Index: 0}, TextDelta{Index: 0, Text: "x"}}, "already ended"},
		{"end twice", []Event{start, BlockEnd{Index: 0}, BlockEnd{Index: 0}}, "already ended"},
		{"end kind mismatch", []Event{start, BlockEnd{Index: 0, Block: ReasoningBlock{}}}, "does not match"},
		{"done with open block", []Event{start, done}, "block 0 still open"},
		{"unknown event", []Event{unknownEvent{}}, "unknown stream event"},
		{"tool without id", []Event{BlockStart{Index: 0, Block: ToolUseBlock{Name: "t"}}, BlockEnd{Index: 0}, done}, "tool use missing id"},
		{"missing provider", []Event{DoneEvent{Model: "m"}}, "missing provider"},
		{"missing model", []Event{DoneEvent{Provider: "test"}}, "missing model"},
		{"nil event", []Event{nil}, "nil event"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for name, stream := range map[string]Stream{
				"external": &testStream{events: tc.events},
				"client":   newValidatedStream("", "", &testStream{events: tc.events}),
			} {
				if _, err := Collect(stream); err == nil || !strings.Contains(err.Error(), tc.want) {
					t.Errorf("%s: err = %v, want %q", name, err, tc.want)
				}
			}
		})
	}
}

func TestCollectorRejectsEventsAfterDone(t *testing.T) {
	collector := newCollector()
	if _, done, err := collector.Apply(DoneEvent{}); !done || err != nil {
		t.Fatalf("done=%v err=%v", done, err)
	}
	if _, _, err := collector.Apply(BlockStart{Index: 0, Block: TextBlock{}}); err == nil || !strings.Contains(err.Error(), "after Done") {
		t.Fatalf("err = %v", err)
	}
	if len(collector.Response().Blocks) != 0 {
		t.Fatal("late event changed completed response")
	}
}

func TestHandleDeliversCompletedBlocks(t *testing.T) {
	events := []Event{
		BlockStart{Index: 0, Block: ReasoningBlock{}},
		ReasoningDelta{Index: 0, Text: "th"},
		ReasoningDelta{Index: 0, Text: "ink"},
		BlockEnd{Index: 0, Block: ReasoningBlock{State: testState(`"sig"`)}},
		DoneEvent{Provider: "test", Model: "m"},
	}
	for name, stream := range map[string]Stream{
		"external": &testStream{events: events},
		"client":   newValidatedStream("test", "m", &testStream{events: events}),
	} {
		var ends []BlockEnd
		var seen int
		_, err := Handle(stream, func(event Event) error {
			seen++
			if end, ok := event.(BlockEnd); ok {
				ends = append(ends, end)
			}
			return nil
		})
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		want := BlockEnd{Index: 0, Block: ReasoningBlock{Text: "think", State: testState(`"sig"`)}}
		if seen != len(events) || len(ends) != 1 || !reflect.DeepEqual(ends[0], want) {
			t.Fatalf("%s: seen=%d ends=%#v", name, seen, ends)
		}
	}
}

func TestHandleAfterNext(t *testing.T) {
	events := append(textEvents(0, "hello"), DoneEvent{Provider: "test", Model: "m"})
	c, err := New(&testProvider{name: "test", streamFunc: func(context.Context, *Request) (Stream, error) {
		return &testStream{events: events}, nil
	}})
	if err != nil {
		t.Fatal(err)
	}
	for _, read := range []int{2, len(events)} {
		stream, err := c.Stream(context.Background(), Request{Model: "m", Messages: hi})
		if err != nil {
			t.Fatal(err)
		}
		for range read {
			if _, err := stream.Next(); err != nil {
				t.Fatal(err)
			}
		}
		// A Client stream aggregates from its first event.
		resp, err := Collect(stream)
		_ = stream.Close()
		if err != nil || resp.Text() != "hello" {
			t.Fatalf("client stream after %d events: text=%q err=%v", read, resp.Text(), err)
		}
	}
	// An external stream aggregates only from the point Handle starts.
	stream := &testStream{events: events}
	_, _ = stream.Next()
	if _, err := Collect(stream); err == nil || !strings.Contains(err.Error(), "was not started") {
		t.Fatalf("external stream after Next: %v", err)
	}
}

func TestHandleReturnsPartialResponse(t *testing.T) {
	boom := errors.New("interrupted")
	partial := textEvents(0, "partial")[:2]
	for _, tc := range []struct {
		name     string
		stream   Stream
		callback func(Event) error
		want     error
	}{
		{"truncated", &testStream{events: partial}, nil, io.ErrUnexpectedEOF},
		{"provider", &testStream{events: partial, err: boom}, nil, boom},
		{"callback", &testStream{events: partial}, func(e Event) error {
			if _, ok := e.(TextDelta); ok {
				return boom
			}
			return nil
		}, boom},
	} {
		t.Run(tc.name, func(t *testing.T) {
			resp, err := Handle(tc.stream, tc.callback)
			if !errors.Is(err, tc.want) || resp == nil || resp.Text() != "partial" || resp.FinishReason != "" {
				t.Fatalf("response=%#v error=%v", resp, err)
			}
		})
	}
}

func TestHandleStopsOnCallbackError(t *testing.T) {
	boom := errors.New("boom")
	var seen int
	_, err := Handle(&testStream{events: append(textEvents(0, "a"), DoneEvent{Provider: "test", Model: "m"})}, func(Event) error {
		seen++
		return boom
	})
	if !errors.Is(err, boom) || seen != 1 {
		t.Fatalf("err=%v seen=%d, want boom after one callback", err, seen)
	}
}

func TestValidatedStreamStopsAtTerminalEvent(t *testing.T) {
	for _, tc := range []struct {
		name   string
		events []Event
		want   error
	}{
		{"done", []Event{DoneEvent{Provider: "test"}, UsageEvent{}}, nil},
		{"truncated", nil, io.ErrUnexpectedEOF},
	} {
		t.Run(tc.name, func(t *testing.T) {
			stream := newValidatedStream("test", "m", &testStream{events: tc.events})
			if _, err := stream.Next(); !errors.Is(err, tc.want) {
				t.Fatalf("first error=%v want=%v", err, tc.want)
			}
			if event, err := stream.Next(); event != nil || err != io.EOF {
				t.Fatalf("after terminal: event=%#v err=%v", event, err)
			}
		})
	}
}

func TestMalformedToolArgumentsWarnOnce(t *testing.T) {
	events := func() []Event {
		return []Event{
			BlockStart{Index: 0, Block: ToolUseBlock{ID: "call", Name: "tool"}},
			ToolUseDelta{Index: 0, Arguments: `{"bad":`},
			BlockEnd{Index: 0},
			DoneEvent{Provider: "test", Model: "m"},
		}
	}
	for name, stream := range map[string]Stream{
		"external": &testStream{events: events()},
		"client":   newValidatedStream("test", "m", &testStream{events: events()}),
	} {
		resp, err := Collect(stream)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if len(resp.Warnings) != 1 || resp.Warnings[0].Code != "litellm.tool_arguments_invalid" {
			t.Fatalf("%s: warnings = %+v", name, resp.Warnings)
		}
		if got := string(resp.ToolCalls()[0].Arguments); got != `{"bad":` {
			t.Fatalf("%s: arguments = %q", name, got)
		}
	}
}

func TestCollectorSnapshotsAreIndependent(t *testing.T) {
	collector := newCollector()
	apply := func(events ...Event) {
		t.Helper()
		for _, event := range events {
			if _, _, err := collector.Apply(event); err != nil {
				t.Fatal(err)
			}
		}
	}
	apply(BlockStart{Index: 0, Block: ReasoningBlock{State: testState(`{"id":1}`)}}, ReasoningDelta{Index: 0, Text: "a"})
	first := collector.Response()
	first.Blocks[0].(ReasoningBlock).State.Data[0] = '!'
	apply(ReasoningDelta{Index: 0, Text: "b"})
	second := collector.Response().Blocks[0].(ReasoningBlock)
	if first.Reasoning() != "a" || second.Text != "ab" || string(second.State.Data) != `{"id":1}` {
		t.Fatalf("snapshot aliases collector: first=%#v second=%#v", first.Blocks, second)
	}
}

func BenchmarkCollectorText(b *testing.B) {
	for _, tc := range []struct {
		name   string
		deltas int
	}{
		{name: "100_deltas", deltas: 100},
		{name: "1000_deltas", deltas: 1_000},
		{name: "5000_deltas", deltas: 5_000},
	} {
		b.Run(tc.name, func(b *testing.B) {
			for b.Loop() {
				collector := newCollector()
				_, _, _ = collector.Apply(BlockStart{Index: 0, Block: TextBlock{}})
				for range tc.deltas {
					_, _, _ = collector.Apply(TextDelta{Index: 0, Text: "x"})
				}
				_, _, _ = collector.Apply(BlockEnd{Index: 0})
				_, _, _ = collector.Apply(DoneEvent{})
				_ = collector.Response()
			}
		})
	}
}

func testState(data string) *ProviderState {
	return &ProviderState{Provider: "test", Data: json.RawMessage(data)}
}
