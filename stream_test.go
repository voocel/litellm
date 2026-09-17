package litellm

import (
	"errors"
	"io"
	"strings"
	"testing"
)

func TestCollectPreservesBlockOrder(t *testing.T) {
	stream := &eventSliceStream{events: []Event{
		ContentDelta{Text: "first "},
		ReasoningDelta{Text: "think", Signature: "sig"},
		ContentDelta{Text: "second "},
		ToolUseStart{ID: "call_1", Name: "lookup"},
		ToolUseDelta{ID: "call_1", ArgumentsDelta: []byte(`{"q":"x"}`)},
		ContentDelta{Text: "third"},
		DoneEvent{FinishReason: FinishReasonToolCall, Provider: "test-provider", Model: "test-model"},
	}}

	resp, err := Collect(stream)
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}
	if len(resp.Blocks) != 5 {
		t.Fatalf("blocks len = %d, want 5: %#v", len(resp.Blocks), resp.Blocks)
	}
	if block, ok := resp.Blocks[0].(TextBlock); !ok || block.Text != "first " {
		t.Fatalf("blocks[0] = %#v", resp.Blocks[0])
	}
	if block, ok := resp.Blocks[1].(ReasoningBlock); !ok || block.Text != "think" || block.Signature != "sig" {
		t.Fatalf("blocks[1] = %#v", resp.Blocks[1])
	}
	if block, ok := resp.Blocks[2].(TextBlock); !ok || block.Text != "second " {
		t.Fatalf("blocks[2] = %#v", resp.Blocks[2])
	}
	if block, ok := resp.Blocks[3].(ToolUseBlock); !ok || block.ID != "call_1" || block.Name != "lookup" || string(block.Arguments) != `{"q":"x"}` {
		t.Fatalf("blocks[3] = %#v", resp.Blocks[3])
	}
	if block, ok := resp.Blocks[4].(TextBlock); !ok || block.Text != "third" {
		t.Fatalf("blocks[4] = %#v", resp.Blocks[4])
	}
	if resp.Provider != "test-provider" || resp.Model != "test-model" {
		t.Fatalf("provider/model = %q/%q", resp.Provider, resp.Model)
	}
}

func TestEventCollectorContinuesAfterResponseSnapshot(t *testing.T) {
	collector := NewEventCollector()
	for _, event := range []Event{
		ContentDelta{Text: "hel"},
		ContentDelta{Text: "lo"},
	} {
		if _, err := collector.Apply(event); err != nil {
			t.Fatalf("Apply(%T): %v", event, err)
		}
	}
	if got := collector.Response().Text(); got != "hello" {
		t.Fatalf("first snapshot text = %q, want %q", got, "hello")
	}

	for _, event := range []Event{
		ContentDelta{Text: " world"},
		ReasoningDelta{Text: "think"},
		ReasoningDelta{Text: "ing", Signature: "sig"},
	} {
		if _, err := collector.Apply(event); err != nil {
			t.Fatalf("Apply(%T): %v", event, err)
		}
	}
	resp := collector.Response()
	if got := resp.Text(); got != "hello world" {
		t.Fatalf("second snapshot text = %q, want %q", got, "hello world")
	}
	if got := resp.Reasoning(); got != "thinking" {
		t.Fatalf("second snapshot reasoning = %q, want %q", got, "thinking")
	}
	block, ok := resp.Blocks[1].(ReasoningBlock)
	if !ok || block.Signature != "sig" {
		t.Fatalf("reasoning block = %#v", resp.Blocks[1])
	}
}

func BenchmarkEventCollectorContent(b *testing.B) {
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
				collector := NewEventCollector()
				for range tc.deltas {
					_, _ = collector.Apply(ContentDelta{Text: "x"})
				}
				_, _ = collector.Apply(DoneEvent{})
				_ = collector.Response()
			}
		})
	}
}

func TestCollectMergesToolUseWhenStableIDArrivesAfterIndex(t *testing.T) {
	stream := &eventSliceStream{events: []Event{
		ToolUseStart{Name: "lookup", Index: IntPtr(0)},
		ToolUseDelta{Index: IntPtr(0), ArgumentsDelta: []byte(`{"q":`)},
		ToolUseDelta{ID: "call_1", Index: IntPtr(0), ArgumentsDelta: []byte(`"x"}`)},
		ToolUseDone{ID: "call_1", Index: IntPtr(0)},
		DoneEvent{FinishReason: FinishReasonToolCall, Provider: "test", Model: "m"},
	}}
	resp, err := Collect(stream)
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}
	calls := resp.ToolCalls()
	if len(calls) != 1 {
		t.Fatalf("tool calls len = %d, want 1: %#v", len(calls), calls)
	}
	if calls[0].ID != "call_1" || calls[0].Name != "lookup" || string(calls[0].Arguments) != `{"q":"x"}` {
		t.Fatalf("tool call = %#v", calls[0])
	}
}

func TestCollectSeparatesToolUseByOutputAndIndex(t *testing.T) {
	stream := &eventSliceStream{events: []Event{
		ToolUseStart{ID: "call_a", Name: "first", Index: IntPtr(0), OutputIndex: IntPtr(0)},
		ToolUseDelta{ID: "call_a", Index: IntPtr(0), OutputIndex: IntPtr(0), ArgumentsDelta: []byte(`{"a":1}`)},
		ToolUseStart{ID: "call_b", Name: "second", Index: IntPtr(0), OutputIndex: IntPtr(1)},
		ToolUseDelta{ID: "call_b", Index: IntPtr(0), OutputIndex: IntPtr(1), ArgumentsDelta: []byte(`{"b":2}`)},
		ToolUseDone{ID: "call_a", Index: IntPtr(0), OutputIndex: IntPtr(0)},
		ToolUseDone{ID: "call_b", Index: IntPtr(0), OutputIndex: IntPtr(1)},
		DoneEvent{FinishReason: FinishReasonToolCall, Provider: "test", Model: "m"},
	}}
	resp, err := Collect(stream)
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}
	calls := resp.ToolCalls()
	if len(calls) != 2 {
		t.Fatalf("tool calls len = %d, want 2: %#v", len(calls), calls)
	}
	if calls[0].ID != "call_a" || calls[0].Name != "first" || string(calls[0].Arguments) != `{"a":1}` {
		t.Fatalf("first call = %#v", calls[0])
	}
	if calls[1].ID != "call_b" || calls[1].Name != "second" || string(calls[1].Arguments) != `{"b":2}` {
		t.Fatalf("second call = %#v", calls[1])
	}
}

func TestCollectRejectsNilEventWithoutError(t *testing.T) {
	_, err := Collect(&eventSliceStream{events: []Event{nil}})
	if err == nil || err.Error() != "stream returned nil event without error" {
		t.Fatalf("expected nil event error, got %v", err)
	}
}

func TestCollectRequiresDoneOrError(t *testing.T) {
	_, err := Collect(&eventSliceStream{events: []Event{ContentDelta{Text: "partial"}}})
	if !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatalf("expected unexpected EOF when stream ends without Done or error, got %v", err)
	}
}

func TestCollectNormalizesInvalidToolArguments(t *testing.T) {
	resp, err := Collect(&eventSliceStream{events: []Event{
		ToolUseStart{ID: "call_1", Name: "lookup"},
		ToolUseDelta{ID: "call_1", ArgumentsDelta: []byte(`{"q":`)},
		DoneEvent{FinishReason: FinishReasonToolCall, Provider: "test", Model: "m"},
	}})
	if err != nil {
		t.Fatalf("Collect returned error: %v", err)
	}
	calls := resp.ToolCalls()
	if len(calls) != 1 {
		t.Fatalf("tool calls len = %d, want 1", len(calls))
	}
	if got := string(calls[0].Arguments); got != `{"q":` {
		t.Fatalf("arguments = %q, want raw malformed args", got)
	}
	if len(resp.Warnings) != 1 || resp.Warnings[0].Code != "stream.tool_arguments_invalid" {
		t.Fatalf("warnings = %+v", resp.Warnings)
	}
}

func TestCollectRejectsMissingProviderOrModel(t *testing.T) {
	_, err := Collect(&eventSliceStream{events: []Event{
		ContentDelta{Text: "ok"},
		DoneEvent{FinishReason: FinishReasonStop, Model: "m"},
	}})
	if err == nil || !strings.Contains(err.Error(), "missing provider") {
		t.Fatalf("expected missing provider error, got %v", err)
	}

	_, err = Collect(&eventSliceStream{events: []Event{
		ContentDelta{Text: "ok"},
		DoneEvent{FinishReason: FinishReasonStop, Provider: "test"},
	}})
	if err == nil || !strings.Contains(err.Error(), "missing model") {
		t.Fatalf("expected missing model error, got %v", err)
	}
}

func TestCollectRejectsToolCallMissingID(t *testing.T) {
	_, err := Collect(&eventSliceStream{events: []Event{
		ToolUseStart{Name: "lookup", Index: IntPtr(0)},
		ToolUseDelta{Index: IntPtr(0), ArgumentsDelta: []byte(`{"q":"x"}`)},
		DoneEvent{FinishReason: FinishReasonToolCall, Provider: "test", Model: "m"},
	}})
	if err == nil || err.Error() == "" {
		t.Fatalf("expected missing tool id error, got %v", err)
	}
}

func TestCollectRejectsToolCallMissingName(t *testing.T) {
	_, err := Collect(&eventSliceStream{events: []Event{
		ToolUseDelta{ID: "call_1", ArgumentsDelta: []byte(`{"q":"x"}`)},
		DoneEvent{FinishReason: FinishReasonToolCall, Provider: "test", Model: "m"},
	}})
	if err == nil || err.Error() == "" {
		t.Fatalf("expected missing tool name error, got %v", err)
	}
}

func TestCollectRejectsToolDoneMissingIDAndIndex(t *testing.T) {
	_, err := Collect(&eventSliceStream{events: []Event{
		ToolUseDone{},
	}})
	if err == nil || !strings.Contains(err.Error(), "missing id and index") {
		t.Fatalf("expected missing tool done id/index error, got %v", err)
	}
}

func TestCollectRejectsToolDoneUnknownToolUse(t *testing.T) {
	_, err := Collect(&eventSliceStream{events: []Event{
		ToolUseDone{ID: "call_missing"},
	}})
	if err == nil || !strings.Contains(err.Error(), "references unknown tool use") {
		t.Fatalf("expected unknown tool done error, got %v", err)
	}
	if strings.Contains(err.Error(), "missing id and index") {
		t.Fatalf("unknown tool use must not be reported as missing id/index: %v", err)
	}
}

func TestHandleInvokesCallbackPerEventAndAggregates(t *testing.T) {
	stream := &eventSliceStream{events: []Event{
		ContentDelta{Text: "a"},
		ReasoningDelta{Text: "r"},
		ContentDelta{Text: "b"},
		DoneEvent{FinishReason: FinishReasonStop, Provider: "test", Model: "m"},
	}}
	var seen int
	resp, err := Handle(stream, func(event Event) error {
		seen++
		return nil
	})
	if err != nil {
		t.Fatalf("Handle: %v", err)
	}
	if seen != 4 {
		t.Fatalf("callback invoked %d times, want 4", seen)
	}
	if resp.Text() != "ab" {
		t.Fatalf("aggregated text = %q, want %q", resp.Text(), "ab")
	}
}

func TestCollectStampsUsageProviderAndModelFromDoneEvent(t *testing.T) {
	resp, err := Collect(&eventSliceStream{events: []Event{
		ContentDelta{Text: "ok"},
		UsageEvent{Usage: Usage{InputTokens: IntPtr(1), OutputTokens: IntPtr(2), TotalTokens: IntPtr(3)}},
		DoneEvent{FinishReason: FinishReasonStop, Provider: "test-provider", Model: "test-model"},
	}})
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}
	if resp.Usage.Provider != "test-provider" || resp.Usage.Model != "test-model" {
		t.Fatalf("usage provider/model = %q/%q", resp.Usage.Provider, resp.Usage.Model)
	}
}

func TestHandleStopsOnCallbackError(t *testing.T) {
	boom := errors.New("boom")
	var seen int
	_, err := Handle(&eventSliceStream{events: []Event{
		ContentDelta{Text: "a"},
		ContentDelta{Text: "b"},
		DoneEvent{FinishReason: FinishReasonStop, Provider: "test", Model: "m"},
	}}, func(event Event) error {
		seen++
		return boom
	})
	if !errors.Is(err, boom) {
		t.Fatalf("err = %v, want boom", err)
	}
	if seen != 1 {
		t.Fatalf("callback invoked %d times, want 1 (stop on first error)", seen)
	}
}

func TestHandleTextReceivesOnlyContentDeltas(t *testing.T) {
	var text strings.Builder
	resp, err := HandleText(&eventSliceStream{events: []Event{
		ReasoningDelta{Text: "ignored"},
		ContentDelta{Text: "hello "},
		ContentDelta{Text: "world"},
		DoneEvent{FinishReason: FinishReasonStop, Provider: "test", Model: "m"},
	}}, func(s string) error {
		text.WriteString(s)
		return nil
	})
	if err != nil {
		t.Fatalf("HandleText: %v", err)
	}
	if text.String() != "hello world" {
		t.Fatalf("streamed text = %q, want %q", text.String(), "hello world")
	}
	if resp.Reasoning() != "ignored" {
		t.Fatalf("reasoning still aggregated, got %q", resp.Reasoning())
	}
}

func TestHandleWithSplitsReasoningAndContent(t *testing.T) {
	var reasoning, content strings.Builder
	resp, err := HandleWith(&eventSliceStream{events: []Event{
		ReasoningDelta{Text: "think "},
		ContentDelta{Text: "ans"},
		ReasoningDelta{Text: "more"},
		ContentDelta{Text: "wer"},
		DoneEvent{FinishReason: FinishReasonStop, Provider: "test", Model: "m"},
	}}, StreamHandler{
		Reasoning: func(s string) error { reasoning.WriteString(s); return nil },
		Content:   func(s string) error { content.WriteString(s); return nil },
	})
	if err != nil {
		t.Fatalf("HandleWith: %v", err)
	}
	if reasoning.String() != "think more" {
		t.Fatalf("reasoning = %q, want %q", reasoning.String(), "think more")
	}
	if content.String() != "answer" {
		t.Fatalf("content = %q, want %q", content.String(), "answer")
	}
	if resp.Text() != "answer" {
		t.Fatalf("aggregated text = %q", resp.Text())
	}
}

func TestHandleWithNilCallbacksAreSkipped(t *testing.T) {
	// Only Content is set; reasoning deltas must not panic and are still aggregated.
	var content strings.Builder
	resp, err := HandleWith(&eventSliceStream{events: []Event{
		ReasoningDelta{Text: "r"},
		ContentDelta{Text: "c"},
		DoneEvent{FinishReason: FinishReasonStop, Provider: "test", Model: "m"},
	}}, StreamHandler{
		Content: func(s string) error { content.WriteString(s); return nil },
	})
	if err != nil {
		t.Fatalf("HandleWith: %v", err)
	}
	if content.String() != "c" || resp.Reasoning() != "r" {
		t.Fatalf("content=%q reasoning=%q", content.String(), resp.Reasoning())
	}
}

func TestCollectPreservesRefusalAndMarksSafety(t *testing.T) {
	resp, err := Collect(&eventSliceStream{events: []Event{
		RefusalDelta{Text: "I can't help."},
		DoneEvent{FinishReason: FinishReasonStop, FinishReasonRaw: "completed", Provider: "test", Model: "m"},
	}})
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}
	if resp.Refusal != "I can't help." || resp.Text() != resp.Refusal {
		t.Fatalf("refusal/text = %q/%q", resp.Refusal, resp.Text())
	}
	if resp.FinishReason != FinishReasonSafety || resp.FinishReasonRaw != "completed" {
		t.Fatalf("finish/raw = %q/%q", resp.FinishReason, resp.FinishReasonRaw)
	}
}

type eventSliceStream struct {
	events []Event
	index  int
}

func (s *eventSliceStream) Next() (Event, error) {
	if s.index >= len(s.events) {
		return nil, io.EOF
	}
	event := s.events[s.index]
	s.index++
	return event, nil
}

func (s *eventSliceStream) Close() error {
	return nil
}

func TestCollectPreservesInterleavedContentIdentity(t *testing.T) {
	stream := &eventSliceStream{events: []Event{
		ContentDelta{Text: "a", OutputIndex: IntPtr(0), ContentIndex: IntPtr(0)},
		ContentDelta{Text: "b", OutputIndex: IntPtr(0), ContentIndex: IntPtr(1)},
		ReasoningDelta{Text: "think", OutputIndex: IntPtr(1), ContentIndex: IntPtr(0)},
		ContentDelta{Text: "c", OutputIndex: IntPtr(0), ContentIndex: IntPtr(0)},
		ReasoningDelta{Signature: "sig", OutputIndex: IntPtr(1), ContentIndex: IntPtr(0)},
		ContentDelta{Text: "d", OutputIndex: IntPtr(2), ContentIndex: IntPtr(0)},
		DoneEvent{Provider: "test", Model: "m", FinishReason: FinishReasonStop},
	}}
	resp, err := Collect(stream)
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Blocks) != 4 {
		t.Fatalf("blocks = %#v", resp.Blocks)
	}
	if resp.Blocks[0].(TextBlock).Text != "ac" || resp.Blocks[1].(TextBlock).Text != "b" || resp.Blocks[3].(TextBlock).Text != "d" {
		t.Fatalf("text blocks lost identity: %#v", resp.Blocks)
	}
	if block := resp.Blocks[2].(ReasoningBlock); block.Text != "think" || block.Signature != "sig" {
		t.Fatalf("reasoning lost identity: %#v", block)
	}
}

func TestCollectorSnapshotOwnsReasoningData(t *testing.T) {
	collector := NewEventCollector()
	_, err := collector.Apply(ReasoningDelta{Text: "a", ContentIndex: IntPtr(0), Extra: []byte(`{"id":1}`)})
	if err != nil {
		t.Fatal(err)
	}
	first := collector.Response()
	first.Blocks[0].(ReasoningBlock).Extra[0] = '!'
	_, err = collector.Apply(ReasoningDelta{Text: "b", ContentIndex: IntPtr(0)})
	if err != nil {
		t.Fatal(err)
	}
	second := collector.Response().Blocks[0].(ReasoningBlock)
	if first.Reasoning() != "a" || second.Text != "ab" || string(second.Extra) != `{"id":1}` {
		t.Fatalf("snapshot aliases collector: first=%#v second=%#v", first.Blocks, second)
	}
}

func TestHandleReturnsPartialResponse(t *testing.T) {
	boom := errors.New("interrupted")
	for _, tc := range []struct {
		name     string
		stream   Stream
		callback func(Event) error
		want     error
	}{
		{"truncated", &eventSliceStream{events: []Event{ContentDelta{Text: "partial"}}}, nil, io.ErrUnexpectedEOF},
		{"provider", &testStreamWithError{events: []Event{ContentDelta{Text: "partial"}}, err: boom}, nil, boom},
		{"callback", &eventSliceStream{events: []Event{ContentDelta{Text: "partial"}}}, func(Event) error { return boom }, boom},
	} {
		t.Run(tc.name, func(t *testing.T) {
			resp, err := Handle(tc.stream, tc.callback)
			if !errors.Is(err, tc.want) || resp == nil || resp.Text() != "partial" || resp.FinishReason != "" {
				t.Fatalf("response=%#v error=%v", resp, err)
			}
		})
	}
}

func TestCollectorRejectsEventsAfterCompletion(t *testing.T) {
	collector := NewEventCollector()
	if done, err := collector.Apply(DoneEvent{}); !done || err != nil {
		t.Fatalf("done=%v err=%v", done, err)
	}
	if _, err := collector.Apply(ContentDelta{Text: "late"}); err == nil {
		t.Fatal("accepted event after completion")
	}
	if collector.Response().Text() != "" {
		t.Fatal("late event changed completed response")
	}
}

func TestProviderStreamStopsAtTerminalEvent(t *testing.T) {
	for _, tc := range []struct {
		name   string
		events []Event
		want   error
	}{
		{"done", []Event{DoneEvent{}, ContentDelta{Text: "late"}}, nil},
		{"truncated", nil, io.ErrUnexpectedEOF},
	} {
		t.Run(tc.name, func(t *testing.T) {
			stream := newValidatedStream("test", "m", &eventSliceStream{events: tc.events})
			_, err := stream.Next()
			if !errors.Is(err, tc.want) {
				t.Fatalf("first error=%v want=%v", err, tc.want)
			}
			if event, err := stream.Next(); event != nil || err != io.EOF {
				t.Fatalf("after terminal: event=%#v err=%v", event, err)
			}
		})
	}
}
