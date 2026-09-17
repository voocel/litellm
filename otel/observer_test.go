package otel

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"reflect"
	"sync"
	"testing"

	"github.com/voocel/litellm"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace"
)

func newTestObserver(t *testing.T, opts ...Option) (*Observer, *tracetest.SpanRecorder) {
	t.Helper()
	rec := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(rec))
	t.Cleanup(func() { _ = tp.Shutdown(context.Background()) })
	return New(tp.Tracer("test"), opts...), rec
}

func attrMap(kvs []attribute.KeyValue) map[string]attribute.Value {
	m := make(map[string]attribute.Value, len(kvs))
	for _, kv := range kvs {
		m[string(kv.Key)] = kv.Value
	}
	return m
}

func assertJSONEqual(t *testing.T, got, want string) {
	t.Helper()
	var gotValue, wantValue any
	if err := json.Unmarshal([]byte(got), &gotValue); err != nil {
		t.Fatalf("invalid JSON attribute %q: %v", got, err)
	}
	if err := json.Unmarshal([]byte(want), &wantValue); err != nil {
		t.Fatalf("invalid expected JSON %q: %v", want, err)
	}
	if !reflect.DeepEqual(gotValue, wantValue) {
		t.Fatalf("JSON attribute = %s, want %s", got, want)
	}
}

func TestSemanticConventionMessageEncoding(t *testing.T) {
	messages := []litellm.Message{
		litellm.System("be concise"),
		litellm.User(
			litellm.Text("weather?"),
			litellm.ImageURL("https://example.test/image.png"),
		),
		litellm.Assistant(
			litellm.ReasoningBlock{Text: "check weather"},
			litellm.ToolUseBlock{ID: "call_1", Name: "weather", Arguments: litellm.MustJSONRaw(map[string]any{"city": "Paris"})},
		),
		litellm.ToolResultText("call_1", "sunny"),
	}

	got, err := marshalInputMessages(messages)
	if err != nil {
		t.Fatalf("marshalInputMessages returned error: %v", err)
	}
	assertJSONEqual(t, got, `[
		{"role":"system","parts":[{"type":"text","content":"be concise"}]},
		{"role":"user","parts":[{"type":"text","content":"weather?"},{"type":"uri","modality":"image","uri":"https://example.test/image.png"}]},
		{"role":"assistant","parts":[{"type":"reasoning","content":"check weather"},{"type":"tool_call","id":"call_1","name":"weather","arguments":{"city":"Paris"}}]},
		{"role":"tool","parts":[{"type":"tool_call_response","id":"call_1","response":"sunny"}]}
	]`)

	got, err = marshalOutputMessages([]litellm.Block{
		litellm.ToolUseBlock{ID: "call_2", Name: "lookup", Arguments: litellm.MustJSONRaw(map[string]any{"q": "x"})},
	}, litellm.FinishReasonToolCall)
	if err != nil {
		t.Fatalf("marshalOutputMessages returned error: %v", err)
	}
	assertJSONEqual(t, got, `[
		{"role":"assistant","parts":[{"type":"tool_call","id":"call_2","name":"lookup","arguments":{"q":"x"}}],"finish_reason":"tool_call"}
	]`)
}

func TestSemanticProviderNames(t *testing.T) {
	tests := map[string]string{
		"bedrock":    "aws.bedrock",
		"gemini":     "gcp.gemini",
		"grok":       "x_ai",
		"openrouter": "openrouter",
	}
	for provider, want := range tests {
		if got := semanticProvider(provider); got != want {
			t.Errorf("semanticProvider(%q) = %q, want %q", provider, got, want)
		}
	}
}

func TestSemanticOperationNames(t *testing.T) {
	tests := []struct {
		meta litellm.CallInfo
		want string
	}{
		{meta: litellm.CallInfo{Provider: "openai", Operation: "chat"}, want: "chat"},
		{meta: litellm.CallInfo{Provider: "openai", Operation: "stream", Streaming: true}, want: "chat"},
		{meta: litellm.CallInfo{Provider: "gemini", Operation: "stream", Streaming: true}, want: "generate_content"},
	}
	for _, test := range tests {
		if got := semanticOperation(test.meta); got != test.want {
			t.Errorf("semanticOperation(%+v) = %q, want %q", test.meta, got, test.want)
		}
	}
}

type testProvider struct {
	chat   func(context.Context, *litellm.Request) (*litellm.Response, error)
	stream func(context.Context, *litellm.Request) (litellm.Stream, error)
}

func (p testProvider) Name() string { return "openai" }
func (p testProvider) Chat(ctx context.Context, r *litellm.Request) (*litellm.Response, error) {
	return p.chat(ctx, r)
}
func (p testProvider) Stream(ctx context.Context, r *litellm.Request) (litellm.Stream, error) {
	return p.stream(ctx, r)
}

type testStream struct {
	events []litellm.Event
	err    error
}

func (s *testStream) Next() (litellm.Event, error) {
	if len(s.events) > 0 {
		e := s.events[0]
		s.events = s.events[1:]
		return e, nil
	}
	if s.err != nil {
		return nil, s.err
	}
	return nil, io.EOF
}
func (s *testStream) Close() error { return nil }

func TestObserverContextPropagationAndContent(t *testing.T) {
	for _, streaming := range []bool{false, true} {
		for _, capture := range []bool{false, true} {
			observer, rec := newTestObserver(t, WithCaptureContent(capture))
			parentCtx, parent := observer.tracer.Start(context.Background(), "parent")
			checkContext := func(ctx context.Context) {
				span := trace.SpanFromContext(ctx)
				if !span.SpanContext().IsValid() || span.SpanContext().SpanID() == parent.SpanContext().SpanID() {
					t.Fatal("provider did not receive generation span")
				}
				_, child := observer.tracer.Start(ctx, "http")
				child.End()
			}
			usage := litellm.Usage{InputTokens: litellm.IntPtr(10), OutputTokens: litellm.IntPtr(5), ReasoningTokens: litellm.IntPtr(2), CacheReadTokens: litellm.IntPtr(3), CacheWriteTokens: litellm.IntPtr(4)}
			provider := testProvider{
				chat: func(ctx context.Context, _ *litellm.Request) (*litellm.Response, error) {
					checkContext(ctx)
					return &litellm.Response{Blocks: []litellm.Block{litellm.Text("hello")}, Usage: usage, FinishReason: litellm.FinishReasonStop}, nil
				},
				stream: func(ctx context.Context, _ *litellm.Request) (litellm.Stream, error) {
					checkContext(ctx)
					return &testStream{events: []litellm.Event{litellm.ContentStart{Block: litellm.TextBlock{Text: "hel"}, ContentIndex: litellm.IntPtr(0)}, litellm.ContentDelta{Text: "lo", ContentIndex: litellm.IntPtr(0)}, litellm.ContentEnd{ContentIndex: litellm.IntPtr(0)}, litellm.UsageEvent{Usage: usage}, litellm.DoneEvent{Provider: "openai", Model: "m", FinishReason: litellm.FinishReasonStop}}}, nil
				},
			}
			c, err := litellm.New(provider, litellm.WithObservers(observer))
			if err != nil {
				t.Fatal(err)
			}
			req := litellm.Request{Model: "m", Messages: []litellm.Message{litellm.UserText("private prompt")}}
			if streaming {
				s, err := c.Stream(parentCtx, req)
				if err != nil {
					t.Fatal(err)
				}
				if len(rec.Ended()) != 1 {
					t.Fatal("generation ended at stream creation")
				}
				if _, err = litellm.Collect(s); err != nil {
					t.Fatal(err)
				}
				_ = s.Close()
			} else {
				if _, err = c.Chat(parentCtx, req); err != nil {
					t.Fatal(err)
				}
			}
			parent.End()
			spans := rec.Ended()
			if len(spans) != 3 {
				t.Fatalf("spans=%d", len(spans))
			}
			child, generation := spans[0], spans[1]
			if child.Parent().SpanID() != generation.SpanContext().SpanID() || generation.Parent().SpanID() != parent.SpanContext().SpanID() {
				t.Fatal("broken parent → generation → HTTP trace")
			}
			attrs := attrMap(generation.Attributes())
			if attrs["litellm.call.status"].AsString() != "completed" || attrs[attrInputTokens].AsInt64() != 10 || attrs[attrOutputTokens].AsInt64() != 5 || attrs[attrReasoningTokens].AsInt64() != 2 || attrs[attrCacheReadTokens].AsInt64() != 3 || attrs[attrCacheWriteTokens].AsInt64() != 4 {
				t.Fatalf("attributes=%v", attrs)
			}
			if capture {
				assertJSONEqual(t, attrs[attrOutputMessages].AsString(), `[{"role":"assistant","parts":[{"type":"text","content":"hello"}],"finish_reason":"stop"}]`)
			} else {
				if _, ok := attrs[attrInputMessages]; ok {
					t.Fatal("recorded private input")
				}
				if _, ok := attrs[attrOutputMessages]; ok {
					t.Fatal("recorded private output")
				}
			}
		}
	}
}

func TestObserverPartialAndTerminalStatus(t *testing.T) {
	for _, kind := range []string{"error", "canceled", "closed", "protocol"} {
		t.Run(kind, func(t *testing.T) {
			observer, rec := newTestObserver(t, WithCaptureContent(true))
			p := testProvider{stream: func(context.Context, *litellm.Request) (litellm.Stream, error) {
				s := &testStream{events: []litellm.Event{litellm.ContentDelta{Text: "partial"}}, err: errors.New("provider failed")}
				if kind == "canceled" {
					s.err = context.Canceled
				}
				if kind == "protocol" {
					s.events = []litellm.Event{litellm.ContentStart{Block: litellm.TextBlock{Text: "partial"}, ContentIndex: litellm.IntPtr(0)}, litellm.DoneEvent{Provider: "openai", Model: "m"}}
					s.err = nil
				}
				return s, nil
			}}
			c, _ := litellm.New(p, litellm.WithObservers(observer))
			s, err := c.Stream(context.Background(), litellm.Request{Model: "m", Messages: []litellm.Message{litellm.UserText("hi")}})
			if err != nil {
				t.Fatal(err)
			}
			if kind == "closed" {
				_, err = s.Next()
				if err != nil {
					t.Fatal(err)
				}
			} else {
				_, err = litellm.Collect(s)
				if err == nil {
					t.Fatal("expected failure")
				}
			}
			_ = s.Close()
			_ = s.Close()
			spans := rec.Ended()
			if len(spans) != 1 {
				t.Fatalf("spans=%d", len(spans))
			}
			want := "failed"
			if kind == "closed" || kind == "canceled" {
				want = kind
			}
			attrs := attrMap(spans[0].Attributes())
			if attrs["litellm.call.status"].AsString() != want {
				t.Fatalf("status=%v", attrs)
			}
			if (spans[0].Status().Code == codes.Error) != (kind != "closed") {
				t.Fatalf("trace status=%v", spans[0].Status())
			}
			assertJSONEqual(t, attrs[attrOutputMessages].AsString(), `[{"role":"assistant","parts":[{"type":"text","content":"partial"}],"finish_reason":"unknown"}]`)
		})
	}
}

func TestObserverConcurrentCallsAndAttributes(t *testing.T) {
	type ctxKey struct{}
	observer, rec := newTestObserver(t, WithSpanAttributes(func(ctx context.Context) []attribute.KeyValue {
		return []attribute.KeyValue{attribute.Int("request.index", ctx.Value(ctxKey{}).(int))}
	}))
	provider := testProvider{chat: func(ctx context.Context, _ *litellm.Request) (*litellm.Response, error) {
		return &litellm.Response{Blocks: []litellm.Block{litellm.Text("ok")}}, nil
	}}
	client, _ := litellm.New(provider, litellm.WithObservers(observer))
	var wg sync.WaitGroup
	for i := range 20 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, err := client.Chat(context.WithValue(context.Background(), ctxKey{}, i), litellm.Request{Model: "m", Messages: []litellm.Message{litellm.UserText("hi")}})
			if err != nil {
				t.Error(err)
			}
		}()
	}
	wg.Wait()
	spans := rec.Ended()
	if len(spans) != 20 {
		t.Fatalf("spans=%d", len(spans))
	}
	seen := make(map[int64]bool)
	for _, span := range spans {
		index := attrMap(span.Attributes())["request.index"].AsInt64()
		if seen[index] {
			t.Fatal("shared call state")
		}
		seen[index] = true
	}
}

func TestObserverPanicIsolation(t *testing.T) {
	observer, rec := newTestObserver(t, WithSpanAttributes(func(context.Context) []attribute.KeyValue { panic("resolver") }))
	ctx := context.Background()
	next, call := observer.Start(ctx, litellm.CallInfo{Model: "m"})
	if next != ctx || call != nil || len(rec.Ended()) != 0 {
		t.Fatal("panic corrupted context or leaked observation")
	}
}

func TestUsageUnknownIsOmittedAndZeroIsRecorded(t *testing.T) {
	observer, rec := newTestObserver(t)
	_, call := observer.Start(context.Background(), litellm.CallInfo{Provider: "test", Model: "m", Operation: "chat"})
	call.End(litellm.CallResult{Status: litellm.CallCompleted, Response: &litellm.Response{Model: "m", Usage: litellm.Usage{InputTokens: litellm.IntPtr(0)}}})
	spans := rec.Ended()
	if len(spans) != 1 {
		t.Fatalf("spans = %d", len(spans))
	}
	attrs := attrMap(spans[0].Attributes())
	if value, ok := attrs[attrInputTokens]; !ok || value.AsInt64() != 0 {
		t.Fatal("known zero was omitted")
	}
	if _, ok := attrs[attrOutputTokens]; ok {
		t.Fatal("unknown output recorded as zero")
	}
	if _, ok := attrs[attrCacheReadTokens]; ok {
		t.Fatal("unknown cache recorded as zero")
	}
}
