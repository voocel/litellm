package otel

import (
	"context"
	"encoding/json"
	"reflect"
	"testing"

	"github.com/voocel/litellm"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace"
)

func newTestHook(t *testing.T, opts ...Option) (*OTelHook, *tracetest.SpanRecorder) {
	t.Helper()
	rec := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(rec))
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

func TestNonStreaming(t *testing.T) {
	h, rec := newTestHook(t, WithCaptureContent(true))
	ctx := context.Background()
	meta := litellm.CallMeta{CallID: "c1", Provider: "openai", Model: "gpt-4", Operation: "chat"}
	req := &litellm.Request{Model: "gpt-4", Messages: []litellm.Message{litellm.UserText("hi")}}

	h.BeforeRequest(ctx, meta, req)
	h.AfterResponse(ctx, meta, &litellm.Response{
		Blocks:       []litellm.Block{litellm.TextBlock{Text: "hello"}},
		Model:        "gpt-4",
		Provider:     "openai",
		FinishReason: litellm.FinishReasonStop,
		Usage: litellm.Usage{
			InputTokens:      10,
			OutputTokens:     5,
			ReasoningTokens:  2,
			CacheReadTokens:  3,
			CacheWriteTokens: 4,
		},
	}, nil)

	spans := rec.Ended()
	if len(spans) != 1 {
		t.Fatalf("want 1 ended span, got %d", len(spans))
	}
	a := attrMap(spans[0].Attributes())
	if got := a[attrRequestModel].AsString(); got != "gpt-4" {
		t.Fatalf("request model = %q", got)
	}
	if got := a[attrProviderName].AsString(); got != "openai" {
		t.Fatalf("provider name = %q", got)
	}
	if got := a[attrOperationName].AsString(); got != "chat" {
		t.Fatalf("operation name = %q", got)
	}
	if got := a[attrFinishReasons].AsStringSlice(); !reflect.DeepEqual(got, []string{"stop"}) {
		t.Fatalf("finish reasons = %v", got)
	}
	if got := a[attrInputTokens].AsInt64(); got != 10 {
		t.Fatalf("input_tokens = %d", got)
	}
	if got := a[attrOutputTokens].AsInt64(); got != 5 {
		t.Fatalf("output_tokens = %d", got)
	}
	if got := a[attrReasoningTokens].AsInt64(); got != 2 {
		t.Fatalf("reasoning output tokens = %d", got)
	}
	if got := a[attrCacheReadTokens].AsInt64(); got != 3 {
		t.Fatalf("cache read input tokens = %d", got)
	}
	if got := a[attrCacheWriteTokens].AsInt64(); got != 4 {
		t.Fatalf("cache creation input tokens = %d", got)
	}
	assertJSONEqual(t, a[attrInputMessages].AsString(), `[
		{"role":"user","parts":[{"type":"text","content":"hi"}]}
	]`)
	assertJSONEqual(t, a[attrOutputMessages].AsString(), `[
		{"role":"assistant","parts":[{"type":"text","content":"hello"}],"finish_reason":"stop"}
	]`)
	if got := spans[0].SpanKind(); got != trace.SpanKindClient {
		t.Fatalf("span kind = %v, want Client", got)
	}
	if got := spans[0].Name(); got != "chat gpt-4" {
		t.Fatalf("span name = %q", got)
	}
	for _, legacy := range []string{
		"gen_ai.system",
		"gen_ai.response.finish_reason",
		"gen_ai.usage.cache_read_input_tokens",
		"gen_ai.prompt",
		"gen_ai.completion",
	} {
		if _, ok := a[legacy]; ok {
			t.Errorf("legacy attribute %q is still emitted", legacy)
		}
	}
}

func TestStreaming(t *testing.T) {
	h, rec := newTestHook(t, WithCaptureContent(true))
	ctx := context.Background()
	meta := litellm.CallMeta{CallID: "s1", Provider: "anthropic", Model: "claude", Operation: "stream", Streaming: true}

	h.BeforeRequest(ctx, meta, &litellm.Request{Model: "claude", Messages: []litellm.Message{litellm.UserText("hi")}})
	// stream established: resp nil, no span yet
	h.AfterResponse(ctx, meta, nil, nil)
	if len(rec.Ended()) != 0 {
		t.Fatal("span ended too early on streaming")
	}
	h.OnStreamEvent(ctx, meta, litellm.ContentDelta{Text: "hel"})
	h.OnStreamEvent(ctx, meta, litellm.ContentDelta{Text: "lo"})
	h.OnStreamEvent(ctx, meta, litellm.UsageEvent{Usage: litellm.Usage{InputTokens: 7, OutputTokens: 3}})
	h.OnStreamEvent(ctx, meta, litellm.DoneEvent{FinishReason: litellm.FinishReasonStop})

	spans := rec.Ended()
	if len(spans) != 1 {
		t.Fatalf("want 1 ended span, got %d", len(spans))
	}
	a := attrMap(spans[0].Attributes())
	assertJSONEqual(t, a[attrOutputMessages].AsString(), `[
		{"role":"assistant","parts":[{"type":"text","content":"hello"}],"finish_reason":"stop"}
	]`)
	if got := a[attrInputTokens].AsInt64(); got != 7 {
		t.Fatalf("input_tokens = %d", got)
	}
	if got := a[attrOutputTokens].AsInt64(); got != 3 {
		t.Fatalf("output_tokens = %d", got)
	}
	if got := a[attrOperationName].AsString(); got != "chat" {
		t.Fatalf("streaming operation name = %q", got)
	}
	if got := a[attrRequestStream].AsBool(); !got {
		t.Fatal("gen_ai.request.stream is not true")
	}
}

func TestStreamingStructuredContent(t *testing.T) {
	h, rec := newTestHook(t, WithCaptureContent(true))
	ctx := context.Background()
	meta := litellm.CallMeta{CallID: "structured", Provider: "anthropic", Model: "claude", Operation: "stream", Streaming: true}

	h.BeforeRequest(ctx, meta, nil)
	h.OnStreamEvent(ctx, meta, litellm.ReasoningDelta{Text: "checking"})
	h.OnStreamEvent(ctx, meta, litellm.ToolUseStart{ID: "call_1", Name: "lookup"})
	h.OnStreamEvent(ctx, meta, litellm.ToolUseDelta{ID: "call_1", ArgumentsDelta: []byte(`{"q":"x"}`)})
	h.OnStreamEvent(ctx, meta, litellm.ToolUseDone{ID: "call_1"})
	h.OnStreamEvent(ctx, meta, litellm.DoneEvent{FinishReason: litellm.FinishReasonToolCall})

	a := attrMap(rec.Ended()[0].Attributes())
	assertJSONEqual(t, a[attrOutputMessages].AsString(), `[
		{"role":"assistant","parts":[
			{"type":"reasoning","content":"checking"},
			{"type":"tool_call","id":"call_1","name":"lookup","arguments":{"q":"x"}}
		],"finish_reason":"tool_call"}
	]`)
}

func TestStreamingRefusal(t *testing.T) {
	h, rec := newTestHook(t, WithCaptureContent(true))
	ctx := context.Background()
	meta := litellm.CallMeta{CallID: "refusal", Provider: "openai", Model: "gpt-4", Operation: "stream", Streaming: true}

	h.BeforeRequest(ctx, meta, nil)
	h.OnStreamEvent(ctx, meta, litellm.RefusalDelta{Text: "cannot comply"})
	h.OnStreamEvent(ctx, meta, litellm.DoneEvent{FinishReason: litellm.FinishReasonStop})

	a := attrMap(rec.Ended()[0].Attributes())
	if got := a[attrFinishReasons].AsStringSlice(); !reflect.DeepEqual(got, []string{"content_filter"}) {
		t.Fatalf("finish reasons = %v, want content_filter", got)
	}
	assertJSONEqual(t, a[attrOutputMessages].AsString(), `[
		{"role":"assistant","parts":[{"type":"text","content":"cannot comply"}],"finish_reason":"content_filter"}
	]`)
}

func TestErrorStatus(t *testing.T) {
	h, rec := newTestHook(t)
	ctx := context.Background()
	meta := litellm.CallMeta{CallID: "e1", Provider: "openai", Model: "gpt-4", Operation: "chat"}

	h.BeforeRequest(ctx, meta, nil)
	h.AfterResponse(ctx, meta, nil, context.DeadlineExceeded)

	spans := rec.Ended()
	if len(spans) != 1 {
		t.Fatalf("want 1 ended span, got %d", len(spans))
	}
	if spans[0].Status().Code != codes.Error {
		t.Fatalf("status = %v, want Error", spans[0].Status().Code)
	}
	if got := attrMap(spans[0].Attributes())[attrErrorType].AsString(); got != "timeout" {
		t.Fatalf("error.type = %q, want timeout", got)
	}
}

func TestCaptureContentDisabledByDefault(t *testing.T) {
	t.Run("non-streaming", func(t *testing.T) {
		h, rec := newTestHook(t)
		ctx := context.Background()
		meta := litellm.CallMeta{CallID: "n1", Provider: "openai", Model: "gpt-4", Operation: "chat"}

		h.BeforeRequest(ctx, meta, &litellm.Request{Model: "gpt-4", Messages: []litellm.Message{litellm.UserText("secret")}})
		h.AfterResponse(ctx, meta, &litellm.Response{Blocks: []litellm.Block{litellm.TextBlock{Text: "private"}}, Usage: litellm.Usage{InputTokens: 1, OutputTokens: 1}}, nil)

		a := attrMap(rec.Ended()[0].Attributes())
		assertContentNotCaptured(t, a)
		// Token usage remains available; only content is suppressed.
		if got := a[attrInputTokens].AsInt64(); got != 1 {
			t.Fatalf("input_tokens = %d", got)
		}
	})

	t.Run("streaming", func(t *testing.T) {
		h, rec := newTestHook(t)
		ctx := context.Background()
		meta := litellm.CallMeta{CallID: "n2", Provider: "openai", Model: "gpt-4", Operation: "stream", Streaming: true}

		h.BeforeRequest(ctx, meta, &litellm.Request{Model: "gpt-4", Messages: []litellm.Message{litellm.UserText("secret")}})
		h.OnStreamEvent(ctx, meta, litellm.ContentDelta{Text: "private"})
		h.OnStreamEvent(ctx, meta, litellm.DoneEvent{FinishReason: litellm.FinishReasonStop})

		assertContentNotCaptured(t, attrMap(rec.Ended()[0].Attributes()))
	})
}

func assertContentNotCaptured(t *testing.T, attributes map[string]attribute.Value) {
	t.Helper()
	if _, ok := attributes[attrInputMessages]; ok {
		t.Fatal("input messages recorded without WithCaptureContent(true)")
	}
	if _, ok := attributes[attrOutputMessages]; ok {
		t.Fatal("output messages recorded without WithCaptureContent(true)")
	}
}

// TestStreamEndAbortClosesSpan covers a stream that aborts mid-flight (provider
// error / context cancel) with no final Done chunk: the span must still close,
// carry Error status, and keep whatever output streamed before the abort.
func TestStreamEndAbortClosesSpan(t *testing.T) {
	h, rec := newTestHook(t, WithCaptureContent(true))
	ctx := context.Background()
	meta := litellm.CallMeta{CallID: "abort", Provider: "openai", Model: "gpt-4", Operation: "stream", Streaming: true}

	h.BeforeRequest(ctx, meta, &litellm.Request{Model: "gpt-4", Messages: []litellm.Message{litellm.UserText("hi")}})
	h.AfterResponse(ctx, meta, nil, nil) // stream established
	h.OnStreamEvent(ctx, meta, litellm.ContentDelta{Text: "par"})
	h.OnStreamEvent(ctx, meta, litellm.ContentDelta{Text: "tial"})
	h.OnStreamEnd(ctx, meta, context.DeadlineExceeded) // no Done chunk

	spans := rec.Ended()
	if len(spans) != 1 {
		t.Fatalf("want 1 ended span, got %d", len(spans))
	}
	if spans[0].Status().Code != codes.Error {
		t.Fatalf("status = %v, want Error", spans[0].Status().Code)
	}
	assertJSONEqual(t, attrMap(spans[0].Attributes())[attrOutputMessages].AsString(), `[
		{"role":"assistant","parts":[{"type":"text","content":"partial"}],"finish_reason":"error"}
	]`)
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
		meta litellm.CallMeta
		want string
	}{
		{meta: litellm.CallMeta{Provider: "openai", Operation: "chat"}, want: "chat"},
		{meta: litellm.CallMeta{Provider: "openai", Operation: "stream", Streaming: true}, want: "chat"},
		{meta: litellm.CallMeta{Provider: "gemini", Operation: "stream", Streaming: true}, want: "generate_content"},
	}
	for _, test := range tests {
		if got := semanticOperation(test.meta); got != test.want {
			t.Errorf("semanticOperation(%+v) = %q, want %q", test.meta, got, test.want)
		}
	}
}

// TestStreamEndCleanCloseClosesSpan covers a caller closing the stream early
// (err == nil) before a Done chunk: the span closes without Error status.
func TestStreamEndCleanCloseClosesSpan(t *testing.T) {
	h, rec := newTestHook(t)
	ctx := context.Background()
	meta := litellm.CallMeta{CallID: "early", Provider: "openai", Model: "gpt-4", Operation: "stream", Streaming: true}

	h.BeforeRequest(ctx, meta, nil)
	h.AfterResponse(ctx, meta, nil, nil)
	h.OnStreamEvent(ctx, meta, litellm.ContentDelta{Text: "hi"})
	h.OnStreamEnd(ctx, meta, nil)

	spans := rec.Ended()
	if len(spans) != 1 {
		t.Fatalf("want 1 ended span, got %d", len(spans))
	}
	if spans[0].Status().Code == codes.Error {
		t.Fatal("clean close must not mark the span as Error")
	}
}

// TestStreamEndAfterDoneIsNoop ensures the terminal event is harmless when the
// stream already finished through a Done chunk — no double-end, no extra span.
func TestStreamEndAfterDoneIsNoop(t *testing.T) {
	h, rec := newTestHook(t)
	ctx := context.Background()
	meta := litellm.CallMeta{CallID: "done", Provider: "anthropic", Model: "claude", Operation: "stream", Streaming: true}

	h.BeforeRequest(ctx, meta, nil)
	h.AfterResponse(ctx, meta, nil, nil)
	h.OnStreamEvent(ctx, meta, litellm.UsageEvent{Usage: litellm.Usage{InputTokens: 2, OutputTokens: 1}})
	h.OnStreamEvent(ctx, meta, litellm.DoneEvent{FinishReason: litellm.FinishReasonStop})
	h.OnStreamEnd(ctx, meta, nil) // late terminal: span already gone from the map

	if got := len(rec.Ended()); got != 1 {
		t.Fatalf("want exactly 1 ended span, got %d", got)
	}
}

// TestWithSpanAttributes verifies the resolver's attributes land on the
// generation span and that it can read per-call values from the context.
func TestWithSpanAttributes(t *testing.T) {
	type ctxKey struct{}
	resolver := func(ctx context.Context) []attribute.KeyValue {
		sid, _ := ctx.Value(ctxKey{}).(string)
		if sid == "" {
			return nil
		}
		return []attribute.KeyValue{attribute.String("langfuse.session.id", sid)}
	}
	h, rec := newTestHook(t, WithSpanAttributes(resolver))
	ctx := context.WithValue(context.Background(), ctxKey{}, "sess-42")
	meta := litellm.CallMeta{CallID: "c1", Provider: "openai", Model: "gpt-4", Operation: "chat"}

	h.BeforeRequest(ctx, meta, nil)
	h.AfterResponse(ctx, meta, &litellm.Response{Blocks: []litellm.Block{litellm.TextBlock{Text: "ok"}}}, nil)

	a := attrMap(rec.Ended()[0].Attributes())
	if got := a["langfuse.session.id"].AsString(); got != "sess-42" {
		t.Fatalf("langfuse.session.id = %q, want sess-42", got)
	}
}

// TestWithSpanAttributesNilResolverResult ensures a resolver returning nil adds
// nothing and does not break the span.
func TestWithSpanAttributesNilResolverResult(t *testing.T) {
	h, rec := newTestHook(t, WithSpanAttributes(func(context.Context) []attribute.KeyValue { return nil }))
	meta := litellm.CallMeta{CallID: "c2", Provider: "openai", Model: "gpt-4", Operation: "chat"}
	h.BeforeRequest(context.Background(), meta, nil)
	h.AfterResponse(context.Background(), meta, &litellm.Response{Blocks: []litellm.Block{litellm.TextBlock{Text: "ok"}}}, nil)

	a := attrMap(rec.Ended()[0].Attributes())
	if _, ok := a["langfuse.session.id"]; ok {
		t.Fatal("no session attribute should be set when the resolver returns nil")
	}
	if got := a[attrRequestModel].AsString(); got != "gpt-4" {
		t.Fatalf("base attributes must still be present, model = %q", got)
	}
}

// TestUnknownCallID ensures callbacks for an unstarted call are no-ops, never panics.
func TestUnknownCallID(t *testing.T) {
	h, rec := newTestHook(t)
	ctx := context.Background()
	meta := litellm.CallMeta{CallID: "ghost", Provider: "openai", Model: "gpt-4", Operation: "chat"}

	h.AfterResponse(ctx, meta, &litellm.Response{}, nil)
	h.OnStreamEvent(ctx, meta, litellm.DoneEvent{})
	h.OnStreamEnd(ctx, meta, context.Canceled)
	if len(rec.Ended()) != 0 {
		t.Fatal("no span should be created for unknown call id")
	}
}
