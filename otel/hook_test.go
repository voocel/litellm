package otel

import (
	"context"
	"testing"

	"github.com/voocel/litellm"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
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

func TestNonStreaming(t *testing.T) {
	h, rec := newTestHook(t)
	ctx := context.Background()
	meta := litellm.CallMeta{CallID: "c1", Provider: "openai", Model: "gpt-4", Operation: "chat"}
	req := &litellm.Request{Model: "gpt-4", Messages: []litellm.Message{litellm.UserText("hi")}}

	h.BeforeRequest(ctx, meta, req)
	h.AfterResponse(ctx, meta, &litellm.Response{
		Blocks:       []litellm.Block{litellm.TextBlock{Text: "hello"}},
		Model:        "gpt-4",
		Provider:     "openai",
		FinishReason: litellm.FinishReasonStop,
		Usage:        litellm.Usage{InputTokens: 10, OutputTokens: 5},
	}, nil)

	spans := rec.Ended()
	if len(spans) != 1 {
		t.Fatalf("want 1 ended span, got %d", len(spans))
	}
	a := attrMap(spans[0].Attributes())
	if got := a[attrRequestModel].AsString(); got != "gpt-4" {
		t.Fatalf("request model = %q", got)
	}
	if got := a[attrSystem].AsString(); got != "openai" {
		t.Fatalf("system = %q", got)
	}
	if got := a[attrFinishReason].AsString(); got != "stop" {
		t.Fatalf("finish_reason = %q", got)
	}
	if got := a[attrInputTokens].AsInt64(); got != 10 {
		t.Fatalf("input_tokens = %d", got)
	}
	if got := a[attrOutputTokens].AsInt64(); got != 5 {
		t.Fatalf("output_tokens = %d", got)
	}
	if got := a[attrCompletion].AsString(); got != "hello" {
		t.Fatalf("completion = %q", got)
	}
	if _, ok := a[attrPrompt]; !ok {
		t.Fatal("prompt attribute missing")
	}
}

func TestStreaming(t *testing.T) {
	h, rec := newTestHook(t)
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
	if got := a[attrCompletion].AsString(); got != "hello" {
		t.Fatalf("aggregated completion = %q", got)
	}
	if got := a[attrInputTokens].AsInt64(); got != 7 {
		t.Fatalf("input_tokens = %d", got)
	}
	if got := a[attrOutputTokens].AsInt64(); got != 3 {
		t.Fatalf("output_tokens = %d", got)
	}
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
}

func TestCaptureContentDisabled(t *testing.T) {
	h, rec := newTestHook(t, WithCaptureContent(false))
	ctx := context.Background()
	meta := litellm.CallMeta{CallID: "n1", Provider: "openai", Model: "gpt-4", Operation: "chat"}

	h.BeforeRequest(ctx, meta, &litellm.Request{Model: "gpt-4", Messages: []litellm.Message{litellm.UserText("secret")}})
	h.AfterResponse(ctx, meta, &litellm.Response{Blocks: []litellm.Block{litellm.TextBlock{Text: "private"}}, Usage: litellm.Usage{InputTokens: 1, OutputTokens: 1}}, nil)

	a := attrMap(rec.Ended()[0].Attributes())
	if _, ok := a[attrPrompt]; ok {
		t.Fatal("prompt recorded despite WithCaptureContent(false)")
	}
	if _, ok := a[attrCompletion]; ok {
		t.Fatal("completion recorded despite WithCaptureContent(false)")
	}
	// token usage must still be recorded — only content is suppressed
	if got := a[attrInputTokens].AsInt64(); got != 1 {
		t.Fatalf("input_tokens = %d", got)
	}
}

// TestStreamEndAbortClosesSpan covers a stream that aborts mid-flight (provider
// error / context cancel) with no final Done chunk: the span must still close,
// carry Error status, and keep whatever output streamed before the abort.
func TestStreamEndAbortClosesSpan(t *testing.T) {
	h, rec := newTestHook(t)
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
	if got := attrMap(spans[0].Attributes())[attrCompletion].AsString(); got != "partial" {
		t.Fatalf("partial completion = %q, want %q", got, "partial")
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
